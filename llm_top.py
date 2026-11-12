#!/usr/bin/env python3
"""
LLMtop — htop for your LLM Mac.

  * CPU / GPU / RAM temps + power + fan + memory pressure (via `macmon pipe`, no sudo)
  * GPU utilization 0-100% history chart (macmon gpu_active_ratio)
  * Loaded models: oMLX (/admin/api), Ollama (/api/ps), LM Studio (/api/v0/models)
  * Prompt processing + generation tok/s: oMLX session stats AND/OR capture proxy
  * System health: RAM, swap, battery, thermal state, disk, load, uptime
  * htop-style process table (q=quit  r=sort  p=pause  k=kill top)

Usage:
  python llm_top.py                      # monitor everything found
  python llm_top.py --once               # print ONE snapshot and exit (testing)
  python llm_top.py --proxy              # + capture proxy :11500 -> Ollama :11434
  python llm_top.py --proxy --upstream http://127.0.0.1:8000   # proxy -> oMLX
  (oMLX key: export OMLX_API_KEY=... or pass --omlx-key)
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import psutil
import requests
from rich import box
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

__version__ = "0.1.1"

console = Console()
GB = 1024 ** 3
MB = 1024 ** 2
TOP_PROC = [None]  # (pid, name) of the first row in the process table


# ------------------------------ formatting ---------------------------------
def fmt_bytes(n) -> str:
    if n is None:
        return "—"
    if n >= GB:
        return f"{n / GB:6.2f}G"
    if n >= MB:
        return f"{n / MB:6.0f}M"
    return f"{n / 1024:6.0f}K"


def fmt_gib(n) -> str:
    return "—" if n is None else f"{n / GB:.1f}G"


def bar(pct: float, width: int = 22) -> Text:
    pct = max(0.0, min(100.0, pct))
    n = int(pct / 100 * width)
    color = "red" if pct > 90 else "yellow" if pct > 70 else "green"
    return Text("█" * n + "░" * (width - n), style=color)


def temp_style(t) -> str:
    if t is None:
        return "dim"
    return "red" if t > 85 else "yellow" if t > 70 else "green"


def spark(vals, blocks="▁▂▃▄▅▆▇█") -> str:
    return "".join(blocks[min(len(blocks) - 1, int(v / 100 * len(blocks)))] for v in vals)


def area_chart(values, width=44, height=4, style="yellow"):
    """0-100% filled area chart (newest sample on the right)."""
    vals = [max(0.0, min(100.0, float(v))) for v in values][-width:]
    pad = width - len(vals)
    renderables = []
    for r in range(height, 0, -1):
        line = Text()
        line.append(f"{r * 100 // height:>3} ┤ ", style="dim")
        line.append(" " * pad, style="dim")
        for v in vals:
            on = v >= (r - 1) * (100.0 / height)
            line.append("█" if on else " ", style=style if on else "dim")
        renderables.append(line)
    base = Text()
    base.append("    └", style="dim")
    base.append("─" * width, style="dim")
    base.append(" now", style="dim")
    renderables.append(base)
    return Group(*renderables)


# ------------------- macmon bridge (brew install macmon) --------------------
def _flat(obj, prefix=""):
    """Flatten macmon JSON → look up keys like 'cputempcelsius'."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}_{k}" if prefix else str(k)
            norm = re.sub(r"[^a-z0-9]", "", key.lower())
            if isinstance(v, (dict, list)):
                out.update(_flat(v, key))
            else:
                out[norm] = v
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.update(_flat(v, f"{prefix}{i}"))
    return out


class Macmon:
    def __init__(self):
        self.proc = None
        self.latest: dict = {}
        self.gpu_hist: deque = deque(maxlen=44)
        self.ok = bool(shutil.which("macmon"))
        if self.ok:
            try:
                self.proc = subprocess.Popen(
                    ["macmon", "pipe", "-i", "1000"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            except Exception:
                self.ok = False
            if self.ok:
                threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        dec = json.JSONDecoder()
        buf = ""
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            buf += line
            while True:
                s = buf.lstrip()
                if not s:
                    buf = ""
                    break
                if s[0] != "{":            # skip non-JSON prefix (e.g. a header line)
                    cut = s.find("{")
                    buf = s[cut:] if cut >= 0 else ""
                    continue
                try:
                    obj, end = dec.raw_decode(s)
                except ValueError:
                    break                  # incomplete object — wait for more data
                self.latest = _flat(obj)
                buf = s[end:]

    def g(self, *names):
        for n in names:
            if n in self.latest:
                return self.latest[n]
        return None

    def stop(self):
        if self.proc:
            try:
                self.proc.terminate()
            except Exception:
                pass


# ------------------- macOS health fallbacks (no sudo) ------------------------
def battery_temp() -> float | None:
    try:
        out = subprocess.check_output(["ioreg", "-rc", "AppleSmartBattery"], text=True, timeout=2)
        m = re.search(r'"Temperature"\s*=\s*(\d+)', out)
        return int(m.group(1)) / 1000.0 if m else None
    except Exception:
        return None


def thermal_state() -> str:
    try:
        out = subprocess.check_output(["pmset", "-g", "therm"], text=True, timeout=2)
    except Exception:
        return "n/a"
    m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", out)
    if m and int(m.group(1)) < 100:
        return "THROTTLED"
    for line in out.splitlines():
        if "warning level" in line and not line.startswith("Note: No"):
            return "warning"
    return "nominal"


# ------------------- LLM stats (fed by the proxy) ---------------------------
@dataclass
class ReqStats:
    ts: float
    model: str
    endpoint: str
    prompt_tokens: int | None
    completion_tokens: int | None
    prompt_tps: float | None
    gen_tps: float | None
    ttft_s: float | None
    total_s: float | None


class LLMStats:
    def __init__(self):
        self.reqs: deque[ReqStats] = deque(maxlen=300)
        self._lock = threading.Lock()

    def add(self, rs: ReqStats):
        with self._lock:
            self.reqs.append(rs)

    def window(self, secs: float = 60.0):
        now = time.time()
        with self._lock:
            return [r for r in self.reqs if now - r.ts <= secs]


# ------------------- capture proxy (OpenAI/Ollama/oMLX compatible) -----------
def _has_stats(obj) -> bool:
    return isinstance(obj, dict) and bool(
        obj.get("usage") or obj.get("eval_count") is not None
        or obj.get("prompt_eval_count") is not None)


def make_handler(upstream: str, stats: LLMStats):
    base = urlsplit(upstream)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send_chunk(self, data: bytes):
            if data:
                self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")

        def _passthrough_headers(self, resp):
            skip = {"transfer-encoding", "connection", "content-encoding", "content-length"}
            for k, v in resp.headers.items():
                if k.lower() not in skip:
                    self.send_header(k, v)

        def _record(self, obj, t0, t1, first_byte):
            if not isinstance(obj, dict):
                return
            model = obj.get("model") or "?"
            ttft = (first_byte - t0) if first_byte else None
            if obj.get("prompt_eval_count") is not None or obj.get("eval_count") is not None:
                # Ollama-style: counts + durations in nanoseconds
                pc, ec = obj.get("prompt_eval_count"), obj.get("eval_count")
                pd, ed = obj.get("prompt_eval_duration"), obj.get("eval_duration")
                stats.add(ReqStats(
                    ts=time.time(), model=model, endpoint=self.path.split("?")[0],
                    prompt_tokens=pc, completion_tokens=ec,
                    prompt_tps=(pc / (pd / 1e9)) if pc and pd else None,
                    gen_tps=(ec / (ed / 1e9)) if ec and ed else None,
                    ttft_s=ttft, total_s=t1 - t0))
            else:
                # OpenAI / Anthropic / LM Studio / oMLX style
                usage = obj.get("usage") or {}
                st = obj.get("stats") or {}
                pt = usage.get("prompt_tokens") or usage.get("input_tokens")
                ct = usage.get("completion_tokens") or usage.get("output_tokens")
                tt = usage.get("time_to_first_token") or st.get("time_to_first_token")
                if tt is None:
                    tt = (first_byte - t0) if first_byte else None
                gen_tps = (st.get("tokens_per_second")
                           or usage.get("generation_tokens_per_second"))
                pr_tps = (usage.get("prompt_tokens_per_second")
                          or ((pt / tt) if pt and tt else None))
                stats.add(ReqStats(
                    ts=time.time(), model=model, endpoint=self.path.split("?")[0],
                    prompt_tokens=pt, completion_tokens=ct,
                    prompt_tps=pr_tps, gen_tps=gen_tps,
                    ttft_s=tt, total_s=t1 - t0))

        def _handle(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            fwd = {k: v for k, v in self.headers.items()
                   if k.lower() not in ("host", "content-length", "connection")}
            fwd["Host"] = base.netloc
            url = f"{base.scheme}://{base.netloc}{self.path}"

            t0 = time.time()
            try:
                resp = requests.request(self.command, url, data=body, headers=fwd,
                                        stream=True, timeout=(5, 600), allow_redirects=False)
            except requests.RequestException:
                self.send_response(502)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            try:
                self._relay(resp, t0)
            except (OSError, requests.RequestException):
                self.close_connection = True  # client or upstream went away mid-stream
            finally:
                resp.close()

        def _relay(self, resp, t0):
            is_gen = any(p in self.path.lower()
                         for p in ("/api/generate", "/api/chat", "/completions",
                                   "/messages", "/chat/"))
            ctype = (resp.headers.get("Content-Type") or "").lower()
            sse = "event-stream" in ctype
            streamed = sse or "ndjson" in ctype

            self.send_response(resp.status_code)
            self._passthrough_headers(resp)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            first_byte = None
            if streamed:
                last_obj = None
                for raw in resp.iter_lines():
                    if not raw and not sse:
                        continue
                    if first_byte is None and raw:
                        first_byte = time.time()
                    self.wfile.write(b"%X\r\n%s\n\r\n" % (len(raw) + 1, raw))  # blank lines delimit SSE events
                    self.wfile.flush()
                    payload = raw[5:].strip() if raw.startswith(b"data:") else raw
                    if not payload.startswith(b"{"):
                        continue
                    try:
                        obj = json.loads(payload)
                    except ValueError:
                        continue
                    # usage often arrives before the final event — keep the last one that has it
                    if _has_stats(obj) or not _has_stats(last_obj):
                        last_obj = obj
                self.wfile.write(b"0\r\n\r\n")
                t1 = time.time()
                if is_gen and last_obj:
                    self._record(last_obj, t0, t1, first_byte)
            else:
                data = resp.content
                first_byte = time.time()
                self._send_chunk(data)
                self.wfile.write(b"0\r\n\r\n")
                t1 = time.time()
                if is_gen:
                    try:
                        self._record(json.loads(data), t0, t1, first_byte)
                    except Exception:
                        pass

        do_GET = do_POST = do_PUT = do_DELETE = do_PATCH = _handle

    return Handler


class Proxy(threading.Thread):
    def __init__(self, port, upstream, stats):
        super().__init__(daemon=True)
        # bind here (not in run) so a busy port fails loudly at startup
        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(upstream, stats))

    def run(self):
        self.httpd.serve_forever()

    def stop(self):
        self.httpd.shutdown()


# ------------------- oMLX bridge (/admin/api — no proxy needed) --------------
def _dig(obj, *names):
    """Recursively find the first numeric value under any of these key names."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k.lower() in names and isinstance(v, (int, float)):
                return v
        for v in obj.values():
            r = _dig(v, *names)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = _dig(v, *names)
            if r is not None:
                return r
    return None


class Omlx:
    def __init__(self, base, key=None):
        self.base = base.rstrip("/")
        self.key = key
        self.prev = None  # (ts, prompt_tokens, completion_tokens)
        self.avg_gen = None
        self.avg_pr = None

    def _get(self, path):
        h = {"Authorization": f"Bearer {self.key}"} if self.key else {}
        r = requests.get(self.base + path, headers=h, timeout=2)
        r.raise_for_status()
        return r.json()

    def sample(self):
        """-> (gen_tps, prompt_tps, models_rows, live_rows) or None when unreachable"""
        ok = False
        gen_tps = prompt_tps = None
        models_rows, live_rows = [], []
        size_map = {}
        try:
            s = self._get("/admin/api/stats?scope=session")
            ok = True
            self.avg_pr = _dig(s, "avg_prefill_tps")
            self.avg_gen = _dig(s, "avg_generation_tps")
            pt = _dig(s, "total_prompt_tokens", "prompt_tokens") or 0
            ct = _dig(s, "total_completion_tokens", "completion_tokens") or 0
            now = time.time()
            if self.prev and now - self.prev[0] > 0.5:
                dt = now - self.prev[0]
                dct, dpt = ct - self.prev[2], pt - self.prev[1]
                if dct > 0:
                    g = dct / dt
                    if self.avg_gen is None or g <= 4 * self.avg_gen:
                        gen_tps = g
                if dpt > 0:
                    p = dpt / dt
                    if self.avg_pr is None or p <= 4 * self.avg_pr:
                        prompt_tps = p
            self.prev = (now, pt, ct)
            am = s.get("active_models") or {}
            for m in am.get("models") or []:
                if not isinstance(m, dict):
                    continue
                mid = str(m.get("id") or "?")
                size_map[mid] = (m.get("actual_size_formatted")
                                 or m.get("estimated_size_formatted"))
                for state in ("waiting", "prefilling", "generating"):
                    for r in m.get(state) or []:
                        rid = (r.get("request_id") or r.get("id")) if isinstance(r, dict) else str(r)
                        el = r.get("elapsed_seconds") if isinstance(r, dict) else ""
                        if isinstance(el, (int, float)):
                            el = f"{el:.1f}s"
                        live_rows.append((mid[:24], str(rid or "")[:12], state, el))
        except Exception:
            pass
        try:
            a = self._get("/admin/api/activity")
            ok = True
            if not live_rows:
                rows = a if isinstance(a, list) else (a.get("activity") or a.get("requests") or [])
                for r in rows:
                    if isinstance(r, dict):
                        live_rows.append((
                            str(r.get("model") or r.get("model_id") or "?")[:24],
                            str(r.get("request_id") or r.get("id") or "")[:12],
                            str(r.get("state") or r.get("status") or "?")[:10],
                            r.get("elapsed") or r.get("elapsed_s") or ""))
        except Exception:
            pass
        try:
            m = self._get("/admin/api/models")
            ok = True
            items = m if isinstance(m, list) else (m.get("models") or m.get("data") or [])
            for x in items:
                if not isinstance(x, dict):
                    continue
                name = str(x.get("id") or x.get("name") or "?")
                loaded = bool(x.get("loaded") or x.get("is_loaded")
                              or x.get("state") == "loaded")
                sz = (x.get("size_formatted") or x.get("actual_size_formatted")
                      or size_map.get(name) or "")
                models_rows.append(("oMLX", name[:40], sz, None,
                                    x.get("max_context_length"),
                                    "LOADED" if loaded else ""))
        except Exception:
            pass
        # counters may only commit when a request completes — fall back to the
        # server-reported session averages so the line is never blank
        if gen_tps is None:
            gen_tps = self.avg_gen
        if prompt_tps is None:
            prompt_tps = self.avg_pr
        if not ok:
            return None
        return gen_tps, prompt_tps, models_rows, live_rows


# ------------------- Ollama / LM Studio polling -----------------------------
def poll_models(ollama, lmstudio):
    rows = []
    if ollama:
        try:
            r = requests.get(ollama.rstrip("/") + "/api/ps", timeout=1.5).json()
            for m in r.get("models", []):
                d = m.get("details", {})
                rows.append(("Ollama", m.get("name", "?"),
                             f'{d.get("parameter_size", "?")} {d.get("quantization_level", "")}'.strip(),
                             m.get("size_vram"), m.get("context_length"),
                             (m.get("expires_at") or "")[:19]))
        except Exception:
            rows.append(("Ollama", "unreachable", "", None, None, ""))
    if lmstudio:
        try:
            r = requests.get(lmstudio.rstrip("/") + "/api/v0/models", timeout=1.5).json()
            for m in r.get("data", []):
                if m.get("state") == "loaded":
                    rows.append(("LM Studio", m.get("id", "?"), m.get("quantization", ""),
                                 None, m.get("max_context_length"), ""))
        except Exception:
            rows.append(("LM Studio", "unreachable", "", None, None, ""))
    return rows


# ------------------------------ panels ---------------------------------------
def header(cfg, proxy_up: bool) -> Panel:
    load = os.getloadavg()
    up = time.time() - psutil.boot_time()
    h, m = int(up // 3600), int(up % 3600 // 60)
    disk = psutil.disk_usage("/")
    th = thermal_state()
    txt = Text.assemble(
        ("  LLMtop ", "bold bright_cyan"),
        (f"  {datetime.now():%H:%M:%S}  ", "dim"),
        ("thermal ", "dim"), (f" {th}", "red" if th == "THROTTLED" else "green"),
        (f"  load {load[0]:.2f} {load[1]:.2f} {load[2]:.2f}  ", "dim"),
        (f"up {h}h{m:02d}m  ", "dim"),
        (f"disk {disk.used / GB:.0f}G/{disk.total / GB:.0f}G ({disk.percent:.0f}%)  ", "dim"),
        (f"procs {len(psutil.pids())}  ", "dim"),
        ("proxy ", "dim"),
        (f"http://127.0.0.1:{cfg.proxy_port} → {cfg.upstream}" if proxy_up else "off",
         "green" if proxy_up else "dim"))
    return Panel(txt, border_style="cyan")


def sample_cpu() -> float:
    cores = psutil.cpu_percent(None, percpu=True)
    return sum(cores) / len(cores) if cores else 0.0


def cpu_panel(mm: Macmon, hist: deque) -> Panel:
    pct = hist[-1] if hist else 0.0
    cores = psutil.cpu_percent(None, percpu=True)
    t = mm.g("tempcputempavg", "cputempcelsius", "cputemp")
    p = mm.g("cpupower", "cpupowerwatts")
    lines = [
        Text.assemble(("CPU ", "bold"), f"{pct:5.1f}%", ("  spark ", "dim"), (spark(hist), "cyan")),
        Text.assemble(("temp ", "dim"),
                      (f"{t:5.1f}°C" if t is not None else "  —  ", temp_style(t)),
                      ("   power ", "dim"),
                      (f"{p:5.2f}W" if p is not None else "  —  ", "cyan")),
        bar(pct),
    ]
    core_rows = []
    for i in range(0, len(cores), 2):
        row = Text()
        for j, c in enumerate(cores[i:i + 2]):
            row.append(f"C{i + j:02d} ", style="dim")
            row.append("█" * int(c / 10) + "░" * (10 - int(c / 10)),
                       style="dim" if c < 30 else ("yellow" if c < 70 else "red"))
            row.append("   ", style="dim")
        core_rows.append(row)
    try:
        freq = psutil.cpu_freq()
        if freq and freq.current:
            lines.append(Text(f"freq {freq.current:.0f} MHz", style="dim"))
    except (SystemError, ValueError, OSError):
        pass  # psutil cpu_freq is broken on some Apple Silicon Macs
    return Panel(Group(*lines, *core_rows), title=" CPU ", border_style="blue")


def gpu_panel(mm: Macmon) -> Panel:
    if not mm.ok:
        return Panel(Text("install macmon for GPU metrics\n$ brew install macmon", style="dim"),
                     title=" GPU ", border_style="magenta")
    t = mm.g("tempgputempavg", "gputempcelsius", "gputemp")
    p = mm.g("gpupower", "gpupowerwatts")
    chip = mm.g("chippower", "allpower", "chippowerwatts")
    ane = mm.g("anepower")
    ma = mm.g("gpumemalloc", "gpumemoryallocbytes", "gpumemoryalloc")
    mu = mm.g("gpumeminuse", "gpumemoryinusebytes", "gpumemoryinuse")
    fan = mm.g("fans0rpm", "fandspeedrpm")
    ratio = mm.g("gpuactiveratio", "gpuscaledratio")
    use = (ratio * 100.0) if ratio is not None else None
    mm.gpu_hist.append(use if use is not None else 0.0)
    items = [
        Text.assemble(("temp  ", "dim"),
                      (f"{t:6.1f}°C" if t is not None else "   —   ", temp_style(t)),
                      ("  power ", "dim"), (f"{p:6.2f}W" if p is not None else "   —   ", "cyan")),
        Text.assemble(("chip  ", "dim"),
                      (f"{chip:6.2f}W" if chip is not None else "   —   ", "magenta"),
                      ("  ane ", "dim"), (f"{ane:4.2f}W" if ane is not None else "   —  ", "magenta"),
                      ("  fan ", "dim"), (f"{fan:5.0f}rpm" if fan else "   —   ", "cyan")),
        Text.assemble(("use   ", "dim"),
                      (f"{use:5.1f}%" if use is not None else "   —  ",
                       "red" if use is not None and use > 90 else "yellow")),
    ]
    if ma is not None or mu is not None:
        items.append(Text.assemble(
            ("vram  ", "dim"), (f"alloc {fmt_gib(ma)}" if ma is not None else "   —", "magenta"),
            (("   in-use " + fmt_gib(mu)) if mu is not None else "", "cyan")))
    items.append(Text("GPU use, last 44s  (0-100%)", style="dim"))
    items.append(area_chart(mm.gpu_hist, style="yellow"))
    return Panel(Group(*items), title=" GPU (Metal) ", border_style="magenta")


def ram_panel(mm: Macmon) -> Panel:
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    bt = battery_temp()
    press = mm.g("memorypressurepercent")
    try:
        batt = psutil.sensors_battery()
        batt_s = f"{batt.percent:.0f}%{'⚡' if batt.power_plugged else ''}" if batt else ""
    except Exception:
        batt_s = ""
    items = [
        Text("RAM ", style="bold") + bar(vm.percent) + Text(f" {vm.percent:4.1f}%"),
        Text(f"used {vm.used / GB:.1f}G / {vm.total / GB:.1f}G   avail {vm.available / GB:.1f}G",
             style="dim"),
        Text.assemble(("swap ", "dim"),
                      (f"{sw.used / GB:.2f}G / {sw.total / GB:.2f}G",
                       "magenta" if sw.used > GB else "dim")),
        Text.assemble(("batt  ", "dim"),
                      (f"{bt:5.1f}°C" if bt is not None else "   —  ", temp_style(bt)),
                      ("   " + batt_s if batt_s else "", "cyan")),
    ]
    if press is not None:
        items.append(Text.assemble(
            ("pressure ", "dim"),
            (f"{press:3.0f}%", "red" if press >= 80 else ("yellow" if press >= 60 else "cyan"))))
    return Panel(Group(*items), title=" RAM ", border_style="green")


def models_panel(cfg, extra_rows=()):
    rows = poll_models(cfg.ollama, cfg.lmstudio) + list(extra_rows)
    t = Table(box=box.SIMPLE_HEAD, pad_edge=False, expand=True)
    for col, just, ov in (("SRC", "left", "crop"), ("MODEL", "left", "ellipsis"),
                          ("SIZE/QUANT", "left", "crop"), ("VRAM", "right", "crop"),
                          ("CTX", "right", "crop"), ("STATUS", "left", "crop")):
        t.add_column(col, justify=just, overflow=ov)
    for src, name, sz, vram, ctx, status in rows[:8]:
        t.add_row(src, name[:40], str(sz)[:14], fmt_bytes(vram), str(ctx or "—"), status)
    if not rows:
        t.add_row("—", "no servers found (start omlx / ollama / lms server)", "", "", "", "")
    return Panel(t, title=" LLM models ", border_style="yellow")


def rates_panel(stats: LLMStats, proxy_up: bool, omlx_now=None, omlx=None) -> Panel:
    w = stats.window(60)
    gen = [r.gen_tps for r in w if r.gen_tps]
    pr = [r.prompt_tps for r in w if r.prompt_tps]
    tt = [r.ttft_s for r in w if r.ttft_s]

    t = Table(box=box.SIMPLE_HEAD, pad_edge=False, expand=True)
    for col, just in (("TIME", "left"), ("MODEL", "left"), ("PROMPT t/s", "right"),
                      ("GEN t/s", "right"), ("TTFT ms", "right"), ("s", "right"), ("tok p/c", "left")):
        t.add_column(col, justify=just, overflow="fold")
    for r in list(w)[-5:][::-1]:
        t.add_row(
            datetime.fromtimestamp(r.ts).strftime("%H:%M:%S"), r.model[:18],
            f"{r.prompt_tps:8.1f}" if r.prompt_tps else "      —",
            f"{r.gen_tps:8.1f}" if r.gen_tps else "      —",
            f"{r.ttft_s * 1000:8.0f}" if r.ttft_s else "      —",
            f"{r.total_s:8.1f}" if r.total_s else "      —",
            f"{r.prompt_tokens or 0:>4}/{r.completion_tokens or 0:<4}")

    if w:
        summary = Text.assemble(
            ("proxy 60s: ", "bold"),
            (f"gen {sum(gen) / len(gen):5.1f} tok/s", "green"),
            ("  prompt " + (f"{sum(pr) / len(pr):6.1f} tok/s" if pr else "—"), "cyan"),
            ("  TTFT " + (f"{sum(tt) / len(tt) * 1000:5.0f} ms" if tt else "—"), "magenta"),
            (f"  {len(w)} req", "dim"))
    elif proxy_up:
        summary = Text("proxy listening… point a client at it to capture tok/s", style="dim")
    else:
        summary = Text("proxy off — start with --proxy to capture per-request tok/s", style="dim")

    parts = [summary, t]
    if omlx_now:
        g, p, _, live = omlx_now
        parts.append(Text.assemble(
            ("oMLX live:  ", "bold"),
            (f"gen {g:5.1f} tok/s" if g is not None else "gen     —", "green"),
            ("   prompt " + (f"{p:6.1f} tok/s" if p is not None else "—"), "cyan")))
        if omlx and (omlx.avg_gen is not None or omlx.avg_pr is not None):
            parts.append(Text.assemble(
                ("session avg:  ", "dim"),
                ("prefill " + (f"{omlx.avg_pr:6.1f} tok/s" if omlx.avg_pr is not None else "—"), "dim"),
                ("   gen " + (f"{omlx.avg_gen:5.1f} tok/s" if omlx.avg_gen is not None else "—"), "dim")))
        if live:
            lt = Table(box=box.SIMPLE_HEAD, pad_edge=False, expand=True)
            for c, j in (("MODEL", "left"), ("REQ", "left"), ("STATE", "left"), ("ELAPSED", "right")):
                lt.add_column(c, justify=j, overflow="crop")
            for model, rid, state, el in live[:6]:
                color = {"generating": "green", "prefilling": "yellow",
                         "waiting": "dim"}.get(state, "white")
                lt.add_row(model[:24], rid[:12], f"[{color}]{state}[/]", str(el))
            parts.append(lt)
    return Panel(Group(*parts), title=" token rates (prompt processing / generation) ",
                 border_style="bright_yellow")


def proc_table(sort_key: str) -> Table:
    t = Table(box=box.SIMPLE_HEAD, pad_edge=False, expand=True)
    for col, just in (("PID", "right"), ("NAME", "left"), ("CPU%", "right"), ("RSS", "right")):
        t.add_column(col, justify=just)
    procs = []
    for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_info"]):
        try:
            i = p.info
            rss = i["memory_info"].rss if i.get("memory_info") else 0
            procs.append((i["pid"], i.get("name") or "?", i.get("cpu_percent") or 0, rss))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    procs.sort(key=lambda x: x[2] if sort_key == "cpu" else x[3], reverse=True)
    TOP_PROC[0] = procs[0][:2] if procs else None
    for pid, name, c, m in procs[:12]:
        color = "red" if c > 80 else "yellow" if c > 40 else "white"
        t.add_row(str(pid), name[:30], f"[{color}]{c:5.1f}[/]",
                  f"{m / GB:.1f}G" if m >= GB else f"{m / MB:.0f}M")
    return t


def key_reader(flags: dict):
    if not sys.stdin.isatty():
        return
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    # atexit, not try/finally: this daemon thread is blocked in read() when Ctrl-C lands
    atexit.register(termios.tcsetattr, fd, termios.TCSADRAIN, old)
    tty.setcbreak(fd)
    while flags["run"]:
        b = os.read(fd, 1)
        if flags["kill"]:  # awaiting y/n for a pending kill
            pid = flags["kill"][0]
            flags["kill"] = None
            if b == b"y":
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass
        elif b in (b"q", b"\x03", b"\x1b"):
            flags["run"] = False
        elif b == b"r":
            flags["sort"] = "mem" if flags["sort"] == "cpu" else "cpu"
        elif b == b"p":
            flags["paused"] = not flags["paused"]
        elif b == b"k" and not flags["paused"]:
            flags["kill"] = TOP_PROC[0]


# ------------------------------ modes ----------------------------------------
def once_mode(cfg, mm, stats, omlx, proxy_up):
    psutil.cpu_percent(None)
    list(psutil.process_iter(["cpu_percent"]))  # prime per-process CPU counters
    time.sleep(1)
    for _ in range(30):  # wait for the first macmon sample
        if mm.latest:
            break
        time.sleep(0.1)
    hist = deque([sample_cpu()] * 20, maxlen=60)
    omlx_now = omlx.sample() if omlx else None
    console.print(header(cfg, proxy_up))
    console.print(cpu_panel(mm, hist))
    console.print(gpu_panel(mm))
    console.print(ram_panel(mm))
    console.print(models_panel(cfg, omlx_now[2] if omlx_now else []))
    console.print(rates_panel(stats, proxy_up, omlx_now, omlx))
    console.print(Panel(proc_table("cpu"), title=" top processes ", border_style="white"))


def main():
    ap = argparse.ArgumentParser(description="LLMtop — htop for your LLM Mac")
    ap.add_argument("--proxy", action="store_true", help="run capture proxy for tok/s")
    ap.add_argument("--proxy-port", type=int, default=11500)
    ap.add_argument("--upstream", default="http://127.0.0.1:11434",
                    help="proxy target (Ollama default; oMLX: http://127.0.0.1:8000)")
    ap.add_argument("--omlx", default="http://127.0.0.1:8000")
    ap.add_argument("--no-omlx", action="store_true")
    ap.add_argument("--omlx-key", default=os.environ.get("OMLX_API_KEY"))
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--lmstudio", default="http://127.0.0.1:1234")
    ap.add_argument("--no-ollama", action="store_true")
    ap.add_argument("--no-lmstudio", action="store_true")
    ap.add_argument("--once", action="store_true", help="print one snapshot and exit (testing)")
    ap.add_argument("--version", action="version", version=f"llmtop {__version__}")
    cfg = ap.parse_args()
    if cfg.no_ollama:
        cfg.ollama = None
    if cfg.no_lmstudio:
        cfg.lmstudio = None

    stats = LLMStats()
    proxy = None
    if cfg.proxy:
        try:
            proxy = Proxy(cfg.proxy_port, cfg.upstream, stats)
        except OSError as e:
            ap.error(f"cannot start proxy on port {cfg.proxy_port}: {e}")
        proxy.start()
    mm = Macmon()
    omlx = None if cfg.no_omlx else Omlx(cfg.omlx, cfg.omlx_key)

    try:
        if cfg.once:
            once_mode(cfg, mm, stats, omlx, cfg.proxy)
            return

        flags = {"run": True, "sort": "cpu", "paused": False, "kill": None}
        threading.Thread(target=key_reader, args=(flags,), daemon=True).start()

        hist = deque([0.0] * 20, maxlen=60)
        psutil.cpu_percent(None)

        layout = Layout()
        layout.split_column(Layout(name="header", size=3), Layout(name="main"),
                            Layout(name="proc", size=16))
        layout["main"].split_column(Layout(name="sysrow"), Layout(name="llmrow"))
        layout["sysrow"].split_row(Layout(name="cpu"), Layout(name="gpu"), Layout(name="ram"))
        layout["llmrow"].split_row(Layout(name="models"), Layout(name="rates"))

        with Live(console=console, refresh_per_second=2) as live:
            while flags["run"]:
                time.sleep(1)
                if flags["paused"]:
                    continue
                hist.append(sample_cpu())
                omlx_now = omlx.sample() if omlx else None
                layout["header"].update(header(cfg, cfg.proxy))
                layout["cpu"].update(cpu_panel(mm, hist))
                layout["gpu"].update(gpu_panel(mm))
                layout["ram"].update(ram_panel(mm))
                layout["models"].update(models_panel(cfg, omlx_now[2] if omlx_now else []))
                layout["rates"].update(rates_panel(stats, cfg.proxy, omlx_now, omlx))
                table = proc_table(flags["sort"])
                kill = flags["kill"]
                title = (f" kill {kill[1]} (pid {kill[0]})?   y confirm · any other key cancels "
                         if kill else
                         f" top processes (sort: {flags['sort']})   q quit · r sort · p pause · k kill top ")
                layout["proc"].update(Panel(table, title=title,
                                            border_style="red" if kill else "white"))
                live.update(layout)
    except KeyboardInterrupt:
        pass
    finally:
        mm.stop()
        if proxy:
            proxy.stop()


if __name__ == "__main__":
    main()
