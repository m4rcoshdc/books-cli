"""`books serve`: a small local HTTP server that keeps bge-m3 loaded.

Why: starting Python, importing PyTorch and loading the model takes ~50 s, while
embedding one question takes ~0.06 s. A process that stays up pays the 50 s once.
The CLI checks for this server and sends meaning searches to it; every other
tool is fast without it.

Safety: it listens on 127.0.0.1 only (never the network), is read-only, and
refuses requests that come from a web browser (they carry an Origin header),
so a web page can't query your library through it.

  GET  /health          is it up, which model, how many requests
  POST /tool/<name>     JSON arguments -> the library function's dict
  POST /shutdown        stop (used by `books serve --stop`)
"""

import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

from . import embed as emb
from . import library as lib

_started = time.time()
_requests = 0


class _Handler(BaseHTTPRequestHandler):
    server_version = "books/1"

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _from_browser(self) -> bool:
        return bool(self.headers.get("Origin") or self.headers.get("Sec-Fetch-Site"))

    def do_GET(self):
        if self.path != "/health":
            return self._send(404, {"error": "unknown path"})
        self._send(200, {"ok": True, "model": f"{emb.MODEL} {emb.precision()} on {emb.device()}",
                         "up_seconds": round(time.time() - _started), "requests": _requests})

    def do_POST(self):
        global _requests
        if self._from_browser():
            return self._send(403, {"error": "browser requests are refused"})
        if self.path == "/shutdown":
            self._send(200, {"ok": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return
        name = self.path.removeprefix("/tool/")
        fn = lib.TOOLS.get(name)
        if not self.path.startswith("/tool/") or fn is None:
            return self._send(404, {"error": f"unknown tool '{name}'"})
        try:
            args = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            t = time.time()
            result = fn(**args)
            _requests += 1
            self.server.log(f"{name} {json.dumps(args, ensure_ascii=False)[:90]}  {time.time() - t:.2f}s")
            self._send(200, result)
        except (lib.NotFound, ValueError, TypeError) as e:
            self._send(200, {"error": str(e)})

    def log_message(self, *args):  # silence the default per-request access log
        pass


def run(log=print) -> None:
    if lib.server_up():
        log(f"Already running on {lib.SERVER_HOST}:{lib.SERVER_PORT}.")
        return
    t = time.time()
    log("Loading bge-m3 (about a minute)...")
    emb.encode(["warm-up"])  # first call also initialises the CUDA kernels
    lib._catalog()
    lib._chunk_counts()
    httpd = HTTPServer((lib.SERVER_HOST, lib.SERVER_PORT), _Handler)
    httpd.log = log
    log(f"Ready in {time.time() - t:.0f} s: {emb.MODEL} {emb.precision()} on {emb.device()}, "
        f"http://{lib.SERVER_HOST}:{lib.SERVER_PORT} (local only). Ctrl+C or `books serve --stop` to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    log("Stopped.")


def _call(method: str, path: str) -> dict | None:
    if not lib.server_up():
        return None
    req = urllib.request.Request(f"http://{lib.SERVER_HOST}:{lib.SERVER_PORT}{path}", method=method,
                                 data=b"{}" if method == "POST" else None)
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read().decode("utf-8"))


def status() -> str:
    h = _call("GET", "/health")
    if not h:
        return "Not running. Start it with `books serve` (meaning search falls back to keyword meanwhile)."
    return f"Running: {h['model']}, up {h['up_seconds'] // 60} min, {h['requests']} requests."


def stop() -> str:
    return "Stopped." if _call("POST", "/shutdown") else "Not running."
