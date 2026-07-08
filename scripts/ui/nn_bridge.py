"""Localhost HTTP + SSE bridge for the neural-network live HTML viz."""

import json
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = REPO_ROOT / "web"
HTML_FILE = WEB_DIR / "neural-network-live.html"
DEFAULT_PORT = 8765


class _BridgeState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.latest: Optional[Dict[str, Any]] = None
        self.subscribers: List[queue.Queue] = []
        self.weights_provider: Optional[Callable[[], Optional[Dict]]] = None
        self.tick_count = 0


_state = _BridgeState()
_server: Optional[ThreadingHTTPServer] = None
_server_thread: Optional[threading.Thread] = None
_base_url = ""


class NeuralNetworkBridge:
    """Serve HTML and stream live agent payloads to the browser."""

    @staticmethod
    def start(port: int = DEFAULT_PORT) -> str:
        global _server, _server_thread, _base_url
        if _server is not None:
            return _base_url
        handler = _make_handler()
        _server = ThreadingHTTPServer(("127.0.0.1", port), handler)
        _server.daemon_threads = True
        _server_thread = threading.Thread(target=_server.serve_forever, daemon=True)
        _server_thread.start()
        _base_url = f"http://127.0.0.1:{port}/"
        return _base_url

    @staticmethod
    def publish(payload: Dict[str, Any]) -> None:
        if _server is None:
            return
        data = json.dumps(payload, separators=(",", ":"))
        with _state.lock:
            _state.latest = payload
            _state.tick_count += 1
            dead: List[queue.Queue] = []
            for sub in _state.subscribers:
                try:
                    sub.put_nowait(data)
                except queue.Full:
                    dead.append(sub)
            for sub in dead:
                if sub in _state.subscribers:
                    _state.subscribers.remove(sub)

    @staticmethod
    def set_weights_provider(provider: Callable[[], Optional[Dict]]) -> None:
        _state.weights_provider = provider

    @staticmethod
    def stop() -> None:
        global _server, _server_thread, _base_url
        if _server is not None:
            _server.shutdown()
            _server.server_close()
            _server = None
            _server_thread = None
            _base_url = ""
        with _state.lock:
            _state.subscribers.clear()

    @staticmethod
    def running() -> bool:
        return _server is not None

    @staticmethod
    def has_live_data() -> bool:
        with _state.lock:
            return _state.latest is not None


def _make_handler():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._serve_html()
            elif path == "/snapshot":
                self._serve_snapshot()
            elif path == "/stream":
                self._serve_sse()
            elif path == "/weights":
                self._serve_weights()
            else:
                self.send_error(404)

        def _serve_html(self) -> None:
            if not HTML_FILE.exists():
                self.send_error(404, "neural-network-live.html not found")
                return
            body = HTML_FILE.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_snapshot(self) -> None:
            with _state.lock:
                payload = _state.latest
            body = json.dumps(payload or {"status": "waiting"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_weights(self) -> None:
            provider = _state.weights_provider
            weights = provider() if provider else None
            body = json.dumps(weights or {"error": "no weights"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_sse(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            sub: queue.Queue = queue.Queue(maxsize=32)
            with _state.lock:
                _state.subscribers.append(sub)
                if _state.latest is not None:
                    try:
                        sub.put_nowait(json.dumps(_state.latest, separators=(",", ":")))
                    except queue.Full:
                        pass

            try:
                while True:
                    try:
                        data = sub.get(timeout=15)
                        msg = f"data: {data}\n\n".encode("utf-8")
                        self.wfile.write(msg)
                        self.wfile.flush()
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                with _state.lock:
                    if sub in _state.subscribers:
                        _state.subscribers.remove(sub)

    return Handler
