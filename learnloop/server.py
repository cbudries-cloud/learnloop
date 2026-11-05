"""Local web UI. Binds to 127.0.0.1 only; there is no auth."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from urllib.parse import parse_qs, urlparse

from . import engine


def _deck(qs):
    return (qs.get("deck") or [None])[0] or None


ROUTES = {
    ("GET", "/api/decks"): lambda b, qs: {"decks": engine.decks()},
    ("GET", "/api/queue"): lambda b, qs: engine.queue(_deck(qs)),
    ("GET", "/api/stats"): lambda b, qs: engine.stats(_deck(qs)),
    ("GET", "/api/cards"): lambda b, qs: {"cards": list(engine.load_cards(_deck(qs)).values())},
    ("POST", "/api/grade"): lambda b, qs: engine.grade_answer(int(b["card_id"]), b["answer"]),
    ("POST", "/api/review"): lambda b, qs: engine.review(b),
    ("POST", "/api/tune"): lambda b, qs: engine.tune(),
    ("POST", "/api/tune/apply"): lambda b, qs: engine.apply_params(b),
}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle(self, method):
        url = urlparse(self.path)
        if method == "GET" and url.path in ("/", "/index.html"):
            page = files("learnloop").joinpath("static/index.html").read_bytes()
            return self._send(200, page, "text/html; charset=utf-8")
        fn = ROUTES.get((method, url.path))
        if not fn:
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("content-length") or 0)
            body = json.loads(self.rfile.read(n)) if n else {}
            self._send(200, fn(body, parse_qs(url.query)))
        except Exception as e:  # surface errors to the UI instead of hanging
            self._send(500, {"error": str(e)})

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    def log_message(self, *a):
        pass


def serve(port: int):
    engine.init_db()
    print(f"learnloop on http://127.0.0.1:{port}  (data: {engine.DB_PATH})")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
