"""A deployable, dependency-free web interface for the spam detector."""

from __future__ import annotations

import html
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from classifier import SpamClassifier


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "model.json"


def load_model() -> tuple[SpamClassifier, dict]:
    payload = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    return SpamClassifier.from_dict(payload["model"]), payload


MODEL, MODEL_INFO = load_model()


def render_page(message: str = "", label: str | None = None, confidence: float | None = None) -> str:
    safe_message = html.escape(message)
    result = ""
    if label and confidence is not None:
        is_spam = label == "spam"
        title = "Spam likely" if is_spam else "Looks legitimate"
        description = (
            "Treat this message carefully. Do not open links or share personal information until you verify the sender."
            if is_spam
            else "The model found more signals of a normal message. Still use your own judgement before acting."
        )
        result = f"""
        <section class=\"result {'spam' if is_spam else 'ham'}\" aria-live=\"polite\">
          <span class=\"badge\">{html.escape(label.upper())}</span>
          <h2>{title}</h2>
          <p class=\"confidence\">Model confidence: <strong>{confidence:.1%}</strong></p>
          <p>{description}</p>
        </section>"""
    return f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Email Spam Detector</title>
  <style>
    :root {{ color-scheme: light; --ink:#18212f; --muted:#5d6978; --card:#ffffff; --line:#dce3eb; --blue:#2563eb; --blue-dark:#1d4ed8; --red:#b42318; --red-bg:#fff0ed; --green:#067647; --green-bg:#ecfdf3; }}
    * {{ box-sizing: border-box; }} body {{ margin:0; min-height:100vh; background:#f5f8fc; color:var(--ink); font-family:system-ui,-apple-system,BlinkMacSystemFont,\"Segoe UI\",sans-serif; }}
    main {{ width:min(760px,calc(100% - 32px)); margin:0 auto; padding:72px 0 48px; }}
    h1 {{ margin:.5rem 0 .75rem; font-size:clamp(2rem,5vw,3.2rem); line-height:1.08; }}
    .lead {{ color:var(--muted); font-size:1.1rem; line-height:1.6; max-width:650px; }}
    .card {{ background:var(--card); border:1px solid var(--line); border-radius:18px; box-shadow:0 14px 35px rgba(27,47,75,.08); margin-top:32px; padding:28px; }}
    label {{ display:block; font-weight:700; margin-bottom:10px; }} textarea {{ width:100%; min-height:190px; resize:vertical; border:1px solid #b7c3d0; border-radius:12px; padding:14px; color:var(--ink); font:inherit; line-height:1.5; }}
    textarea:focus {{ outline:3px solid #bfdbfe; border-color:var(--blue); }} .help {{ color:var(--muted); font-size:.88rem; margin:.5rem 0 1.25rem; }}
    button {{ appearance:none; border:0; border-radius:10px; background:var(--blue); color:#fff; cursor:pointer; font:inherit; font-weight:750; padding:12px 18px; }} button:hover {{ background:var(--blue-dark); }}
    .result {{ border-radius:14px; margin-top:22px; padding:20px; border:1px solid; }} .result.spam {{ background:var(--red-bg); border-color:#fecdca; }} .result.ham {{ background:var(--green-bg); border-color:#abefc6; }}
    .badge {{ border-radius:999px; display:inline-block; font-size:.75rem; font-weight:800; letter-spacing:.06em; padding:5px 9px; }} .spam .badge {{ background:#fee4e2; color:var(--red); }} .ham .badge {{ background:#d1fadf; color:var(--green); }}
    .result h2 {{ margin:.6rem 0 .25rem; }} .result p {{ line-height:1.5; }} .confidence {{ color:var(--muted); }}
    footer {{ color:var(--muted); font-size:.85rem; line-height:1.5; margin-top:26px; }} code {{ background:#eaf0f7; border-radius:4px; padding:2px 5px; }}
  </style>
</head>
<body>
  <main>
    <h1>Email Spam Detector</h1>
    <p class=\"lead\">Paste an email or message below. The model uses the words it contains to estimate whether it is spam or a normal message.</p>
    <section class=\"card\">
      <form method=\"post\" action=\"/classify\">
        <label for=\"message\">Message to check</label>
        <textarea id=\"message\" name=\"message\" maxlength=\"10000\" required placeholder=\"Example: Congratulations! You have won a free prize. Click now to claim it.\">{safe_message}</textarea>
        <p class=\"help\">For learning only: this is not a security guarantee. Never enter passwords or sensitive personal data.</p>
        <button type=\"submit\">Check message</button>
      </form>
      {result}
    </section>
    <footer>Model: {html.escape(MODEL_INFO['algorithm'])}. Trained on {MODEL_INFO['training_messages']} labelled example messages.</footer>
  </main>
</body>
</html>"""


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - HTTP method names are fixed
        if urlparse(self.path).path == "/health":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
            return
        self._send_html(render_page())

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/classify":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if not 0 < content_length <= 12_000:
            self._send_html(render_page(), HTTPStatus.BAD_REQUEST)
            return
        values = parse_qs(self.rfile.read(content_length).decode("utf-8", errors="replace"))
        message = values.get("message", [""])[0].strip()
        if not message:
            self._send_html(render_page(), HTTPStatus.BAD_REQUEST)
            return
        label, confidence = MODEL.predict(message)
        self._send_html(render_page(message, label, confidence))

    def _send_html(self, page: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = page.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        """Keep normal requests visible but compact in the terminal."""
        print(f"{self.client_address[0]} - {format % args}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    server = ThreadingHTTPServer(("0.0.0.0", port), RequestHandler)
    print(f"Email Spam Detector is running at http://localhost:{port}")
    server.serve_forever()
