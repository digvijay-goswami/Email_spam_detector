"""Vercel Function entry point for the Email Spam Detector.

Vercel invokes this handler for each request; it must not start its own server.
"""

from __future__ import annotations

import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import MODEL, render_page  # noqa: E402 - project root is added above


class handler(BaseHTTPRequestHandler):
    """Serve the web interface from a Vercel Python Function."""

    def do_GET(self) -> None:  # noqa: N802 - HTTP method names are fixed
        self._send_html(render_page())

    def do_POST(self) -> None:  # noqa: N802
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
        print(f"{self.client_address[0]} - {format % args}")
