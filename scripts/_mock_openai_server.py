"""Tiny deterministic mock of the subset of OpenAI-compatible endpoints the
smoke-check script hits. Single-file, stdlib-only.

Endpoints:
  GET  /v1/models       → 200 with a fixed model list.
  POST /v1/completions  → 200 with a deterministic, byte-identical completion
                          on every call (so the smoke check's
                          determinism assertion exercises a real success path).

Args:
  --port PORT          (default 18099)
  --text  TEXT         what to put in choices[0].text (default ' Paris.')
  --flaky-readiness N  return 503 from /v1/models for the first N calls, then
                       200. Useful for exercising the readiness-wait loop.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


class MockHandler(BaseHTTPRequestHandler):
    text = " Paris."
    flaky_remaining = 0

    def log_message(self, format, *args):
        sys.stderr.write(
            f"[mock-openai] {time.strftime('%H:%M:%S')} "
            f"{self.address_string()} {format % args}\n")

    def _json(self, status: int, body: dict) -> None:
        payload = json.dumps(body, sort_keys=True).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            if MockHandler.flaky_remaining > 0:
                MockHandler.flaky_remaining -= 1
                self.send_response(503)
                self.end_headers()
                return
            self._json(200, {
                "object": "list",
                "data": [{
                    "id": "deepseek-ai/DeepSeek-V4-Flash",
                    "object": "model",
                    "owned_by": "mock",
                }],
            })
            return
        self.send_response(404); self.end_headers()

    def do_POST(self) -> None:
        if self.path == "/v1/completions":
            length = int(self.headers.get("Content-Length", "0"))
            _ = self.rfile.read(length)  # ignore body — deterministic output
            self._json(200, {
                "id": "cmpl-mock-deterministic",
                "object": "text_completion",
                "created": 0,
                "model": "deepseek-ai/DeepSeek-V4-Flash",
                "choices": [{
                    "index": 0,
                    "text": MockHandler.text,
                    "finish_reason": "length",
                }],
                "usage": {"prompt_tokens": 6, "completion_tokens": 8,
                          "total_tokens": 14},
            })
            return
        self.send_response(404); self.end_headers()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18099)
    ap.add_argument("--text", default=" Paris.")
    ap.add_argument("--flaky-readiness", type=int, default=0)
    args = ap.parse_args()

    MockHandler.text = args.text
    MockHandler.flaky_remaining = args.flaky_readiness

    with HTTPServer(("127.0.0.1", args.port), MockHandler) as httpd:
        sys.stderr.write(
            f"[mock-openai] listening on 127.0.0.1:{args.port} "
            f"text={args.text!r} flaky={args.flaky_readiness}\n")
        sys.stderr.flush()
        httpd.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
