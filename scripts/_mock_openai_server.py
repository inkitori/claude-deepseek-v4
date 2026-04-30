"""Tiny deterministic mock of the subset of OpenAI-compatible endpoints the
smoke-check script hits. Single-file, stdlib-only.

Endpoints:
  GET  /v1/models            → 200 with a fixed model list.
  POST /v1/completions       → 200 with a deterministic, byte-identical
                               completion on every call (exercises the smoke
                               check's determinism assertion).
  POST /v1/chat/completions  → 200 with a fixed assistant message
                               containing 'Paris' (exercises the chat-
                               template probe in the smoke check).

Args:
  --port PORT          (default 18099)
  --text  TEXT         text for /v1/completions choices[0].text and
                       /v1/chat/completions choices[0].message.content
                       (default ' Paris.')
  --chat-text TEXT     override message.content on /v1/chat/completions only;
                       lets a test exercise the chat-template failure path
                       while keeping /v1/completions correct.
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
    chat_text = None  # falls back to text.strip() if unset
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
        length = int(self.headers.get("Content-Length", "0"))
        _ = self.rfile.read(length)  # ignore body — deterministic output

        if self.path == "/v1/completions":
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
        if self.path == "/v1/chat/completions":
            chat_content = (MockHandler.chat_text
                            if MockHandler.chat_text is not None
                            else MockHandler.text.strip())
            self._json(200, {
                "id": "chatcmpl-mock-deterministic",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek-ai/DeepSeek-V4-Flash",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": chat_content,
                    },
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 12, "completion_tokens": 1,
                          "total_tokens": 13},
            })
            return
        self.send_response(404); self.end_headers()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18099)
    ap.add_argument("--text", default=" Paris.")
    ap.add_argument("--chat-text", default=None)
    ap.add_argument("--flaky-readiness", type=int, default=0)
    args = ap.parse_args()

    MockHandler.text = args.text
    MockHandler.chat_text = args.chat_text
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
