"""Tiny deterministic mock of the subset of OpenAI-compatible endpoints the
smoke-check script hits. Single-file, stdlib-only.

Endpoints:
  GET  /v1/models            → 200 with a fixed model list.
  POST /v1/completions       → 200 with a deterministic, byte-identical
                               completion on every call (exercises the smoke
                               check's determinism assertion). If the request
                               body contains "stream": true the response is
                               an SSE stream of the same text (exercises the
                               STREAMING_REQUIRED probe).
  POST /v1/chat/completions  → 200 with a fixed assistant message
                               containing 'Paris' (exercises the chat-
                               template probe). When the request body contains
                               chat_template_kwargs.thinking=true, the
                               response also includes message.reasoning
                               populated from --reasoning-text (exercises the
                               REASONING_REQUIRED probe).

Args:
  --port PORT             (default 18099)
  --text  TEXT            text for /v1/completions choices[0].text and
                          /v1/chat/completions choices[0].message.content
                          (default ' Paris.')
  --chat-text TEXT        override message.content on /v1/chat/completions only;
                          lets a test exercise the chat-template failure path
                          while keeping /v1/completions correct.
  --reasoning-text TEXT   message.reasoning value for thinking-mode chat. Empty
                          string (default) simulates a misconfigured
                          --reasoning-parser that swallowed the <think> block.
  --stream-text TEXT      override the text emitted in SSE chunks on the
                          streaming completion path; falls back to --text when
                          unset. Lets a test exercise the streaming/non-streaming
                          mismatch path without breaking the Paris assertion.
  --sampling-text TEXT    override choices[0].text on /v1/completions when the
                          request body has temperature>0; falls back to --text
                          when unset. Lets a test exercise the SAMPLING_REQUIRED
                          probe's empty/garbage path while keeping the
                          deterministic Paris assertion intact.
  --sampling-finish REASON override choices[0].finish_reason on /v1/completions
                          when the request body has temperature>0 (default
                          'length'). Lets a test exercise the
                          invalid-finish-reason failure path.
  --flaky-readiness N     return 503 from /v1/models for the first N calls, then
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
    reasoning_text = ""  # populated on message.reasoning when thinking=true
    stream_text = None  # falls back to text if unset
    sampling_text = None  # falls back to text on /v1/completions when temp>0
    sampling_finish = "length"  # finish_reason on the temp>0 path
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

    def _stream_completion(self, text: str) -> None:
        """Emit a 3-chunk SSE stream that reassembles to ``text``."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        third = max(1, len(text) // 3)
        chunks = [text[:third], text[third:2 * third], text[2 * third:]]
        for chunk in chunks:
            evt = {
                "id": "cmpl-mock-stream",
                "object": "text_completion",
                "created": 0,
                "model": "deepseek-ai/DeepSeek-V4-Flash",
                "choices": [{
                    "index": 0,
                    "text": chunk,
                    "finish_reason": None,
                }],
            }
            self.wfile.write(f"data: {json.dumps(evt)}\n\n".encode())
        # Final chunk with finish_reason, then [DONE].
        terminal = {
            "id": "cmpl-mock-stream",
            "object": "text_completion",
            "created": 0,
            "model": "deepseek-ai/DeepSeek-V4-Flash",
            "choices": [{"index": 0, "text": "", "finish_reason": "length"}],
        }
        self.wfile.write(f"data: {json.dumps(terminal)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body_raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(body_raw) if body_raw else {}
        except json.JSONDecodeError:
            body = {}

        if self.path == "/v1/completions":
            if body.get("stream") is True:
                stream_text = (MockHandler.stream_text
                               if MockHandler.stream_text is not None
                               else MockHandler.text)
                self._stream_completion(stream_text)
                return
            is_sampling = (body.get("temperature") or 0) > 0
            text = MockHandler.text
            finish = "length"
            if is_sampling and MockHandler.sampling_text is not None:
                text = MockHandler.sampling_text
                finish = MockHandler.sampling_finish
            self._json(200, {
                "id": "cmpl-mock-deterministic",
                "object": "text_completion",
                "created": 0,
                "model": "deepseek-ai/DeepSeek-V4-Flash",
                "choices": [{
                    "index": 0,
                    "text": text,
                    "finish_reason": finish,
                }],
                "usage": {"prompt_tokens": 6, "completion_tokens": 8,
                          "total_tokens": 14},
            })
            return
        if self.path == "/v1/chat/completions":
            chat_content = (MockHandler.chat_text
                            if MockHandler.chat_text is not None
                            else MockHandler.text.strip())
            thinking = bool(
                (body.get("chat_template_kwargs") or {}).get("thinking"))
            message = {"role": "assistant", "content": chat_content}
            if thinking:
                message["reasoning"] = MockHandler.reasoning_text
            self._json(200, {
                "id": "chatcmpl-mock-deterministic",
                "object": "chat.completion",
                "created": 0,
                "model": "deepseek-ai/DeepSeek-V4-Flash",
                "choices": [{
                    "index": 0,
                    "message": message,
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
    ap.add_argument("--reasoning-text", default="")
    ap.add_argument("--stream-text", default=None)
    ap.add_argument("--sampling-text", default=None)
    ap.add_argument("--sampling-finish", default="length")
    ap.add_argument("--flaky-readiness", type=int, default=0)
    args = ap.parse_args()

    MockHandler.text = args.text
    MockHandler.chat_text = args.chat_text
    MockHandler.reasoning_text = args.reasoning_text
    MockHandler.stream_text = args.stream_text
    MockHandler.sampling_text = args.sampling_text
    MockHandler.sampling_finish = args.sampling_finish
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
