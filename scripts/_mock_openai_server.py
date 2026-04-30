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
  --stop-honor BOOL       when "1" (default), honor the request body's `stop`
                          field by truncating choices[0].text before the first
                          stop string and reporting finish_reason="stop". When
                          "0", emit the full text and finish_reason="length"
                          regardless — exercises the broken-stop-handler path.
  --logprobs-alts N       number of top-logprob alternatives to emit per token
                          when the request body has logprobs>0 (default 5).
                          Set to 0 to emit no logprobs object — exercises the
                          missing-logprobs failure path. Set to <5 to exercise
                          the dropped-alternatives failure path.
  --n-cap N               cap the number of choices emitted on /v1/completions
                          at N (default 0 = no cap; honor request body's `n`
                          field). Set to 1 to exercise the broken-n-expansion
                          failure path (server ignores n>1).
  --long-gen-text TEXT    override choices[0].text on /v1/completions when the
                          request body has max_tokens>=30 (long-gen probe path).
                          Default is a coherent 30+ word passage. Set to a
                          repeated-word string to exercise the
                          token-repetition-loop failure path; set to a string
                          ending in "...." to exercise the dirty-ending path.
  --long-gen-tokens N     `usage.completion_tokens` to report on /v1/completions
                          when max_tokens>=30 (default 35). Set below 30 to
                          exercise the under-produced failure path.
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
    stop_honor = True  # if False, ignore request `stop` and emit full text
    logprobs_alts = 5  # alternatives per token when logprobs>0; 0 = omit object
    n_cap = 0  # if >0, cap number of emitted choices at this value
    long_gen_text = (
        "On the rust-red plains of Mars, the small robot named Argo "
        "rolled forward, its solar panels gleaming under a pale sun. "
        "It paused to scan a curious rock formation and recorded "
        "every reading.")
    long_gen_tokens = 35
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
            is_long_gen = (body.get("max_tokens") or 0) >= 30
            text = MockHandler.text
            finish = "length"
            if is_sampling and MockHandler.sampling_text is not None:
                text = MockHandler.sampling_text
                finish = MockHandler.sampling_finish
            elif is_long_gen:
                text = MockHandler.long_gen_text
                finish = "length"

            # Stop-sequence handling. When `stop` is set in the body, a
            # well-behaved server truncates `text` before the first stop
            # match and reports finish_reason="stop". With `--stop-honor 0`
            # we deliberately ignore the field — exercises the broken path.
            stop_seqs = body.get("stop") or []
            if isinstance(stop_seqs, str):
                stop_seqs = [stop_seqs]
            if stop_seqs and MockHandler.stop_honor:
                cut = len(text)
                for s in stop_seqs:
                    if not isinstance(s, str) or not s:
                        continue
                    idx = text.find(s)
                    if idx >= 0 and idx < cut:
                        cut = idx
                if cut < len(text):
                    text = text[:cut]
                    finish = "stop"

            # Logprobs handling. When `logprobs` is set in the body, emit a
            # per-position object with `tokens`, `token_logprobs`, and
            # `top_logprobs`. With `--logprobs-alts 0` we omit the object
            # entirely (broken path); with a value <5 we emit fewer than
            # the requested alternatives (dropped-alternatives path).
            logprobs_obj = None
            n_lp = body.get("logprobs")
            if isinstance(n_lp, int) and n_lp > 0 and MockHandler.logprobs_alts > 0:
                # Naively split text on whitespace as a stand-in for tokens.
                # Real vLLM emits token-id-aligned entries; the smoke check
                # only inspects len(top_logprobs[i]), which is what this
                # mock cares about.
                toks = text.split() or [text]
                alts = MockHandler.logprobs_alts
                top = []
                for tok in toks:
                    entry = {tok: -0.1}
                    for k in range(1, alts):
                        entry[f"alt_{k}"] = -1.0 - 0.1 * k
                    top.append(entry)
                logprobs_obj = {
                    "tokens": list(toks),
                    "token_logprobs": [-0.1] * len(toks),
                    "top_logprobs": top,
                    "text_offset": [0] * len(toks),
                }

            # Multi-completion handling. Honor request body's `n` field
            # (default 1), capped by --n-cap if set. Each emitted choice
            # carries the same text + finish_reason + logprobs for
            # mock-determinism — the N_REQUIRED probe asserts on the
            # cardinality of the choices array, not their distinctness.
            n_req = body.get("n") or 1
            if not isinstance(n_req, int) or n_req < 1:
                n_req = 1
            if MockHandler.n_cap > 0:
                n_req = min(n_req, MockHandler.n_cap)
            choices = []
            for i in range(n_req):
                c = {"index": i, "text": text, "finish_reason": finish}
                if logprobs_obj is not None:
                    c["logprobs"] = logprobs_obj
                choices.append(c)

            completion_tokens = (MockHandler.long_gen_tokens
                                 if is_long_gen else 8)
            self._json(200, {
                "id": "cmpl-mock-deterministic",
                "object": "text_completion",
                "created": 0,
                "model": "deepseek-ai/DeepSeek-V4-Flash",
                "choices": choices,
                "usage": {"prompt_tokens": 6,
                          "completion_tokens": completion_tokens,
                          "total_tokens": 6 + completion_tokens},
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
    ap.add_argument("--stop-honor", default="1")
    ap.add_argument("--logprobs-alts", type=int, default=5)
    ap.add_argument("--n-cap", type=int, default=0)
    ap.add_argument("--long-gen-text", default=None)
    ap.add_argument("--long-gen-tokens", type=int, default=35)
    ap.add_argument("--flaky-readiness", type=int, default=0)
    args = ap.parse_args()

    MockHandler.text = args.text
    MockHandler.chat_text = args.chat_text
    MockHandler.reasoning_text = args.reasoning_text
    MockHandler.stream_text = args.stream_text
    MockHandler.sampling_text = args.sampling_text
    MockHandler.sampling_finish = args.sampling_finish
    MockHandler.stop_honor = (args.stop_honor != "0")
    MockHandler.logprobs_alts = args.logprobs_alts
    MockHandler.n_cap = args.n_cap
    if args.long_gen_text is not None:
        MockHandler.long_gen_text = args.long_gen_text
    MockHandler.long_gen_tokens = args.long_gen_tokens
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
