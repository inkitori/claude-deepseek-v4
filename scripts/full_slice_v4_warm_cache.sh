#!/usr/bin/env bash
# Warm the JAX compile cache + libtpu state on a freshly-provisioned VM.
#
# What this does:
#   1. Runs the standard smoke launcher (full_slice_v4_smoke.sh) so vLLM
#      boots and capture_model finishes.
#   2. Polls /v1/models, then fires ONE deterministic completion request
#      so jit_run_model is JIT-compiled for the runtime shape.
#   3. Tears down cleanly via full_slice_v4_reset.sh.
#
# After it exits, JAX_COMPILATION_CACHE_DIR (default /tmp/jax-compile-cache-v4)
# is populated on every host, so the NEXT real `vllm serve` launch's first
# /v1/completions returns in ~1-2s instead of ~5-10 min cold compile.
#
# Use case: run ONCE per VM-lifecycle (after fresh boot, after /tmp wipe,
# after JAX_COMPILATION_CACHE_DIR change). The smoke launcher and real
# serve both honor the same cache dir, so any cache populated by this
# script is reused by serve.
#
# Usage:
#   scripts/full_slice_v4_warm_cache.sh
#
# Reads:
#   PORT  — same as smoke launcher (default 18081)

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-18081}"
CACHE_DIR="${V4_JAX_CACHE_DIR:-/tmp/jax-compile-cache-v4}"

log() { echo "[warm-cache] $*"; }

log "starting smoke launcher (port=$PORT, cache=$CACHE_DIR)"
SMOKE_LOG="$("$REPO/scripts/full_slice_v4_smoke.sh" 2>&1 | tail -1)"
if [ -z "$SMOKE_LOG" ] || [ ! -f "$SMOKE_LOG" ]; then
    log "FAIL: smoke launcher did not return a log path"
    exit 1
fi
log "smoke log: $SMOKE_LOG"

log "running smoke_check (waits for ready, fires curl, validates)"
"$REPO/scripts/full_slice_v4_smoke_check.sh"
rc=$?

log "tearing down via reset"
"$REPO/scripts/full_slice_v4_reset.sh" >/dev/null 2>&1 || true

if [ $rc -ne 0 ]; then
    log "FAIL: smoke_check exit=$rc — see $SMOKE_LOG"
    exit $rc
fi

log "cache state on head ($CACHE_DIR):"
ls -la "$CACHE_DIR" 2>/dev/null | head -5 | sed 's/^/  /'
log "size: $(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)"
log "OK: cache warmed. Subsequent real launches' first /v1/completions"
log "    will hit cache (~1-2s) instead of cold-compiling jit_run_model."
