#!/usr/bin/env bash
# Fingerprint the per-host JAX compile cache across all 8 v6e-32 hosts.
#
# Diagnostic for the lane-2 question (CLAUDE.md "Next attack lanes"):
# is host 0's JAX cache rsync-shareable to the other 7 workers?
#
# Verified 2026-04-30: NO. SPMD compiles produce host-specific binaries.
# After a successful smoke, every host has the same cache *filenames*
# (the JAX cache key is identical: same program, same mesh, same args)
# but the file *contents* differ — observed 8 distinct sha256s for the
# 7 cache entries that appeared on every host. Rsync'ing host 0's
# cache to workers would serve them code compiled for host 0's chip
# topology coordinates, which is unsound.
#
# This script is kept as a diagnostic. Run it whenever the cache flow
# is suspicious — empty cache after a smoke, mysterious recompiles,
# etc. Output groups one `<sha256>  <relpath>` line per entry under
# each host header; diff across hosts with awk/sort.
#
# Cache path: vllm's tpu_inference/runner/compilation_manager.py:53
# calls jax.config.update("jax_compilation_cache_dir",
# vllm_envs.VLLM_XLA_CACHE_PATH) at engine init, overriding whatever
# JAX_COMPILATION_CACHE_DIR the launcher set. Real cache always lives
# at $VLLM_XLA_CACHE_PATH (default ~/.cache/vllm/xla_cache).
#
# Quick analysis to confirm hashes diverge per-host:
#
#     scripts/full_slice_v4_cache_fingerprint.sh > /tmp/cache-fp.txt
#     awk '/^=== /{h=$2; next} /^  [a-f0-9]/{print h" "$1" "$2}' \
#         /tmp/cache-fp.txt > /tmp/fp-flat.txt
#     # for each filename present on >1 host, count distinct hashes
#     awk '{print $3}' /tmp/fp-flat.txt | sort -u \
#         | while read f; do
#             nh=$(grep " $f$" /tmp/fp-flat.txt | awk '{print $2}' | sort -u | wc -l)
#             nhosts=$(grep -c " $f$" /tmp/fp-flat.txt)
#             [ "$nhosts" -gt 1 ] && echo "$f hosts=$nhosts hashes=$nh"
#         done
#
# Usage: scripts/full_slice_v4_cache_fingerprint.sh
#
# Env:
#   WORKERS  — space-separated worker IPs (default = the 7-host slice)
#   CACHE_DIR — defaults to ~/.cache/vllm/xla_cache (= VLLM_XLA_CACHE_PATH)

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/vllm/xla_cache}"
HEAD_IP="${HEAD_IP:-$("$REPO/scripts/full_slice_v4_discover.sh" head)}"
WORKERS="${WORKERS:-$("$REPO/scripts/full_slice_v4_discover.sh" workers)}"
SSH_OPTS="-i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10"

fp_local() {
    if [ ! -d "$CACHE_DIR" ]; then
        echo "  (cache dir does not exist)"
        return
    fi
    cd "$CACHE_DIR" && find . -type f -print0 \
        | sort -z \
        | xargs -0 -r sha256sum 2>/dev/null \
        | sed 's/^/  /' || echo "  (no entries)"
}

fp_remote() {
    local host="$1"
    ssh $SSH_OPTS enyouki@"$host" "
        if [ ! -d '$CACHE_DIR' ]; then
            echo '  (cache dir does not exist)'
        else
            cd '$CACHE_DIR' && find . -type f -print0 \
                | sort -z \
                | xargs -0 -r sha256sum 2>/dev/null \
                | sed 's/^/  /' || echo '  (no entries)'
        fi
    " 2>&1 | grep -v "^Warning:"
}

echo "=== $HEAD_IP (head) ==="
fp_local
for h in $WORKERS; do
    echo "=== $h ==="
    fp_remote "$h"
done
