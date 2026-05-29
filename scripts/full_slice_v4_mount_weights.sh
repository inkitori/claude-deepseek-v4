#!/usr/bin/env bash
# Ensure the DeepSeek-V4-Flash weights are readable BY enyouki on EVERY host.
#
# Why this exists (the divergent-load wedge, QUANT Q.10):
#   The GCS bucket `personal-mark-eu` (subdir `vllm/hub` = the HF hub cache that
#   holds the V4-Flash snapshot) is mounted ASYMMETRICALLY by the bringup:
#     * head  .15 : an ENYOUKI-owned gcsfuse mount at ~/.cache/huggingface/hub
#                   (uid 2002) -> the serve process (enyouki) can read the weights.
#     * workers   : only a ROOT-owned mount of the `vllm` prefix at /tmp/gcs/bucket
#                   (uid 0, default_permissions) -> `enyouki` gets EACCES, so the
#                   worker actor's model.load_weights() sees is_local_dir=False and
#                   SILENTLY drops into the jnp.zeros dummy fallback
#                   (deepseek_v4.py:_materialize). The head loads real weights, the
#                   workers materialize zeros -> the SPMD launch group diverges and
#                   the engine wedges (EngineCore blocks forever in collective_rpc
#                   ray.get; py-spy shows the 3 remotes stuck in jnp.zeros). This was
#                   masked for many iterations by the earlier create_jit_model
#                   scheckne crashing first; Q.9 (eager create_jit_model) removed
#                   that crash and EXPOSED this.
#
#   Fix: give each WORKER its own enyouki gcsfuse mount at the SAME path the head
#   resolves the repo id to (~/.cache/huggingface/hub), replicating the head's mount
#   command. enyouki has GCS creds on the workers (verified), so the mount succeeds.
#
# Idempotent: skips a host that is already mounted. Run once per bringup (mounts do
# NOT survive a reboot). The v6e-16 smoke calls this as a pre-flight.
#
# Usage: scripts/full_slice_v4_mount_weights.sh
# Reads: WORKERS / HEAD_IP (default: auto-discovered, same as sync.sh).

set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HEAD_IP="${HEAD_IP:-$("$REPO/scripts/full_slice_v4_discover.sh" head)}"
WORKERS="${WORKERS:-$("$REPO/scripts/full_slice_v4_discover.sh" workers)}"
SSH_OPTS="-i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10"

MP="/home/enyouki/.cache/huggingface/hub"
SNAP_CFG="$MP/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/fd53f944496234770ba80e15004f9b6d269a71f5/config.json"
# Replicates the head's enyouki mount (proc cmdline), minus --foreground (daemonize)
# and with a private cache dir so it doesn't collide with the root mount's /dev/shm.
MOUNT_CMD="/usr/bin/gcsfuse -o ro --implicit-dirs --only-dir vllm/hub \
--cache-dir /dev/shm/gcsfuse-cache-enyouki --client-protocol grpc \
personal-mark-eu $MP"

log() { echo "[mount-weights] $*"; }

# Returns 0 if the host can already read the snapshot config.json AS enyouki.
ensure_one() {
    local h="$1"
    local remote="
        mkdir -p '$MP' /dev/shm/gcsfuse-cache-enyouki 2>/dev/null || true
        if mount | grep -q 'personal-mark-eu on $MP type fuse.gcsfuse'; then :;
        else $MOUNT_CMD >/dev/null 2>&1 || true; sleep 1; fi
        if [ -f '$SNAP_CFG' ]; then echo OK; else echo FAIL; fi
    "
    local r
    if [ "$h" = "$HEAD_IP" ]; then
        r="$(bash -c "$remote" 2>/dev/null)"
    else
        r="$(ssh $SSH_OPTS "enyouki@$h" "$remote" 2>/dev/null)"
    fi
    if [ "$r" = "OK" ]; then log "  [$h] OK (config.json readable by enyouki)"; return 0
    else log "  [$h] FAIL (weights NOT readable by enyouki — load would dummy-fallback)"; return 1; fi
}

rc=0
for h in "$HEAD_IP" $WORKERS; do
    ensure_one "$h" || rc=1
done
if [ "$rc" = 0 ]; then log "all hosts can read the V4-Flash weights as enyouki."
else log "ONE OR MORE HOSTS CANNOT READ THE WEIGHTS — a smoke will wedge in the dummy-zeros fallback."; fi
exit "$rc"
