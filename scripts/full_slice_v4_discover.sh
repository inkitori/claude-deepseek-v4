#!/usr/bin/env bash
# Auto-discover the v6e TPU slice topology from the GCP metadata server.
# Works on ANY freshly-provisioned slice — no hardcoded IPs. Run from the
# head (worker 0); the metadata is slice-wide and ordered by worker id.
#
# Usage:
#   eval "$(scripts/full_slice_v4_discover.sh export)"  # sets HEAD_IP/WORKERS/RAY_ADDRESS
#   scripts/full_slice_v4_discover.sh head              # print head (worker 0) IP
#   scripts/full_slice_v4_discover.sh workers           # print worker 1..N IPs
#   scripts/full_slice_v4_discover.sh                   # human-readable summary
#
# The full_slice_v4_*.sh scripts call this as the default for HEAD_IP/WORKERS,
# so on a new slice they "just work". Override by exporting HEAD_IP/WORKERS
# yourself before running them.
set -uo pipefail

MD="http://metadata.google.internal/computeMetadata/v1/instance/attributes"
H="Metadata-Flavor: Google"

# worker-network-endpoints looks like "x:y:10.0.0.1,x:y:10.0.0.2,..." ordered
# by worker id. Pull one IP per entry, robustly (don't trust field position).
mapfile -t IPS < <(curl -sf -H "$H" "$MD/worker-network-endpoints" 2>/dev/null \
                     | tr ',' '\n' | grep -oE '([0-9]{1,3}\.){3}[0-9]{1,3}')

if [ "${#IPS[@]}" -eq 0 ]; then
    echo "[discover] FATAL: no worker-network-endpoints from metadata server." >&2
    echo "[discover] Not on a TPU VM? Export HEAD_IP and WORKERS manually instead." >&2
    exit 1
fi

HEAD_IP="${IPS[0]}"
WORKERS="${IPS[*]:1}"
TPU_ENV="$(curl -sf -H "$H" "$MD/tpu-env" 2>/dev/null || true)"
SLICE="$(printf '%s\n' "$TPU_ENV" | awk -F"'" '/^NODE_ID:/{print $2}')"
ZONE="$(printf '%s\n' "$TPU_ENV" | awk -F"'" '/^ZONE:/{print $2}')"

case "${1:-summary}" in
  head)    echo "$HEAD_IP" ;;
  workers) echo "$WORKERS" ;;
  export)
    echo "export HEAD_IP='$HEAD_IP'"
    echo "export WORKERS='$WORKERS'"
    echo "export RAY_ADDRESS='${HEAD_IP}:6379'"
    ;;
  *)
    echo "slice=${SLICE:-?} zone=${ZONE:-?} hosts=${#IPS[@]}"
    echo "HEAD_IP=$HEAD_IP"
    echo "WORKERS=$WORKERS"
    ;;
esac
