#!/usr/bin/env bash
# Idempotent gcsfuse mount of the DeepSeek-V4-Flash GCS staging bucket onto
# the standard HuggingFace cache path so that `vllm serve` / `transformers`
# resolve weights directly from GCS — no local download.
#
# Required env (set in .env or exported):
#   GCS_BUCKET             e.g. personal-mark-eu
#   GCS_ONLY_DIR           e.g. vllm/hub  (subtree of the bucket to expose)
# Optional:
#   HF_HUB_MOUNT_DIR       defaults to $HOME/.cache/huggingface/hub
#   GCS_CACHE_DIR          defaults to /dev/shm/gcsfuse-cache
#   GCS_FILE_CACHE_MB      defaults to 80000
#   GCS_CLIENT_PROTOCOL    defaults to grpc
#
# Exit codes: 0 mounted (or already mounted); non-zero on failure.

set -euo pipefail

if [ "${1:-}" = "--umount" ] || [ "${1:-}" = "-u" ]; then
    target="${HF_HUB_MOUNT_DIR:-$HOME/.cache/huggingface/hub}"
    if mountpoint -q "$target"; then
        fusermount -u "$target"
        echo "[mount_gcs] unmounted $target"
    else
        echo "[mount_gcs] $target not mounted; nothing to do"
    fi
    exit 0
fi

: "${GCS_BUCKET:?GCS_BUCKET not set (e.g. personal-mark-eu)}"
: "${GCS_ONLY_DIR:?GCS_ONLY_DIR not set (e.g. vllm/hub)}"

target="${HF_HUB_MOUNT_DIR:-$HOME/.cache/huggingface/hub}"
cache_dir="${GCS_CACHE_DIR:-/dev/shm/gcsfuse-cache}"
file_cache_mb="${GCS_FILE_CACHE_MB:-80000}"
proto="${GCS_CLIENT_PROTOCOL:-grpc}"

mkdir -p "$target" "$cache_dir"

if mountpoint -q "$target"; then
    echo "[mount_gcs] already mounted at $target"
    exit 0
fi

if ! command -v gcsfuse >/dev/null 2>&1; then
    echo "[mount_gcs] gcsfuse not installed" >&2
    exit 1
fi

echo "[mount_gcs] mounting gs://$GCS_BUCKET/$GCS_ONLY_DIR -> $target"
gcsfuse -o ro --implicit-dirs \
    --only-dir "$GCS_ONLY_DIR" \
    --cache-dir "$cache_dir" \
    --client-protocol "$proto" \
    --file-cache-max-size-mb "$file_cache_mb" \
    --file-cache-cache-file-for-range-read \
    --file-cache-enable-parallel-downloads \
    "$GCS_BUCKET" "$target"

mountpoint -q "$target"
echo "[mount_gcs] mounted; HF cache now resolves from gs://$GCS_BUCKET/$GCS_ONLY_DIR"
