#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="${NAME:-claude-overnight}"
IMAGE="${IMAGE:-claude-overnight}"

if ! [ -f "$REPO_DIR/.env" ]; then
    echo "missing $REPO_DIR/.env" >&2; exit 1
fi
# shellcheck disable=SC1091
set -a; source "$REPO_DIR/.env"; set +a
if [ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
    cat >&2 <<EOF
CLAUDE_CODE_OAUTH_TOKEN is empty in .env.

To populate it:
  1. On THIS host (not in the container), run:  claude setup-token
  2. Copy the printed token.
  3. Edit $REPO_DIR/.env and set:
         CLAUDE_CODE_OAUTH_TOKEN=<paste>
  4. Re-run this script.
EOF
    exit 1
fi

if docker info >/dev/null 2>&1; then
    DOCKER=docker
elif sudo -n docker info >/dev/null 2>&1; then
    DOCKER="sudo docker"
else
    echo "docker is not accessible (no docker group, no passwordless sudo)" >&2
    exit 1
fi

if ! $DOCKER image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "image '$IMAGE' not found — building" >&2
    $DOCKER build \
        --build-arg AGENT_UID="$(id -u)" \
        --build-arg AGENT_GID="$(id -g)" \
        -t "$IMAGE" "$REPO_DIR"
fi

mkdir -p "$REPO_DIR/work" "$REPO_DIR/logs"

# Per-container scratch dir so the agent only sees its own files,
# not other host users' data in /mnt/scratch.
SCRATCH_DIR=/mnt/scratch/claude-overnight
if [ ! -d "$SCRATCH_DIR" ]; then
    sudo mkdir -p "$SCRATCH_DIR"
    sudo chown "$(id -u):$(id -g)" "$SCRATCH_DIR"
fi

$DOCKER rm -f "$NAME" >/dev/null 2>&1 || true

$DOCKER run -d \
    --name "$NAME" \
    --restart unless-stopped \
    --user "$(id -u):$(id -g)" \
    --cap-drop=ALL \
    --cap-add=IPC_LOCK \
    --security-opt no-new-privileges:true \
    --memory=96g \
    --memory-swap=96g \
    --cpus=24 \
    --pids-limit=8192 \
    --ulimit memlock=-1:-1 \
    --ulimit nofile=1048576:1048576 \
    --device /dev/accel0 \
    --device /dev/accel1 \
    --device /dev/accel2 \
    --device /dev/accel3 \
    --shm-size=32g \
    -e JAX_PLATFORMS=tpu,cpu \
    -e TPU_ACCELERATOR_TYPE=v4-8 \
    -e ITER_TIMEOUT_SEC=5400 \
    -v "$REPO_DIR/prompt.md":/workspace/prompt.md:ro \
    -v "$REPO_DIR/entrypoint.sh":/usr/local/bin/entrypoint.sh:ro \
    -v "$REPO_DIR/.env":/workspace/.env:ro \
    -v "$REPO_DIR/work":/workspace/work \
    -v "$REPO_DIR/logs":/workspace/logs \
    -v "$SCRATCH_DIR":/mnt/scratch \
    "$IMAGE"

cat <<EOF
started container '$NAME'.

  live wrapper output:    $DOCKER logs -f $NAME
  live claude iteration:  tail -f $REPO_DIR/logs/iter-*.log
  setup progress:         tail -f $REPO_DIR/logs/setup.log
  poke around inside:     $DOCKER exec -it -u $(id -u):$(id -g) $NAME bash
  stop overnight run:     $DOCKER stop $NAME && $DOCKER rm $NAME

The container will auto-restart if it crashes or the host reboots
(while dockerd is up). To exercise resumability manually:
  $DOCKER kill $NAME      # restart policy brings it back ~immediately

EOF
