#!/usr/bin/env bash
# Run ON a slice host. Permanently occupy the `node` container NAME with an
# inert, never-started dummy so mark's remote controller's
# `docker run --name node vllm/vllm-tpu...` fails with a name conflict. The
# controller does create+start, NEVER `rm` first (verified via `docker events`),
# so a pre-existing `node` blocks its run.
#
# The dummy is `docker create`d (NOT started) from a present image with
# entrypoint /bin/sleep, so it touches NOTHING (no ray, no TPU). It is
# distinguished from the real node by its entrypoint Path == /bin/sleep. If the
# real `node` (ray worker) ever wins the create race, this removes it and
# re-occupies. Idempotent; safe to spam every few seconds.
#
# Prints: occupied | reclaimed | (nothing if our dummy already holds the name).
set -u

# Pick a present image for the inert dummy: prefer the vllm image the real node
# uses (guaranteed present on every host), else a TPU platform image.
IMG="${DUMMY_IMG:-}"
[ -z "$IMG" ] && IMG=$(sudo docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -m1 '^vllm/vllm-tpu')
[ -z "$IMG" ] && IMG=$(sudo docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -m1 '^gcr.io/cloud-tpu-v2-images/tpu_agents')
[ -z "$IMG" ] && { echo "ERR: no image available for dummy" >&2; exit 2; }

occupy() {
    sudo docker create --name node --restart=no --label s1guard=dummy \
        --entrypoint /bin/sleep "$IMG" infinity >/dev/null 2>&1
}

if sudo docker inspect node >/dev/null 2>&1; then
    # A container named `node` exists. Is it OUR inert dummy (entrypoint sleep)?
    if [ "$(sudo docker inspect -f '{{.Path}}' node 2>/dev/null)" != "/bin/sleep" ]; then
        sudo docker rm -f node >/dev/null 2>&1   # real ray node -> evict
        occupy && echo reclaimed
    fi
    # else our dummy already holds the name: nothing to do.
else
    occupy && echo occupied
fi
