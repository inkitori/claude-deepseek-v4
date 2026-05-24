#!/usr/bin/env bash
# Run ON a slice host. Keep mark's `node` ray-worker container permanently OUT.
# Three enforcement layers (idempotent; safe to spam every few seconds):
#
#  (0) PRIMARY — deny `mark` at the SSH layer. mark's tpu-manager VM
#      (35.186.51.62) SSHes in as `mark` every ~10min and runs
#      `/tmp/run_cluster.sh` = `docker stop node; docker rm -f node; docker run
#      -d --restart=unless-stopped --name node vllm/vllm-tpu ... ray start`
#      (confirmed in the sudo audit log + sshd journal). Because it `rm -f`s any
#      existing `node` first, name-occupation alone CANNOT stop it; and iptables
#      by source-IP proved UNRELIABLE here (the L3 source the kernel sees != the
#      controller's logged IP — pkts stayed 0 while logins succeeded). So we deny
#      the `mark` user in sshd: refused PRE-AUTH, independent of source IP / NAT /
#      relay. Reversible: rm the drop-in + `systemctl reload ssh`. Does not touch
#      mark's keys/data; only blocks his SSH to these 8 S1-dedicated hosts.
#
#  (1) BACKUP — firewall-DROP the controller's SSH ingress (best-effort; catches
#      direct packets if their L3 source IS the controller IP).
#
#  (2) BACKUP — occupy the `node` NAME with an inert dummy. `docker create` (NOT
#      start) from a present image, entrypoint /bin/sleep -> touches NOTHING (no
#      ray, no TPU). Distinguished by entrypoint Path == /bin/sleep. Catches a
#      `node` that somehow slips past (0)+(1); the guardian then reclaims it.
#
# Prints: denyusers-mark-applied / fw+<ip> / occupied / reclaimed / (silent).
set -u

# ---- layer (0, PRIMARY): deny `mark` at sshd (path/NAT-independent) ----------
SSHD_DROPIN=/etc/ssh/sshd_config.d/99-s1-block-mark.conf
if ! grep -q '^DenyUsers mark' "$SSHD_DROPIN" 2>/dev/null; then
    printf '# S1: block mark node-redeploy controller (reversible)\nDenyUsers mark\n' \
        | sudo tee "$SSHD_DROPIN" >/dev/null
    sudo sshd -t 2>/dev/null && sudo systemctl reload ssh 2>/dev/null && echo "denyusers-mark-applied"
fi

# ---- layer (1, BACKUP): block the controller's SSH ingress by IP -------------
CTRL_IP="${CTRL_IP:-35.186.51.62}"
if ! sudo iptables -C INPUT -s "$CTRL_IP" -p tcp --dport 22 -j DROP >/dev/null 2>&1; then
    sudo iptables -I INPUT 1 -s "$CTRL_IP" -p tcp --dport 22 -j DROP 2>/dev/null && echo "fw+$CTRL_IP"
fi

# ---- layer (2, BACKUP): occupy the `node` name with an inert dummy -----------

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
