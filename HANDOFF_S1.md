# S1 handoff — fresh session, pick up here (2026-05-24, PHASE 8 / FIX v2)

Goal unchanged: make `vllm serve deepseek-ai/DeepSeek-V4-Flash` produce coherent,
deterministic decode on the v6e-32 slice (bug **S1**). Bypass perms; spawn agents
freely; use the TPU; make every call yourself; commit+push checkpoints; never wait.

## Read first
1. **`CLAUDE.md`** PHASE 6, 7, 8 — the full S1 story. Trust it over priors.
2. `git log --oneline -12` — each commit is a narrative step. HEAD should be the
   seeding-only fix / a smoke-verdict commit.

## Where S1 stands (diagnosis is SOLID; fix committed; verification in progress)
- **NOT in the decode math** (fp32-exact == prefill on CPU, PHASE 6). **NOT runtime
  start_pos** (increments correctly). Collapse is **deterministic, at the FIRST decode
  step**. Prefill `h` (token 1) is correct.
- **Root cause (PHASE 7):** the runtime token-shards activations on `ATTN_DATA`. For a
  SHORT prompt the token axis is sharded with idle ranks; the prefill state-**SEEDING**
  ops (`_swa_kv_cache_from_prefill` roll, `_compressor_state_from_prefill` slice) run
  over that token-sharded activation → **corrupted seeded state** → decode reads a bad
  seed → step-1 collapse.
- **FIX v2 (committed, current HEAD of the attention file):** `_replicate(x)` (=
  `with_sharding_constraint(x, P())`, CPU no-op) at the top of
  `attention_init_state_from_prefill` ONLY (prefill seeding; T>1 tokens → normal gather).
  **Do NOT add `_replicate(x_step)` to `attention_decode_step`** — that was c5f245c7 and
  it Core-halts the TPU (a single decode token has a size-1 token axis; the reshard is a
  degenerate all-gather → `SLICE_FAILURE_SW_INJECT_ERROR`). Removed in FIX v2.

## INFRA — the decode Core-halts were `node`-container contention, NOT the fix
- mark's `node` container (ray 2.54.1, `vllm/vllm-tpu:nightly`) is **redeployed by a
  remote controller** AND **auto-starts on every worker reboot** (it was `restart=
  unless-stopped`). It thrashes the TPU launch group (joins our ray → version-mismatch
  Exit 1 → restart loop) → `SLICE_FAILURE_SW_INJECT_ERROR` Core-halts during the 32-way
  decode collective. It ALSO re-poisons GCS `CLUSTER_METADATA` (breaks `ray status`
  version check; meta_guardian re-stamps it).
- **MITIGATION (done):** `node_guardian` hardened to `docker rm -f node` every 3s in
  parallel (was `stop` every 25s). Plus `sudo docker rm -f node` on all 8 hosts. KEEP
  BOTH GUARDIANS ALIVE: `ps -eo pid,cmd | grep -E 'node_guard[i]an|meta_guard[i]an'`
  (want node_guardian + meta_guardian). Restart node_guardian if dead:
  `INTERVAL=3 setsid bash scripts/full_slice_v4_node_guardian.sh >logs/node_guardian.log 2>&1 </dev/null &`
  **`pkill -f` FOOTGUN:** a command line containing the string `full_slice_v4_node_guardian`
  will self-match `pkill -f "[f]ull_slice_v4_node_guardian"` and kill your own shell
  (exit 144). Don't pkill+restart in one command; the old one is usually already dead.
- **Slice recovery (each wedge):** reboot the 7 WORKERS (not head) → wait for SSH →
  remount GCS each (`cd ~/claude-deepseek-v4 && set -a && source .env && set +a &&
  ./scripts/mount_gcs.sh`) → `scripts/full_slice_v4_ray_restart.sh`. ~6 min. With `node`
  removed first, reboot won't auto-start it.

## IN FLIGHT (check this FIRST)
A clean-slice FIX v2 smoke is loading: `logs/full-slice-v4-smoke-20260524T105632Z.log`
(port 18081, tripwire OFF). When it logs `Application startup complete`, run the decode
test: **`/tmp/s1_verify.sh`** (reads ACTUAL decode text on count/paris/mars + 3-Paris
byte-identity + 5-request survival; first probe absorbs the cold decode compile, ~min).
- COHERENT decode (e.g. "Count to 8: 1,2,3,4," → " 5, 6, 7, 8,...") ⇒ FIX v2 WORKS ⇒
  verify TWICE for closure (DONE gate below).
- Still COLLAPSES (no halt) ⇒ seed fix insufficient ⇒ the decode path also needs correct
  sharding via a NON-halting route: a RUNTIME fix (give V4 *decode* a replicated input at
  `tpu_runner.py` `data_parallel_attn_sharding`, so no mid-forward reshard). NOT another
  model-level `with_sharding_constraint` on the size-1 decode activation.
- Core-halts AGAIN ⇒ check `node` slipped in (node_guardian log) / slice degraded; reboot
  recovery + retry.

## DONE (verify TWICE on a fresh engine) — READ THE TEXT
`LONG_GEN_REQUIRED=1 scripts/full_slice_v4_smoke_check.sh` exits 0 (visible_words≥10,
max_word_run<5); 3 Paris probes byte-identical at temp=0; survives 5 unrelated requests.
"Contains Paris" alone is a false positive. After EVERY code edit: `scripts/full_slice_v4_sync.sh`.
CPU repros only confirm NO-REGRESSION (the fix is a CPU no-op): `s1_cpu_repro_v4flash.py both 8 12 4` → exit 0.

## Session continuity (self-perpetuating loop)
S1 outlives a single Claude Code context. A relaunch loop keeps a FRESH session always
working on it: each session starts with empty context (auto-loads CLAUDE.md), works a
bounded chunk, hands off (trims CLAUDE.md, refreshes this file, commits+pushes), and
exits; the wrapper then launches the next fresh session. This keeps context lean while
progress accumulates in git + this file + CLAUDE.md.

- **START:**
  ```
  tmux new-window -n s1loop        # detachable; survives SSH logout
  scripts/s1_session_loop.sh       # runs `claude --print --effort max --dangerously-skip-permissions`
  ```
  (detach: Ctrl-b d · reattach: `tmux attach`). Or headless: `nohup scripts/s1_session_loop.sh >logs/s1_session_loop.log 2>&1 &`.
- **STOP:** `touch /tmp/s1_loop_stop` — the loop checks it before each relaunch and exits
  cleanly without killing an in-flight session (a session also touches it when S1 is DONE).
  Do **not** `pkill -f s1_session_loop` from a shell whose own command line contains that
  string — it self-matches and kills the wrong process. Use the stop file.
- **Per-session prompt:** `scripts/s1_loop_prompt.txt`. **Trim helper:** `scripts/s1_trim_claudemd.sh`
  (archives old PHASE sections to `CLAUDE.full.md`, keeps the 2 newest). Session logs: `logs/s1_session_*.log`.
- **Note:** `claude --print` is agentic but cannot read `/context`; sessions hand off after a
  validated chunk instead. Run one session interactively to literally watch `/context`:
  `claude --effort max --dangerously-skip-permissions "$(cat scripts/s1_loop_prompt.txt)"`.
