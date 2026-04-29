#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch>=2.5",
#     "safetensors>=0.4",
#     "numpy",
#     "packaging",
# ]
# ///
"""
Generate three synthetic tiny-V4 checkpoints under /mnt/scratch/claude-overnight/:

  - tiny_v4_bf16/        all weights bf16 (RMSNorms / hc_* stay fp32),
                         no quantization_config in config.json. Validates
                         the unquantized loader and `vllm serve` end-to-end.

  - tiny_v4_quant/       FP8 dense + FP4 expert weights with ue8m0 (`*.scale`)
                         companions. quantization_config + expert_dtype set.
                         Block sizes scaled to the tiny model (32 / 8). The
                         FP4 codebook and ue8m0 scale recipe are byte-faithful
                         to DeepSeek's /mnt/scratch/v4_pro/inference/convert.py.

  - tiny_v4_groundtruth/ bf16 weights produced by *dequantizing* tiny_v4_quant
                         with the canonical recipe. Tier 7 compares the JAX
                         loader's dequant against this ground truth, isolating
                         loader correctness from quantization-arithmetic noise.

Run on host (uv installed, Python 3.12 fetched on demand):

    /home/enyouki/claude-overnight/scripts/make_tiny_v4_checkpoint.py

The reference DeepSeek `inference/model.py` (in
/home/enyouki/claude-overnight/work/tpu-inference/tests/models/jax/_deepseek_v4_reference/)
defines all the parameter shapes; we instantiate it with a tiny ModelArgs and
dump state_dict() so naming + shapes match real V4 exactly.
"""

import json
import os
import re
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import safe_open, save_file

ROOT = Path(os.environ.get("V4_REPO_ROOT", str(Path(__file__).resolve().parent.parent)))
WORK = ROOT / "work" / "tpu-inference"
REAL_FLASH = Path(os.environ.get("V4_REAL_FLASH", str(ROOT / "work" / "scratch" / "v4_flash_singleshard")))
OUT_DIR = Path(os.environ.get("V4_SCRATCH_DIR", str(ROOT / "work" / "scratch")))

# Real V4 uses fp8_block_size=128 and fp4_block_size=32 (input axis only). We
# scale down 4x so the tiny model's smallest quantizable dim (q_lora_rank=128)
# still spans multiple blocks and the loader's block-loop bookkeeping is
# exercised. The agent's loader reads these from config.json — see
# `quantization_config.weight_block_size` and our added `fp4_block_size` field.
TINY_FP8_BLOCK = 32
TINY_FP4_BLOCK = 8

# DeepSeek's canonical FP4 codebook (e2m1fn) — copied verbatim from
# /mnt/scratch/v4_pro/inference/convert.py so synthetic FP4 quantization is
# bit-identical to upstream. Index 0..7 are positive, 8..15 are negative.
FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
FP4_MAX = 6.0
FP8_E4M3_MAX = 448.0


# ---------- tiny ModelArgs ----------

def build_tiny_args():
    sys.path.insert(0, str(WORK / "tests" / "models" / "jax"))
    from _deepseek_v4_reference.model import ModelArgs

    return ModelArgs(
        max_batch_size=4,
        max_seq_len=256,
        vocab_size=1024,
        dim=256,
        moe_inter_dim=64,
        n_layers=6,
        n_hash_layers=1,
        n_mtp_layers=1,
        n_heads=4,
        n_kv_heads=1,
        n_routed_experts=8,
        n_shared_experts=1,
        n_activated_experts=2,
        score_func="sqrtsoftplus",
        route_scale=1.5,
        swiglu_limit=10.0,
        q_lora_rank=128,
        head_dim=32,
        rope_head_dim=16,
        norm_eps=1e-6,
        o_groups=2,
        o_lora_rank=128,
        window_size=8,
        compress_ratios=(0, 0, 4, 128, 4, 0, 0),  # n_layers + n_mtp_layers
        compress_rope_theta=160000.0,
        original_seq_len=2048,
        rope_theta=10000.0,
        rope_factor=16,
        beta_fast=32,
        beta_slow=1,
        index_n_heads=4,
        index_head_dim=16,
        index_topk=8,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
    )


# ---------- seeded init for the reference Transformer's params ----------

def init_param(name: str, p: torch.Tensor, gen: torch.Generator) -> None:
    if name.endswith("_norm.weight") or name == "norm.weight" or name.endswith(".norm.weight"):
        p.data.fill_(1.0)
        return
    if "tid2eid" in name:
        # hash-routing table; integer dtype
        p.data.copy_(torch.randint(0, p.shape[-1] if p.ndim > 0 else 8,
                                   p.shape, generator=gen, dtype=p.dtype))
        return
    if "attn_sink" in name or name.endswith("hc_attn_base") or name.endswith("hc_ffn_base") \
            or name.endswith("hc_head_base"):
        p.data.zero_()
        return
    if name.endswith("hc_attn_scale") or name.endswith("hc_ffn_scale") or name.endswith("hc_head_scale"):
        p.data.fill_(1.0)
        return
    if "ape" in name:
        # learned positional embedding inside Compressor; zero-init is benign.
        p.data.zero_()
        return
    if p.numel() < 4 or p.ndim == 0:
        p.data.zero_()
        return
    fan_in = p.shape[-1] if p.ndim >= 2 else p.shape[0]
    std = (1.0 / fan_in) ** 0.5
    p.data.copy_((torch.randn(p.shape, generator=gen, dtype=torch.float32) * std).to(p.dtype))


def build_state_dict(seed: int = 0):
    args = build_tiny_args()
    sys.path.insert(0, str(WORK / "tests" / "models" / "jax"))
    from _deepseek_v4_reference.model import Transformer
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = Transformer(args)
    for name, p in model.named_parameters():
        init_param(name, p, gen)
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return args, sd


# ---------- block quantization (recipe per DeepSeek convert.py) ----------

def _scale_to_e8m0(scale_f32: torch.Tensor) -> torch.Tensor:
    """Round a float32 scale UP to the nearest power of two and store as
    float8_e8m0fnu (unsigned 8-bit exponent, bias 127). Matches DeepSeek's
    convert.py: scale_max_offset_bits = scale.amax / 2**MAX_OFFSET_BITS."""
    expo = torch.ceil(torch.log2(scale_f32.clamp(min=1e-30))).clamp(min=-127, max=127)
    return (2.0 ** expo).to(torch.float8_e8m0fnu)


def quantize_fp8_block(w_bf16: torch.Tensor, block: int):
    """Block (block × block) quantize an (out, in) bf16 weight to FP8 e4m3
    + ue8m0 scale. Returns (q_e4m3 of shape (out, in), scale_e8m0 of shape
    (out/block, in/block))."""
    out, inn = w_bf16.shape
    assert out % block == 0 and inn % block == 0, f"FP8 block-quant: shape {w_bf16.shape} not divisible by {block}"
    bO, bI = out // block, inn // block
    blocks = w_bf16.float().view(bO, block, bI, block).permute(0, 2, 1, 3).contiguous()
    absmax = blocks.abs().amax(dim=(-1, -2))
    scale_f32 = (absmax / FP8_E4M3_MAX).clamp(min=1e-30)
    scale_e8m0 = _scale_to_e8m0(scale_f32)
    scale_back = scale_e8m0.float()
    quant = (blocks / scale_back[..., None, None]).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    q_e4m3 = quant.to(torch.float8_e4m3fn).permute(0, 2, 1, 3).reshape(out, inn).contiguous()
    return q_e4m3, scale_e8m0


def dequantize_fp8_block(q_e4m3: torch.Tensor, scale_e8m0: torch.Tensor, block: int) -> torch.Tensor:
    out, inn = q_e4m3.shape
    bO, bI = scale_e8m0.shape
    assert out == bO * block and inn == bI * block
    blocks = q_e4m3.float().view(bO, block, bI, block).permute(0, 2, 1, 3)
    deq = blocks * scale_e8m0.float()[..., None, None]
    return deq.permute(0, 2, 1, 3).reshape(out, inn).to(torch.bfloat16).contiguous()


def quantize_fp4_block(w_bf16: torch.Tensor, fp4_block: int):
    """Row-wise FP4 e2m1 quantization with per-fp4_block ue8m0 scales.
    Storage layout matches DeepSeek convert.py:
      weight: int8 shape (out, in/2) — two FP4 nibbles packed per byte (low
              nibble first), each nibble indexes FP4_TABLE.
      scale:  float8_e8m0fnu shape (out, in / fp4_block)."""
    out, inn = w_bf16.shape
    assert inn % fp4_block == 0 and inn % 2 == 0
    nB = inn // fp4_block
    w = w_bf16.float().view(out, nB, fp4_block)
    absmax = w.abs().amax(dim=-1)
    scale_f32 = (absmax / FP4_MAX).clamp(min=1e-30)
    scale_e8m0 = _scale_to_e8m0(scale_f32)
    scale_back = scale_e8m0.float()
    w_norm = (w / scale_back[..., None]).clamp(-FP4_MAX, FP4_MAX)
    diffs = (w_norm.unsqueeze(-1) - FP4_TABLE).abs()
    idx = diffs.argmin(dim=-1).to(torch.uint8).view(out, inn)
    low = idx[:, 0::2]
    high = idx[:, 1::2]
    packed = (low | (high << 4)).to(torch.int8).contiguous()
    return packed, scale_e8m0


def dequantize_fp4_block(packed: torch.Tensor, scale_e8m0: torch.Tensor, fp4_block: int) -> torch.Tensor:
    out, half = packed.shape
    inn = half * 2
    nB = scale_e8m0.shape[1]
    assert nB == inn // fp4_block
    p = packed.view(torch.uint8).long()
    low_idx = p & 0x0F
    high_idx = (p >> 4) & 0x0F
    vals = torch.stack([FP4_TABLE[low_idx], FP4_TABLE[high_idx]], dim=-1).flatten(1)
    vals = vals.view(out, nB, fp4_block) * scale_e8m0.float()[..., None]
    return vals.view(out, inn).to(torch.bfloat16).contiguous()


# ---------- which tiny tensors get quantized? derive from real V4 index ----------

def load_real_quant_targets():
    idx = json.loads((REAL_FLASH / "model.safetensors.index.json").read_text())["weight_map"]
    fp4_pat: set[str] = set()
    fp8_pat: set[str] = set()
    for k in idx:
        if not k.endswith(".scale"):
            continue
        wname = k[: -len(".scale")] + ".weight"
        pat = re.sub(r"\.\d+\.", ".N.", wname)
        if "experts" in pat and "shared_experts" not in pat:
            fp4_pat.add(pat)
        else:
            fp8_pat.add(pat)
    return fp4_pat, fp8_pat


def name_pattern(name: str) -> str:
    return re.sub(r"\.\d+\.", ".N.", name)


# ---------- io helpers ----------

def to_storage(v: torch.Tensor) -> torch.Tensor:
    if v.dtype == torch.float32:
        return v.contiguous()
    if v.dtype.is_floating_point:
        return v.contiguous().to(torch.bfloat16)
    return v.contiguous()


def write_index(sd: dict, out_dir: Path) -> None:
    weight_map = {k: "model-00001-of-00001.safetensors" for k in sd}
    total = sum(v.numel() * v.element_size() for v in sd.values())
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2, sort_keys=True)
    )


def write_config_and_aux(args, out_dir: Path, with_quant: bool) -> None:
    real_cfg = json.loads((REAL_FLASH / "config.json").read_text())
    cfg = dict(real_cfg)
    cfg["hidden_size"] = args.dim
    cfg["moe_intermediate_size"] = args.moe_inter_dim
    cfg["num_hidden_layers"] = args.n_layers
    cfg["num_attention_heads"] = args.n_heads
    cfg["num_key_value_heads"] = args.n_kv_heads
    cfg["head_dim"] = args.head_dim
    cfg["qk_rope_head_dim"] = args.rope_head_dim
    cfg["q_lora_rank"] = args.q_lora_rank
    cfg["o_lora_rank"] = args.o_lora_rank
    cfg["o_groups"] = args.o_groups
    cfg["index_head_dim"] = args.index_head_dim
    cfg["index_n_heads"] = args.index_n_heads
    cfg["index_topk"] = args.index_topk
    cfg["hc_mult"] = args.hc_mult
    cfg["hc_sinkhorn_iters"] = args.hc_sinkhorn_iters
    cfg["n_routed_experts"] = args.n_routed_experts
    cfg["n_shared_experts"] = args.n_shared_experts
    cfg["num_experts_per_tok"] = args.n_activated_experts
    cfg["num_hash_layers"] = args.n_hash_layers
    cfg["num_nextn_predict_layers"] = args.n_mtp_layers
    cfg["sliding_window"] = args.window_size
    cfg["max_position_embeddings"] = args.max_seq_len * 16
    cfg["vocab_size"] = args.vocab_size
    cfg["compress_ratios"] = list(args.compress_ratios)
    cfg["compress_rope_theta"] = args.compress_rope_theta
    if "rope_scaling" in cfg:
        cfg["rope_scaling"]["original_max_position_embeddings"] = args.original_seq_len
    if with_quant:
        cfg["quantization_config"] = {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "scale_fmt": "ue8m0",
            "weight_block_size": [TINY_FP8_BLOCK, TINY_FP8_BLOCK],
        }
        cfg["expert_dtype"] = "fp4"
        cfg["fp4_block_size"] = TINY_FP4_BLOCK
    else:
        cfg.pop("quantization_config", None)
        cfg.pop("expert_dtype", None)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True))
    for fn in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        src = REAL_FLASH / fn
        if src.exists():
            shutil.copy(src, out_dir / fn)


# ---------- the three fixtures ----------

def write_bf16(args, sd, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_config_and_aux(args, out_dir, with_quant=False)
    storage = {k: to_storage(v) for k, v in sd.items()}
    save_file(storage, str(out_dir / "model-00001-of-00001.safetensors"))
    write_index(storage, out_dir)


def write_quant(args, sd, out_dir: Path, fp4_pat: set, fp8_pat: set) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_config_and_aux(args, out_dir, with_quant=True)
    storage: dict[str, torch.Tensor] = {}
    meta: dict[str, dict] = {}
    for k, v in sd.items():
        pat = name_pattern(k)
        is_2d = v.ndim == 2
        if is_2d and pat in fp4_pat and v.shape[-1] % TINY_FP4_BLOCK == 0:
            q, s = quantize_fp4_block(v.to(torch.bfloat16), TINY_FP4_BLOCK)
            storage[k] = q
            storage[k.replace(".weight", ".scale")] = s
            meta[k] = {"kind": "fp4", "block": TINY_FP4_BLOCK}
        elif is_2d and pat in fp8_pat \
                and v.shape[0] % TINY_FP8_BLOCK == 0 and v.shape[-1] % TINY_FP8_BLOCK == 0:
            q, s = quantize_fp8_block(v.to(torch.bfloat16), TINY_FP8_BLOCK)
            storage[k] = q
            storage[k.replace(".weight", ".scale")] = s
            meta[k] = {"kind": "fp8", "block": TINY_FP8_BLOCK}
        else:
            storage[k] = to_storage(v)
            meta[k] = {"kind": "bf16" if v.dtype.is_floating_point else "raw",
                       "dtype": str(storage[k].dtype)}
    save_file(storage, str(out_dir / "model-00001-of-00001.safetensors"))
    write_index(storage, out_dir)
    (out_dir / "quant_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))


def write_groundtruth(args, src_quant: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_config_and_aux(args, out_dir, with_quant=False)
    meta = json.loads((src_quant / "quant_meta.json").read_text())
    deq: dict[str, torch.Tensor] = {}
    with safe_open(str(src_quant / "model-00001-of-00001.safetensors"), framework="pt") as f:
        keys = list(f.keys())
        for k in keys:
            if k.endswith(".scale"):
                continue
            t = f.get_tensor(k)
            m = meta.get(k, {"kind": "bf16"})
            if m["kind"] == "fp8":
                s = f.get_tensor(k.replace(".weight", ".scale"))
                deq[k] = dequantize_fp8_block(t, s, m["block"])
            elif m["kind"] == "fp4":
                s = f.get_tensor(k.replace(".weight", ".scale"))
                deq[k] = dequantize_fp4_block(t, s, m["block"])
            else:
                deq[k] = t.contiguous()
    save_file(deq, str(out_dir / "model-00001-of-00001.safetensors"))
    write_index(deq, out_dir)


def main() -> None:
    if not REAL_FLASH.exists():
        sys.exit(f"missing {REAL_FLASH} — stage the V4 assets first")
    if not (WORK / "tests" / "models" / "jax" / "_deepseek_v4_reference" / "model.py").exists():
        sys.exit(f"missing {WORK} — work/tpu-inference is not bootstrapped (start the container once)")

    print("[1/3] building bf16 fixture")
    args, sd = build_state_dict(seed=0)
    print(f"      state_dict has {len(sd)} tensors")
    write_bf16(args, sd, OUT_DIR / "tiny_v4_bf16")

    print("[2/3] building FP8/FP4 quant fixture")
    fp4_pat, fp8_pat = load_real_quant_targets()
    print(f"      quant patterns from real V4-Flash: {len(fp4_pat)} FP4, {len(fp8_pat)} FP8")
    write_quant(args, sd, OUT_DIR / "tiny_v4_quant", fp4_pat, fp8_pat)

    print("[3/3] building groundtruth (dequant of quant)")
    write_groundtruth(args, OUT_DIR / "tiny_v4_quant", OUT_DIR / "tiny_v4_groundtruth")

    print("done. fixtures at:")
    for d in ("tiny_v4_bf16", "tiny_v4_quant", "tiny_v4_groundtruth"):
        p = OUT_DIR / d
        st = p / "model-00001-of-00001.safetensors"
        print(f"  {p}  ({st.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
