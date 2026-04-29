# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""CPU-runnable PyTorch reference for DeepSeek-V4. Sourced from the official
DeepSeek inference/model.py (V4-Pro repo) with custom CUDA kernels replaced by
pure-PyTorch equivalents. Used as numerical ground truth for the JAX TPU
implementation.

See `/workspace/work/tpu-inference/DECISIONS.md` D1 for why this is the
reference (instead of HuggingFace transformers, which does not yet ship the
deepseek_v4 model_type as of transformers 5.6.2).
"""
from .model import (Attention, Block, Compressor, Expert, Gate, Indexer,
                    ModelArgs, MoE, MTPBlock, ParallelEmbedding, ParallelHead,
                    RMSNorm, Transformer, apply_rotary_emb,
                    precompute_freqs_cis)
from .kernel_stubs import (act_quant_noop, fp4_act_quant_noop,
                           hc_split_sinkhorn_torch, sparse_attn_torch)

__all__ = [
    "Attention", "Block", "Compressor", "Expert", "Gate", "Indexer",
    "ModelArgs", "MoE", "MTPBlock", "ParallelEmbedding", "ParallelHead",
    "RMSNorm", "Transformer", "apply_rotary_emb", "precompute_freqs_cis",
    "act_quant_noop", "fp4_act_quant_noop", "hc_split_sinkhorn_torch",
    "sparse_attn_torch",
]
