# SPDX-License-Identifier: Apache-2.0
"""Fused sparse-attention TPU kernel for DeepSeek-V4."""
from tpu_inference.kernels.sparse_attn.kernel import sparse_attn_kernel

__all__ = ["sparse_attn_kernel"]
