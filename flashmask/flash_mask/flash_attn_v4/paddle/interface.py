# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# [2025-07-04] Version in Cute-DSL, for Hopper and Blackwell. You'll need install nvidia-cutlass-dsl==4.2.0.

# Supported features:
# - BF16 & FP16 dtype
# - noncausal & causal attention
# - MHA, GQA, MQA
# - hdim 64, 96, 128.
# - (hdim_qk, hdim_v) = (192, 128) for Blackwell (i.e. DeepSeek shape)
# - varlen
# - sliding window
# - bwd pass for Ampere (will also run on Hopper/Blackwell, but will be slow)

# Features not supported yet:
# - split (i.e. FlashDecoding)
# - tuned block sizes
# - paged KV
# - append KV to existing KV cache
# - FP8
# - bwd pass optimized for Hopper/Blackwell

import os
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple, Callable

import paddle


import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32
from flash_mask.flash_attn_v4.paddle.cute_dsl_utils import make_fake_tensor as fake_tensor
from flash_mask.flash_attn_v4.cache_utils import get_jit_cache
from flash_mask.flash_attn_v4.testing import is_fake_mode


if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
    from flash_mask.flash_attn_v4 import cute_dsl_ptxas  # noqa: F401

    # Patch to dump ptx and then use system ptxas to compile to cubin
    cute_dsl_ptxas.patch()


from flash_mask.flash_attn_v4 import utils
from flash_mask.flash_attn_v4 import fa_logging
from flash_mask.flash_attn_v4.paddle.cute_dsl_utils import (
    to_cute_tensor, to_cute_aux_tensor, get_aux_tensor_metadata, get_broadcast_dims,
    paddle2cute_dtype_map,
)
from flash_mask.flash_attn_v4.flash_fwd import FlashAttentionForwardSm80
from flash_mask.flash_attn_v4.flash_fwd_sm90 import FlashAttentionForwardSm90
from flash_mask.flash_attn_v4.flash_fwd_sm100 import FlashAttentionForwardSm100
from flash_mask.flash_attn_v4.flash_fwd_sm120 import FlashAttentionForwardSm120
from flash_mask.flash_attn_v4.flash_bwd_preprocess import FlashAttentionBackwardPreprocess
from flash_mask.flash_attn_v4.flash_bwd import FlashAttentionBackwardSm80
from flash_mask.flash_attn_v4.flash_bwd_sm90 import FlashAttentionBackwardSm90
from flash_mask.flash_attn_v4.flash_bwd_sm100 import FlashAttentionBackwardSm100
from flash_mask.flash_attn_v4.flash_bwd_sm120 import FlashAttentionBackwardSm120
from flash_mask.flash_attn_v4.flash_bwd_postprocess import FlashAttentionBackwardPostprocess
from flash_mask.flash_attn_v4.flash_fwd_combine import FlashAttentionForwardCombine
from flash_mask.flash_attn_v4.sm100_hd256_2cta_fmha_forward import BlackwellFusedMultiHeadAttentionForward
from flash_mask.flash_attn_v4.sm100_hd256_2cta_fmha_backward import BlackwellFusedMultiHeadAttentionBackward

from flash_mask.flash_attn_v4.paddle.block_sparsity import (
    BlockSparseTensorsPaddle,
    get_sparse_q_block_size,
    to_cute_block_sparse_tensors,
    normalize_block_sparse_config,
    normalize_block_sparse_config_bwd,
)


def _swap_last_two_dims(t: paddle.Tensor) -> paddle.Tensor:
    """Swap the last two dimensions (equivalent to torch's .transpose(-1, -2))."""
    ndim = len(t.shape)
    perm = list(range(ndim))
    perm[-1], perm[-2] = perm[-2], perm[-1]
    return paddle.transpose(t, perm=perm)


def _dim_order(t: paddle.Tensor) -> tuple:
    """Return dimension indices sorted by stride ascending, matching torch.Tensor.dim_order()."""
    strides = list(t.strides)
    return tuple(sorted(range(len(strides)), key=lambda i: strides[i]))


def _parse_arch_str(arch_str):
    """Parse arch string (e.g. 'sm_80', 'sm_90a', '80', '100') to int (e.g. 80, 90, 100)."""
    import re
    match = re.match(r"^(?:sm_?|SM_?)?(\d+)(\d)([af]?)$", arch_str)
    if not match:
        raise ValueError(f"Invalid arch format: {arch_str}")
    major, minor, _ = match.groups()
    return int(major) * 10 + int(minor)


@lru_cache(maxsize=None)
def _get_device_arch():
    """Cached device arch check.

    Override with FLASH_ATTENTION_ARCH (e.g. 'sm_80' or '80') to select which
    kernel path to use (SM80/SM90/SM100/SM120) independently of the compilation
    target (CUTE_DSL_ARCH).

    For CPU-only compilation (no GPU), set both:
      FLASH_ATTENTION_ARCH=sm_80  (kernel selection)
      CUTE_DSL_ARCH=sm_80         (compilation target)
    """
    arch_override = os.environ.get("FLASH_ATTENTION_ARCH", None)
    if arch_override is not None:
        return _parse_arch_str(arch_override)
    major, minor = paddle.device.cuda.get_device_capability()
    return major * 10 + int(minor)


def _validate_head_dims(head_dim: int, head_dim_v: int, compute_capability: int, alignment: int) -> None:
    """Validate head dimension constraints based on compute capability."""
    is_deepseek_shape = head_dim == 192 and head_dim_v == 128
    is_dedicate_kernel_shape = head_dim == 256 and head_dim_v == 256
    is_standard_range = 8 <= head_dim <= 128 and 8 <= head_dim_v <= 128

    is_sm90_range = 8 <= head_dim <= 256 and 8 <= head_dim_v <= 256
    if compute_capability == 9:
        assert is_sm90_range and head_dim % alignment == 0 and head_dim_v % alignment == 0, (
            f"(head_dim, head_dim_v)=({head_dim}, {head_dim_v}) is not supported on SM90. "
            f"head_dim and head_dim_v must be between 8 and 256 and divisible by {alignment}."
        )
    elif compute_capability in [10, 11]:
        assert (is_standard_range or is_deepseek_shape or is_dedicate_kernel_shape) and head_dim % alignment == 0 and head_dim_v % alignment == 0, (
            f"(head_dim, head_dim_v)=({head_dim}, {head_dim_v}) is not supported on SM100/SM110. "
            f"head_dim and head_dim_v must be between 8 and 128 and divisible by {alignment}, or (192, 128) for DeepSeek, or (256, 256) for hd256."
        )


@dataclass(frozen=True)
class FwdConfig:
    m_block_size: int
    n_block_size: int
    mma_pv_is_rs: bool
    intra_wg_overlap: bool


def _tile_size_fwd_sm90(head_dim, head_dim_v, is_causal, is_local, sparse_block_size_q=None):
    """Return FwdConfig for SM90 forward."""
    if head_dim <= 64:
        if sparse_block_size_q is not None and sparse_block_size_q % 192 != 0:
            return FwdConfig(128, 128, True, True)
        return FwdConfig(192, 128, True, True)
    elif head_dim <= 96:
        if sparse_block_size_q is not None and sparse_block_size_q % 192 != 0:
            return FwdConfig(128, 128, False, True)
        if is_causal or is_local:
            return FwdConfig(192, 128, False, True)
        else:
            return FwdConfig(192, 144, False, True)
    elif head_dim <= 128:
        return FwdConfig(128, 128, True, True)
    elif head_dim <= 192:
        tile_n = 96 if is_local else (128 if head_dim_v <= 128 else 112)
        return FwdConfig(128, tile_n, True, True)
    else:  # hdim 256
        tile_n = 64 if is_local else 80
        return FwdConfig(128, tile_n, True, True)


@dataclass(frozen=True)
class BwdConfig:
    m_block_size: int
    n_block_size: int
    num_stages_Q: int
    num_stages_dO: int
    num_stages_PdS: int
    SdP_swapAB: bool
    dKV_swapAB: bool
    dQ_swapAB: bool
    AtomLayoutMSdP: int
    AtomLayoutNdKV: int
    AtomLayoutMdQ: int
    num_wg: int = 2  # MMA warp groups (total threads = (num_wg + 1) * 128)
    dQ_single_wg: bool = False


def _tile_size_bwd_sm90(head_dim, head_dim_v, causal, local, sparse_block_size_q=None):
    """Return BwdConfig for SM90."""
    if head_dim <= 64:
        return BwdConfig(
            m_block_size=128, n_block_size=128,
            num_stages_Q=2, num_stages_dO=2, num_stages_PdS=2,
            SdP_swapAB=True, dKV_swapAB=False, dQ_swapAB=False,
            AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=2,
        )
    elif head_dim <= 96:
        return BwdConfig(
            m_block_size=64, n_block_size=128,
            num_stages_Q=2, num_stages_dO=2, num_stages_PdS=2,
            SdP_swapAB=True, dKV_swapAB=False, dQ_swapAB=False,
            AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
            dQ_single_wg=True,
        )
    elif head_dim <= 128:
        is_causal_or_local = causal or local
        m_block_size = 64 if is_causal_or_local else 80
        if sparse_block_size_q is not None and sparse_block_size_q % m_block_size != 0:
            m_block_size = 64
        return BwdConfig(
            m_block_size=m_block_size,
            n_block_size=128,
            num_stages_Q=2, num_stages_dO=2, num_stages_PdS=2,
            SdP_swapAB=True, dKV_swapAB=False,
            dQ_swapAB=m_block_size % 64 != 0,
            AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
        )
    elif head_dim <= 192:
        hdimv128 = head_dim_v <= 128
        if hdimv128:
            return BwdConfig(
                m_block_size=64, n_block_size=96,
                num_stages_Q=2, num_stages_dO=2, num_stages_PdS=1,
                SdP_swapAB=False, dKV_swapAB=True, dQ_swapAB=False,
                AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
                num_wg=2,
            )
        else:
            return BwdConfig(
                m_block_size=64, n_block_size=96,
                num_stages_Q=2, num_stages_dO=1, num_stages_PdS=1,
                SdP_swapAB=False, dKV_swapAB=True, dQ_swapAB=False,
                AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1,
                num_wg=2,
            )
    else:
        # hdim 256
        return BwdConfig(
            m_block_size=64, n_block_size=64,
            num_stages_Q=1, num_stages_dO=1, num_stages_PdS=1,
            SdP_swapAB=False, dKV_swapAB=False, dQ_swapAB=False,
            AtomLayoutMSdP=1, AtomLayoutNdKV=1, AtomLayoutMdQ=1,
        )


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.strides[-1] != 1 else x


def _validate_tensor(t, name, expected_shape, expected_dtype, expected_place):
    assert list(t.shape) == list(expected_shape), f"{name} shape {t.shape} != expected {expected_shape}"
    assert t.dtype == expected_dtype, f"{name} dtype {t.dtype} != expected {expected_dtype}"
    assert str(t.place) == str(expected_place), f"{name} place {t.place} != expected {expected_place}"
    if not is_fake_mode():
        assert t.place.is_gpu_place(), f"{name} must be on CUDA"


def num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, max_splits):
    if num_n_blocks <= 4:
        return 1
    return min(num_SMs // total_mblocks, max_splits, num_n_blocks)


def _resolve_causal_local_window(causal, window_size_left, window_size_right, mask_mod=None):
    """Resolve causal/local/window settings into canonical form."""
    if mask_mod is not None:
        return False, False, window_size_left, window_size_right
    if causal:
        window_size_right = 0
    if window_size_left is not None and window_size_right is not None and window_size_left + window_size_right < 0:
        window_size_left = None
        window_size_right = None
    if window_size_left is not None or window_size_right is not None:
        if window_size_left is None and window_size_right == 0:
            causal, local = True, False
            window_size_right = None
        else:
            causal, local = False, True
    else:
        local = False
    return causal, local, window_size_left, window_size_right


def _flash_attn_fwd(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    cu_seqlens_q: Optional[paddle.Tensor] = None,
    cu_seqlens_k: Optional[paddle.Tensor] = None,
    seqused_q: Optional[paddle.Tensor] = None,
    seqused_k: Optional[paddle.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    page_table: Optional[paddle.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: Optional[float] = None,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    learnable_sink: Optional[paddle.Tensor] = None,
    tile_mn: Optional[Tuple[int, int]] = None,
    mma_pv_is_rs: Optional[bool] = None,
    intra_wg_overlap: Optional[bool] = None,
    num_threads: int = 384,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    _arch: Optional[int] = None,
    score_mod: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsPaddle] = None,
    return_lse: bool = False,
    out: Optional[paddle.Tensor] = None,
    lse: Optional[paddle.Tensor] = None,
    aux_tensors: Optional[list] = None,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """Forward pass for FlashAttention."""
    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    num_head, head_dim = q.shape[-2:]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
        total_q = q.shape[0]
    if page_table is not None:
        assert cu_seqlens_k is None, "page_table is not supported with cu_seqlens_k"
        assert page_table.dtype == paddle.int32, "page_table must be int32"
        assert page_table.strides[-1] == 1, "page_table must be contiguous in the last dimension"
        max_num_pages_per_seq = page_table.shape[1]
        assert list(page_table.shape) == [batch_size, max_num_pages_per_seq]
        num_pages, page_size = k.shape[:2]
        seqlen_k = num_pages * page_size
    else:
        num_pages, page_size = None, None
        seqlen_k = k.shape[-3]
    num_head_kv = k.shape[-2]
    head_dim_v = v.shape[-1]
    if cu_seqlens_k is None:
        if page_table is None:
            assert list(k.shape) == [batch_size, seqlen_k, num_head_kv, head_dim]
            assert list(v.shape) == [batch_size, seqlen_k, num_head_kv, head_dim_v]
        else:
            assert list(k.shape) == [num_pages, page_size, num_head_kv, head_dim]
            assert list(v.shape) == [num_pages, page_size, num_head_kv, head_dim_v]
    else:
        assert list(k.shape) == [seqlen_k, num_head_kv, head_dim]
        assert list(v.shape) == [seqlen_k, num_head_kv, head_dim_v]
        assert list(cu_seqlens_k.shape) == [batch_size + 1], (
            "cu_seqlens_k must have shape (batch_size + 1,)"
        )

    if cu_seqlens_q is not None:
        assert list(cu_seqlens_q.shape) == [batch_size + 1], (
            "cu_seqlens_q must have shape (batch_size + 1,)"
        )
    assert seqused_q is None or list(seqused_q.shape) == [batch_size], (
        "seqused_q must have shape (batch_size,)"
    )
    assert seqused_k is None or list(seqused_k.shape) == [batch_size], (
        "seqused_k must have shape (batch_size,)"
    )
    assert q.dtype in [paddle.float16, paddle.bfloat16], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"
    for t in [cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k]:
        if t is not None:
            assert t.dtype == paddle.int32, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be int32"
            )
            assert t.strides[0] == 1, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be contiguous"
            )
    if learnable_sink is not None:
        assert list(learnable_sink.shape) == [num_head]
        assert learnable_sink.dtype == paddle.bfloat16, "learnable_sink must be bfloat16"

    if not is_fake_mode():
        assert all(
            t is None or t.place.is_gpu_place()
            for t in (
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                seqused_q, seqused_k,
                page_table, learnable_sink,
            )
        ), "inputs must be on CUDA device"
    arch = _get_device_arch() if _arch is None else _arch
    assert arch // 10 in [8, 9, 10, 11, 12], "Unsupported compute capability. Supported: 8.x, 9.x, 10.x, 11.x, 12.x"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    alignment = 16 // q.element_size()
    if arch // 10 not in [8, 12]:
        _validate_head_dims(head_dim, head_dim_v, arch // 10, alignment)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    if softcap == 0.0:
        softcap = None
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1

    out_paddle_dtype = q.dtype
    place = q.place
    q_batch_seqlen_shape = [batch_size, seqlen_q] if cu_seqlens_q is None else [total_q]
    lse_shape = [batch_size, num_head, seqlen_q] if cu_seqlens_q is None else [num_head, total_q]
    requires_grad = not q.stop_gradient or not k.stop_gradient or not v.stop_gradient

    if out is None:
        if arch // 10 == 10 and head_dim == 256 and head_dim_v == 256:
            out = paddle.zeros(q_batch_seqlen_shape + [num_head, head_dim_v], dtype=out_paddle_dtype)
        else:
            out = paddle.empty(q_batch_seqlen_shape + [num_head, head_dim_v], dtype=out_paddle_dtype)
    else:
        _validate_tensor(out, "out", q_batch_seqlen_shape + [num_head, head_dim_v], out_paddle_dtype, place)

    if lse is None:
        lse = (
            paddle.empty(lse_shape, dtype=paddle.float32)
            if requires_grad or return_lse
            else None
        )
    elif lse is not None:
        _validate_tensor(lse, "lse", lse_shape, paddle.float32, place)

    dtype = paddle2cute_dtype_map[q.dtype]
    use_block_sparsity = block_sparse_tensors is not None

    causal, local, window_size_left, window_size_right = _resolve_causal_local_window(
        causal, window_size_left, window_size_right, mask_mod
    )

    requested_use_clc_scheduler = utils._get_use_clc_scheduler_default()
    requested_disable_2cta = utils._get_disable_2cta_default()

    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # SM80/SM120: uses SM80 MMA, 128 threads (4 warps)
    if arch // 10 in [8, 12]:
        num_threads = 128

    fwd_cfg = FwdConfig(128, 128, True, True)  # default
    if tile_mn is None:
        if arch // 10 == 12:
            if head_dim <= 64:
                fwd_cfg = FwdConfig(128, 128, True, True)
            else:
                fwd_cfg = FwdConfig(128, 64, True, True)
        elif arch // 10 == 8:
            fwd_cfg = FwdConfig(128, 64, True, True)
        elif arch // 10 == 9:
            sparse_q = get_sparse_q_block_size(block_sparse_tensors, seqlen_q)
            fwd_cfg = _tile_size_fwd_sm90(head_dim, head_dim_v, causal, local, sparse_block_size_q=sparse_q)
    else:
        fwd_cfg = FwdConfig(tile_mn[0], tile_mn[1], fwd_cfg.mma_pv_is_rs, fwd_cfg.intra_wg_overlap)
    tile_m, tile_n = fwd_cfg.m_block_size, fwd_cfg.n_block_size
    if mma_pv_is_rs is None:
        mma_pv_is_rs = fwd_cfg.mma_pv_is_rs
    if intra_wg_overlap is None:
        intra_wg_overlap = fwd_cfg.intra_wg_overlap

    # TODO: fix GQA + SplitKV + non-varlen
    if pack_gqa and num_splits != 1 and cu_seqlens_q is None:
        pack_gqa = False

    if max_seqlen_q is None:
        max_seqlen_q = seqlen_q if cu_seqlens_q is None else total_q
    if max_seqlen_k is None:
        max_seqlen_k = seqlen_k
    seqlen_q_packgqa = max_seqlen_q * qhead_per_kvhead
    if arch // 10 == 10:
        q_stage = 2 if seqlen_q_packgqa > tile_m else 1
    else:
        q_stage = 1

    m_block_size_effective = q_stage * tile_m
    seqlen_k_loaded = max_seqlen_k if not local else max(0, min(max_seqlen_k, (window_size_right or max_seqlen_k) + (window_size_left or max_seqlen_k) + 1 + tile_m))
    num_m_blocks = (seqlen_q_packgqa + m_block_size_effective - 1) // m_block_size_effective
    total_mblocks = batch_size * num_head_kv * num_m_blocks
    num_n_blocks = (seqlen_k_loaded + tile_n - 1) // tile_n
    num_SMs = 132 if is_fake_mode() else paddle.device.cuda.get_device_properties(place.gpu_device_id()).multi_processor_count
    if num_splits < 1:
        num_splits = num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, 128)

    # SplitKV uses float32 partial output, which doubles the O buffer size
    # in shared memory, causing OOM for diff-headdim (192, 128)
    if arch // 10 in [10, 11] and head_dim != head_dim_v and num_splits > 1:
        if num_n_blocks >= 64:
            tile_n = 64
            num_n_blocks = (seqlen_k_loaded + tile_n - 1) // tile_n
            num_splits = num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, 128)
        else:
            num_splits = 1

    is_split_kv = num_splits > 1
    if is_split_kv:
        out_partial = paddle.empty(
            [num_splits] + q_batch_seqlen_shape + [num_head, head_dim_v], dtype=paddle.float32
        )
        lse_partial = paddle.empty([num_splits] + lse_shape, dtype=paddle.float32)

    use_2cta_instrs = (
        arch // 10 in [10, 11]
        and not requested_disable_2cta
        and not causal
        and not local
        and not is_split_kv
        and cu_seqlens_q is None
        and seqused_q is None
        and not use_block_sparsity
        and page_size in [None, 128]
        and int(math.ceil(head_dim / 16) * 16) in [128, 192]
        and int(math.ceil(head_dim_v / 16) * 16) == 128
        and seqlen_q_packgqa > 2 * tile_m
        and (tile_m % qhead_per_kvhead == 0 or not pack_gqa)
    )

    # hd=256 2CTA forward uses dedicated kernel (SM100 only)
    use_dedicated_hd256_kernel = arch // 10 == 10 and head_dim == 256 and head_dim_v == 256
    use_2cta_instrs = use_2cta_instrs or use_dedicated_hd256_kernel

    if softcap is not None:
        assert score_mod is None, "softcap and score_mod cannot be used together"
        score_mod = utils.create_softcap_scoremod(softcap)
    elif score_mod is not None:
        if arch // 10 == 8:
            raise NotImplementedError("Custom user-provided score_mod is not supported on SM8x architectures.")

    # hash score and mask mods for compile cache
    score_mod_hash = utils.hash_callable(score_mod) if score_mod is not None else False
    mask_mod_hash = utils.hash_callable(mask_mod) if mask_mod is not None else False

    is_varlen = (
        cu_seqlens_q is not None
        or cu_seqlens_k is not None
        or seqused_q is not None
        or seqused_k is not None
    )

    if mask_mod is not None:
        if is_varlen:
            raise NotImplementedError(
                "mask_mod with aux_tensors is not yet supported for varlen sequences. This will be fixed in a future PR."
            )

    if use_block_sparsity:
        if is_varlen:
            raise NotImplementedError(
                "Block sparsity is not yet supported for varlen sequences. This will be fixed in a future PR."
            )
        # NB: pack_gqa requires block sparse head dim == 1 (broadcasted)
        if pack_gqa and block_sparse_tensors.mask_block_cnt.shape[1] != 1:
            pack_gqa = False
        if is_split_kv:
            raise NotImplementedError(
                "Block sparsity is not yet supported with SplitKV. TODO: partition sparse block lists per split."
            )

    # See get_broadcast_dims for why this is needed in compile key
    block_sparse_broadcast_pattern = None
    normalized_block_sparse_tensors = None
    q_subtile_factor = None
    if block_sparse_tensors is not None:
        if seqlen_q is None:
            raise ValueError("Block sparsity requires fixed-length sequences (seqlen_q must be known).")
        (
            normalized_block_sparse_tensors,
            block_sparse_broadcast_pattern,
            q_subtile_factor,
        ) = normalize_block_sparse_config(
            block_sparse_tensors,
            batch_size=batch_size,
            num_head=num_head,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            block_size=(tile_m, tile_n),
            q_stage=q_stage,
        )
    if aux_tensors is not None:
        aux_tensor_metadata = get_aux_tensor_metadata(aux_tensors)
    else:
        aux_tensor_metadata = None

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        score_mod_hash,
        mask_mod_hash,
        use_block_sparsity,
        block_sparse_broadcast_pattern,
        aux_tensor_metadata,
        lse is None,
        cu_seqlens_q is None,
        cu_seqlens_k is None,
        seqused_q is None,
        seqused_k is None,
        page_table is not None,
        window_size_left is not None,
        window_size_right is not None,
        learnable_sink is not None,
        tile_m,
        tile_n,
        q_stage,
        num_threads,
        is_split_kv,
        pack_gqa,
        arch,
        page_size not in [None, tile_n],  # paged KV non-TMA
        use_2cta_instrs,
        q_subtile_factor,
        mma_pv_is_rs,
        intra_wg_overlap,
        requested_use_clc_scheduler,
        fa_logging.get_fa_log_level(),
    )
    if compile_key not in _flash_attn_fwd.compile_cache:
        (
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            learnable_sink_tensor,
        ) = [
            to_cute_tensor(t, assumed_align=4, leading_dim=0)
            if t is not None
            else None
            for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink)
        ]
        page_table_tensor = (
            to_cute_tensor(page_table, assumed_align=4, leading_dim=1)
            if page_table is not None
            else None
        )
        q_tensor, k_tensor, v_tensor, o_tensor = [
            to_cute_tensor(t) for t in (q, k, v, out if not is_split_kv else out_partial)
        ]
        if is_split_kv:
            lse_tensor = to_cute_tensor(lse_partial, assumed_align=4)
        elif lse is not None:
            lse_tensor = to_cute_tensor(lse, assumed_align=4)
        else:
            lse_tensor = None

        sparse_tensors = None
        if normalized_block_sparse_tensors is not None:
            sparse_tensors = to_cute_block_sparse_tensors(normalized_block_sparse_tensors)

        cute_aux_tensors = None
        aux_tensor_metadata = None
        if aux_tensors is not None:
            cute_aux_tensors = [to_cute_aux_tensor(buf) for buf in aux_tensors]

        if arch // 10 == 8:
            assert page_table is None, "paged KV not supported on SM 8.0"
            assert not is_split_kv, "SplitKV not supported on SM 8.0"
            fa_fwd = FlashAttentionForwardSm80(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                pack_gqa=pack_gqa,
                tile_m=tile_m,
                tile_n=tile_n,
                num_stages=1,
                num_threads=num_threads,
                Q_in_regs=False,
                score_mod=score_mod,
                mask_mod=mask_mod,
                has_aux_tensors=aux_tensors is not None,
            )
        elif arch // 10 == 9:
            assert not is_split_kv, "SplitKV not supported on SM 9.0"
            fa_fwd = FlashAttentionForwardSm90(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                pack_gqa=pack_gqa,
                tile_m=tile_m,
                tile_n=tile_n,
                num_stages=2,
                num_threads=num_threads,
                Q_in_regs=False,
                intra_wg_overlap=intra_wg_overlap,
                mma_pv_is_rs=mma_pv_is_rs,
                mask_mod=mask_mod,
                score_mod=score_mod,
                has_aux_tensors=aux_tensors is not None,
                q_subtile_factor=q_subtile_factor,
                paged_kv_non_tma=page_size not in [None, tile_n],
            )
        elif arch // 10 in [10, 11]:
            if use_dedicated_hd256_kernel:
                # hd=256 2CTA forward: check for currently unsupported features
                assert softcap is None, "SM100 forward with head_dim=256 does not support softcap"
                assert not use_block_sparsity, \
                    "SM100 forward with head_dim=256 does not support block sparsity"
                assert learnable_sink is None, \
                    "SM100 forward with head_dim=256 does not support learnable_sink"
                assert seqused_q is None and seqused_k is None, \
                    "SM100 forward with head_dim=256 does not support seqused_q/seqused_k"
                # pack_gqa is an auto-selected optimization; disable it for hd256 kernel
                pack_gqa = False
            flash_fwd_obj_cls = (
                BlackwellFusedMultiHeadAttentionForward
                if use_dedicated_hd256_kernel
                else FlashAttentionForwardSm100
            )
            fa_fwd = flash_fwd_obj_cls(
                head_dim,
                head_dim_v,
                qhead_per_kvhead=qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                is_split_kv=is_split_kv,
                pack_gqa=pack_gqa,
                m_block_size=tile_m,
                n_block_size=tile_n,
                q_stage=q_stage,
                is_persistent=not causal
                    and not local
                    and cu_seqlens_q is None
                    and seqused_q is None
                    and not is_split_kv,
                score_mod=score_mod,
                mask_mod=mask_mod,
                has_aux_tensors=aux_tensors is not None,
                paged_kv_non_tma=page_size not in [None, tile_n],
                is_varlen_q=cu_seqlens_q is not None or seqused_q is not None,
                q_subtile_factor=q_subtile_factor,
                use_2cta_instrs=use_2cta_instrs,
                use_clc_scheduler=requested_use_clc_scheduler,
            )
        elif arch // 10 == 12:
            assert not use_block_sparsity, "Block sparsity not supported on SM 12.0"
            assert page_table is None, "Paged KV not supported on SM 12.0 in this PR"
            assert not is_split_kv, "SplitKV not supported on SM 12.0 in this PR"
            fa_fwd = FlashAttentionForwardSm120(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                is_causal=causal,
                is_local=local,
                pack_gqa=pack_gqa,
                tile_m=tile_m,
                tile_n=tile_n,
                num_stages=1,
                num_threads=num_threads,
                Q_in_regs=False,
                score_mod=score_mod,
                mask_mod=mask_mod,
                has_aux_tensors=aux_tensors is not None,
            )
        else:
            raise ValueError(
                f"Unsupported compute capability: {arch}. Supported: 8.x, 9.x, 10.x, 11.x, 12.x"
            )
        # TODO: check @can_implement
        _flash_attn_fwd.compile_cache[compile_key] = cute.compile(
            fa_fwd,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            page_table_tensor,
            window_size_left,
            window_size_right,
            learnable_sink_tensor,
            sparse_tensors,
            cute_aux_tensors,
            current_stream,
            options="--enable-tvm-ffi",
        )

    # In "fake mode", skip actual kernel invocation; just return the pre-allocated output.
    if not is_fake_mode():
        _flash_attn_fwd.compile_cache[compile_key](
            q.detach(),
            k.detach(),
            v.detach(),
            out.detach() if not is_split_kv else out_partial,
            lse_partial if is_split_kv else lse,
            softmax_scale,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            page_table,
            window_size_left,
            window_size_right,
            learnable_sink,
            normalized_block_sparse_tensors[:4] if normalized_block_sparse_tensors is not None else None,
            aux_tensors,
        )
    if is_split_kv:
        _flash_attn_fwd_combine(
            out_partial,
            _swap_last_two_dims(lse_partial),
            out,
            _swap_last_two_dims(lse) if lse is not None else None,
            cu_seqlens_q,
            seqused_q,
        )
    return out, lse


_flash_attn_fwd.compile_cache = get_jit_cache("fwd")


def make_fake_bwd_tensors(dtype, has_gqa, varlen_q, varlen_k):
    sym = cute.sym_int
    div = 128 // dtype.width  # 8 for fp16/bf16
    b, seqlen_q, seqlen_k, h_q, d, d_v = sym(), sym(), sym(), sym(), sym(), sym()
    h_kv = h_q if not has_gqa else sym()
    seqlen_q_rounded, seqlen_k_rounded = sym(), sym()
    seqlen_q_d_rounded, seqlen_k_d_rounded, seqlen_k_dv_rounded = sym(), sym(), sym()
    total_q, total_k, total_q_rounded, total_k_rounded = sym(), sym(), sym(), sym()
    total_q_d_rounded, total_k_d_rounded, total_k_dv_rounded = sym(), sym(), sym()
    b_seqlenq = (b, seqlen_q) if not varlen_q else (total_q,)
    b_seqlenk = (b, seqlen_k) if not varlen_k else (total_k,)
    mQ = fake_tensor(dtype, (*b_seqlenq, h_q, d), divisibility=div)
    mO = fake_tensor(dtype, (*b_seqlenq, h_q, d_v), divisibility=div)
    mdO = fake_tensor(dtype, (*b_seqlenq, h_q, d_v), divisibility=div)
    mK = fake_tensor(dtype, (*b_seqlenk, h_kv, d), divisibility=div)
    mV = fake_tensor(dtype, (*b_seqlenk, h_kv, d_v), divisibility=div)
    mdQ = fake_tensor(dtype, (*b_seqlenq, h_q, d), divisibility=div)
    mdK = fake_tensor(dtype, (*b_seqlenk, h_kv, d), divisibility=div)
    mdV = fake_tensor(dtype, (*b_seqlenk, h_kv, d_v), divisibility=div)
    if not varlen_q:
        mLSE = fake_tensor(Float32, (b, h_q, seqlen_q), divisibility=1)
        mLSElog2 = fake_tensor(Float32, (b, h_q, seqlen_q_rounded), divisibility=4)
        mPdPsum = fake_tensor(Float32, (b, h_q, seqlen_q_rounded), divisibility=4)
        dQaccum = fake_tensor(Float32, (b, h_q, seqlen_q_d_rounded), divisibility=4)
    else:
        mLSE = fake_tensor(Float32, (h_q, total_q), divisibility=1)
        mLSElog2 = fake_tensor(Float32, (h_q, total_q_rounded), divisibility=4)
        mPdPsum = fake_tensor(Float32, (h_q, total_q_rounded), divisibility=4)
        dQaccum = fake_tensor(Float32, (h_q, total_q_d_rounded), divisibility=4)
    if not has_gqa:
        mdKaccum, mdVaccum = None, None
    else:
        if not varlen_k:
            mdKaccum = fake_tensor(Float32, (b, h_kv, seqlen_k_rounded), divisibility=4)
            mdVaccum = fake_tensor(Float32, (b, h_kv, seqlen_k_dv_rounded), divisibility=4)
        else:
            mdKaccum = fake_tensor(Float32, (h_kv, total_k_rounded), divisibility=4)
            mdVaccum = fake_tensor(Float32, (h_kv, total_k_dv_rounded), divisibility=4)
    return mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, dQaccum, mdKaccum, mdVaccum


def _compile_bwd_preprocess(
    dtype, head_dim, head_dim_v, m_block_size, has_cuseqlens_q, has_seqused_q, has_dlse, has_dq_accum,
    use_padded_offsets,
):
    """Compile bwd preprocess kernel using cute fake tensors."""
    mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, mdQaccum, mdKaccum, mdVaccum = make_fake_bwd_tensors(
        dtype, has_gqa=True, varlen_q=has_cuseqlens_q, varlen_k=False
    )
    batch = mQ.shape[0] if not has_cuseqlens_q else cute.sym_int()
    batchp1 = cute.sym_int()
    mCuSeqlensQ = fake_tensor(Int32, (batchp1,), divisibility=1) if has_cuseqlens_q else None
    mSequsedQ = fake_tensor(Int32, (batch,), divisibility=1) if has_seqused_q else None
    mdLSE = fake_tensor(Float32, mLSE.shape, divisibility=1) if has_dlse else None
    mdQaccum = mdQaccum if has_dq_accum else None
    fa_bwd_pre = FlashAttentionBackwardPreprocess(dtype, head_dim, head_dim_v, m_block_size, use_padded_offsets=use_padded_offsets)
    return cute.compile(
        fa_bwd_pre, mO, mdO, mPdPsum, mLSE, mLSElog2, mdQaccum, mCuSeqlensQ, mSequsedQ, mdLSE,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _bwd_preprocess(
    out, dout, dpsum, lse, lse_log2, dq_accum,
    cu_seqlens_q, seqused_q, dlse,
    dtype, head_dim, head_dim_v, m_block_size,
    use_padded_offsets=True,
):
    """Backward preprocess: compute (o * dout).sum(dim=-1) - dLSE, lse * log2_e, zero out dq_accum."""
    is_varlen = cu_seqlens_q is not None
    compile_key = (
        dtype, head_dim, head_dim_v, m_block_size, is_varlen, seqused_q is not None, dlse is not None,
        dq_accum is not None, use_padded_offsets,
    )
    if compile_key not in _bwd_preprocess.compile_cache:
        _bwd_preprocess.compile_cache[compile_key] = _compile_bwd_preprocess(*compile_key)
    if not is_fake_mode():
        _bwd_preprocess.compile_cache[compile_key](
            out, dout, dpsum, lse, lse_log2, dq_accum, cu_seqlens_q, seqused_q, dlse
        )


_bwd_preprocess.compile_cache = get_jit_cache("bwd_pre")


def _compile_bwd_postprocess(
    dtype, hdim, block_size, num_threads, atom_layout, swap_ab,
    has_cuseqlens_q, has_seqused_q,
    use_2cta_instrs, cluster_size, arch,
):
    """Compile bwd postprocess kernel using cute fake tensors."""
    mQ, mK, mV, mO, mdO, mdQ, mdK, mdV, mLSE, mLSElog2, mPdPsum, mdQaccum, mdKaccum, mdVaccum = make_fake_bwd_tensors(
        dtype, has_gqa=True, varlen_q=has_cuseqlens_q, varlen_k=False
    )
    batch = mQ.shape[0] if not has_cuseqlens_q else cute.sym_int()
    batchp1 = cute.sym_int()
    mCuSeqlensQ = fake_tensor(Int32, (batchp1,), divisibility=1) if has_cuseqlens_q else None
    mSeqUsedQ = fake_tensor(Int32, (batch,), divisibility=1) if has_seqused_q else None
    fa_bwd_post = FlashAttentionBackwardPostprocess(
        dtype, hdim, arch, block_size, num_threads, atom_layout, swap_ab,
        use_2cta_instrs=use_2cta_instrs,
        cluster_size=cluster_size,
    )
    return cute.compile(
        fa_bwd_post, mdQaccum, mdQ, Float32(0.0), mCuSeqlensQ, mSeqUsedQ,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _bwd_postprocess_convert(
    accum, output, scale,
    cu_seqlens, seqused,
    arch, dtype, hdim, block_size, num_threads,
    atom_layout, swap_ab,
    use_2cta_instrs=False, cluster_size=1,
):
    """Backward postprocess: convert float32 accumulator to bf16/fp16 output."""
    compile_key = (
        dtype, hdim, block_size, num_threads, atom_layout, swap_ab,
        cu_seqlens is not None, seqused is not None,
        use_2cta_instrs, cluster_size, arch,
    )
    if compile_key not in _bwd_postprocess_convert.compile_cache:
        _bwd_postprocess_convert.compile_cache[compile_key] = _compile_bwd_postprocess(*compile_key)
    if not is_fake_mode():
        _bwd_postprocess_convert.compile_cache[compile_key](
            accum, output, scale, cu_seqlens, seqused,
        )


_bwd_postprocess_convert.compile_cache = get_jit_cache("bwd_post")


def _flash_attn_bwd(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    out: paddle.Tensor,
    dout: paddle.Tensor,
    lse: paddle.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: float = 0.0,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    m_block_size: int = 64,
    n_block_size: int = 128,
    num_threads: int = 256,
    pack_gqa: bool = False,
    num_stages_Q: int = 2,
    num_stages_dO: int = 2,
    SdP_swapAB: bool = False,
    dKV_swapAB: bool = False,
    dQ_swapAB: bool = False,
    AtomLayoutMSdP: int = 2,
    AtomLayoutNdKV: int = 2,
    AtomLayoutMdQ: int = 2,
    V_in_regs: bool = False,
    cu_seqlens_q: Optional[paddle.Tensor] = None,
    cu_seqlens_k: Optional[paddle.Tensor] = None,
    seqused_q: Optional[paddle.Tensor] = None,
    seqused_k: Optional[paddle.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    deterministic: bool = False,
    dq: Optional[paddle.Tensor] = None,
    dk: Optional[paddle.Tensor] = None,
    dv: Optional[paddle.Tensor] = None,
    score_mod: Optional[Callable] = None,
    score_mod_bwd: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsPaddle] = None,
    dlse: Optional[paddle.Tensor] = None,
) -> Tuple[paddle.Tensor, paddle.Tensor, paddle.Tensor]:
    arch = _get_device_arch()
    assert arch // 10 in [9, 10, 11, 12], "Unsupported compute capability. Supported: 9.x, 10.x, 11.x, 12.x"
    sparse_q = None
    if block_sparse_tensors is not None and arch // 10 == 9:
        sparse_q = block_sparse_tensors.block_size[0] if block_sparse_tensors.block_size is not None else 128

    num_head, head_dim = q.shape[-2:]
    head_dim_v = v.shape[-1]

    causal, local, window_size_left, window_size_right = _resolve_causal_local_window(
        causal, window_size_left, window_size_right
    )

    if arch // 10 == 12:
        m_block_size = 64
        n_block_size = 64
        if head_dim <= 64:
            num_stages_Q = 2
            num_stages_dO = 2
        else:
            num_stages_Q = 1
            num_stages_dO = 1
        SdP_swapAB = False
        dKV_swapAB = False
        dQ_swapAB = False
        AtomLayoutMSdP = 4
        AtomLayoutNdKV = 4
        AtomLayoutMdQ = 4
        V_in_regs = False
        cluster_size = 1
        use_2cta_instrs = False
        num_threads = 128
        dQ_single_wg = False
        assert not (block_sparse_tensors is not None), "Block sparsity backward not supported on SM 12.0"
        assert score_mod is None and score_mod_bwd is None, "score_mod backward not supported on SM 12.0"
        assert mask_mod is None, "mask_mod backward not supported on SM 12.0"
        assert deterministic is False, "deterministic backward not supported on SM 12.0"
    elif arch // 10 == 9:
        cfg = _tile_size_bwd_sm90(
            head_dim,
            head_dim_v,
            causal,
            local,
            sparse_block_size_q=sparse_q,
        )
        m_block_size = cfg.m_block_size
        n_block_size = cfg.n_block_size
        num_stages_Q = cfg.num_stages_Q
        num_stages_dO = cfg.num_stages_dO
        num_stages_PdS = cfg.num_stages_PdS
        SdP_swapAB = cfg.SdP_swapAB
        dKV_swapAB = cfg.dKV_swapAB
        dQ_swapAB = cfg.dQ_swapAB
        AtomLayoutMSdP = cfg.AtomLayoutMSdP
        AtomLayoutNdKV = cfg.AtomLayoutNdKV
        AtomLayoutMdQ = cfg.AtomLayoutMdQ
        num_threads = (cfg.num_wg + 1) * 128
        dQ_single_wg = cfg.dQ_single_wg
        cluster_size = 1
        use_2cta_instrs = False
    else:
        m_block_size = 128
        n_block_size = 128
        dQ_swapAB = False
        dKV_swapAB = False
        AtomLayoutMdQ = 1
        AtomLayoutNdKV = 1
        dQ_single_wg = False
        requested_disable_2cta = utils._get_disable_2cta_default()
        disable_2cta = (
            requested_disable_2cta
            or score_mod is not None
            or score_mod_bwd is not None
            or mask_mod is not None
            or block_sparse_tensors is not None
        )
        cluster_size = 2 if head_dim >= 128 and not disable_2cta else 1
        use_2cta_instrs = cluster_size == 2

    use_dedicated_hd256_kernel = arch // 10 == 10 and head_dim == 256 and head_dim_v == 256
    use_2cta_instrs = use_2cta_instrs or use_dedicated_hd256_kernel

    q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = [
        maybe_contiguous(t)
        for t in (q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
    ]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        total_q = q.shape[0]
        seqlen_q = max_seqlen_q if max_seqlen_q is not None else total_q

    if cu_seqlens_k is None:
        batch_size, seqlen_k = k.shape[:2]
        total_k = batch_size * seqlen_k
    else:
        batch_size = cu_seqlens_k.shape[0] - 1
        total_k = k.shape[0]
        seqlen_k = max_seqlen_k if max_seqlen_k is not None else total_k

    num_head_kv = k.shape[-2]

    use_block_sparsity = block_sparse_tensors is not None
    subtile_factor = sparse_q // m_block_size if sparse_q is not None else 2
    seqlen_q_rounded = (seqlen_q + m_block_size - 1) // m_block_size * m_block_size
    seqlen_k_rounded = (seqlen_k + n_block_size - 1) // n_block_size * n_block_size
    num_n_blocks = seqlen_k_rounded // n_block_size
    if cluster_size == 2 and num_n_blocks % cluster_size != 0:
        seqlen_k_rounded = seqlen_k_rounded + n_block_size

    if cu_seqlens_k is None:
        assert list(k.shape) == [batch_size, seqlen_k, num_head_kv, head_dim]
        assert list(v.shape) == [batch_size, seqlen_k, num_head_kv, head_dim_v]
    else:
        assert list(k.shape) == [total_k, num_head_kv, head_dim]
        assert list(v.shape) == [total_k, num_head_kv, head_dim_v]
        assert list(cu_seqlens_k.shape) == [batch_size + 1], (
            "cu_seqlens_k must have shape (batch_size + 1,)"
        )

    if cu_seqlens_q is not None:
        assert list(cu_seqlens_q.shape) == [batch_size + 1], (
            "cu_seqlens_q must have shape (batch_size + 1,)"
        )
        assert list(out.shape) == [total_q, num_head, head_dim_v]
        assert list(dout.shape) == [total_q, num_head, head_dim_v]
        assert list(lse.shape) == [num_head, total_q], "lse must have shape (num_head, total_q)"
    else:
        assert list(out.shape) == [batch_size, seqlen_q, num_head, head_dim_v]
        assert list(dout.shape) == [batch_size, seqlen_q, num_head, head_dim_v]
        assert list(lse.shape) == [batch_size, num_head, seqlen_q], (
            "lse must have shape (batch_size, num_head, seqlen_q)"
        )

    assert q.dtype in [paddle.float16, paddle.bfloat16], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype, (
        "inputs must have the same dtype"
    )
    for t in [cu_seqlens_q, cu_seqlens_k]:
        if t is not None:
            assert t.dtype == paddle.int32, "cu_seqlens_q, cu_seqlens_k must be int32"
    assert lse.dtype == paddle.float32, "lse must be float32"
    if dlse is not None:
        dlse = maybe_contiguous(dlse)
    if not is_fake_mode():
        assert all(
            t is None or t.place.is_gpu_place()
            for t in (q, k, v, out, dout, lse, cu_seqlens_q, cu_seqlens_k)
        ), "inputs must be on CUDA device"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    alignment = 16 // q.element_size()
    if arch // 10 != 12:
        _validate_head_dims(head_dim, head_dim_v, arch // 10, alignment)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1
    # pack_gqa backward not yet supported
    pack_gqa = False

    if softcap != 0.0:
        assert score_mod is None and score_mod_bwd is None, (
            "softcap and score_mod/score_mod_bwd cannot be used together"
        )
        score_mod = utils.create_softcap_scoremod(softcap)
        score_mod_bwd = utils.create_softcap_scoremod_bwd(softcap)
    elif score_mod is not None:
        assert score_mod_bwd is not None, "score_mod_bwd is required when score_mod is provided"
        assert cu_seqlens_q is None and cu_seqlens_k is None, (
            "varlen + score_mod not supported in bwd yet"
        )
        if arch // 10 == 8:
            raise NotImplementedError("Custom user-provided score_mod is not supported on SM8x architectures.")

    place = q.place
    out_paddle_dtype = q.dtype

    if dq is None:
        if use_dedicated_hd256_kernel:
            dq = paddle.zeros_like(q)
        else:
            dq = paddle.empty_like(q)
    else:
        _validate_tensor(dq, "dq", q.shape, out_paddle_dtype, place)

    if dk is None:
        dk = paddle.empty_like(k)
    else:
        _validate_tensor(dk, "dk", k.shape, out_paddle_dtype, place)

    if dv is None:
        dv = paddle.empty_like(v)
    else:
        _validate_tensor(dv, "dv", v.shape, out_paddle_dtype, place)

    head_dim_rounded = (head_dim + 32 - 1) // 32 * 32

    if cu_seqlens_q is None:
        dq_accum = (
            None
            if use_dedicated_hd256_kernel
            else paddle.empty(
                [batch_size, num_head, seqlen_q_rounded * head_dim_rounded],
                dtype=paddle.float32,
            )
        )
        dpsum = paddle.empty([batch_size, num_head, seqlen_q_rounded], dtype=paddle.float32)
        lse_log2 = paddle.empty([batch_size, num_head, seqlen_q_rounded], dtype=paddle.float32)
    else:
        total_q_rounded_padded = (
            (total_q + cu_seqlens_q.shape[0] * m_block_size - 1) // m_block_size * m_block_size
        )
        dq_accum = (
            None
            if use_dedicated_hd256_kernel
            else paddle.empty(
                [num_head, total_q_rounded_padded * head_dim_rounded], dtype=paddle.float32
            )
        )
        dpsum = paddle.empty([num_head, total_q_rounded_padded], dtype=paddle.float32)
        lse_log2 = paddle.empty([num_head, total_q_rounded_padded], dtype=paddle.float32)

    # GQA needs dK/dV accum+postprocess since multiple Q heads accumulate into the same dK/dV.
    # hd=256 2CTA backward has its own internal postprocess for dK/dV.
    dKV_postprocess = qhead_per_kvhead > 1 and not use_dedicated_hd256_kernel
    if dKV_postprocess:
        head_dim_v_rounded = (head_dim_v + 32 - 1) // 32 * 32
        if cu_seqlens_k is None:
            dk_accum = paddle.zeros(
                [batch_size, num_head_kv, seqlen_k_rounded * head_dim_rounded],
                dtype=paddle.float32,
            )
            dv_accum = paddle.zeros(
                [batch_size, num_head_kv, seqlen_k_rounded * head_dim_v_rounded],
                dtype=paddle.float32,
            )
        else:
            cluster_tile_n = cluster_size * n_block_size
            total_k_rounded_padded = (
                (total_k + cu_seqlens_k.shape[0] * cluster_tile_n - 1) // cluster_tile_n * cluster_tile_n
            )
            dk_accum = paddle.zeros(
                [num_head_kv, total_k_rounded_padded * head_dim_rounded],
                dtype=paddle.float32,
            )
            dv_accum = paddle.zeros(
                [num_head_kv, total_k_rounded_padded * head_dim_v_rounded],
                dtype=paddle.float32,
            )

    dtype = paddle2cute_dtype_map[q.dtype]
    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    if deterministic:
        dQ_semaphore = paddle.zeros(
            [batch_size, num_head, seqlen_q_rounded // m_block_size, cluster_size],
            dtype=paddle.int32,
        )
    else:
        dQ_semaphore = None

    if deterministic and qhead_per_kvhead > 1:
        dK_semaphore = paddle.zeros(
            [batch_size, num_head_kv, seqlen_k_rounded // n_block_size, 2],
            dtype=paddle.int32,
        )
        dV_semaphore = paddle.zeros(
            [batch_size, num_head_kv, seqlen_k_rounded // n_block_size, 2],
            dtype=paddle.int32,
        )
    else:
        dK_semaphore = None
        dV_semaphore = None

    # Preprocess kernel: compute (o * dout).sum(dim=-1) - dLSE, lse * log2_e, and zero out dq_accum.
    # For hd=256 dedicated path, dq_accum is None so preprocess only fills dpsum/lse_log2.
    _bwd_preprocess(
        out, dout, dpsum, lse, lse_log2, dq_accum,
        cu_seqlens_q, seqused_q, dlse,
        dtype, head_dim, head_dim_v, m_block_size,
        use_padded_offsets=use_dedicated_hd256_kernel,
    )
    if arch // 10 not in [9, 12]:
        num_threads = 384

    # Backward kernel: compute dk, dv, dq_accum.
    score_mod_hash = utils.hash_callable(score_mod) if score_mod else False
    score_mod_bwd_hash = utils.hash_callable(score_mod_bwd) if score_mod_bwd else False
    mask_mod_hash = utils.hash_callable(mask_mod) if mask_mod else False
    num_aux_tensors = len(aux_tensors) if aux_tensors else 0
    cute_aux_tensors = None
    if aux_tensors is not None:
        cute_aux_tensors = [to_cute_tensor(buf, assumed_align=None, fully_dynamic=True) for buf in aux_tensors]

    block_sparse_broadcast_pattern = None
    normalized_block_sparse_tensors = None
    if block_sparse_tensors is not None:
        (
            normalized_block_sparse_tensors,
            block_sparse_broadcast_pattern,
        ) = normalize_block_sparse_config_bwd(
            block_sparse_tensors,
            batch_size=batch_size,
            num_head=num_head,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            block_size=(m_block_size, n_block_size),
            subtile_factor=subtile_factor,
        )

    if arch // 10 in [8, 9, 12]:
        compile_key = (
            arch,
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            causal,
            window_size_left is not None,
            window_size_right is not None,
            m_block_size,
            n_block_size,
            num_threads,
            pack_gqa,
            num_stages_Q,
            num_stages_dO,
            SdP_swapAB,
            dKV_swapAB,
            dQ_swapAB,
            AtomLayoutMSdP,
            AtomLayoutNdKV,
            AtomLayoutMdQ,
            V_in_regs,
            dQ_single_wg,
            deterministic,
            cu_seqlens_q is None,
            cu_seqlens_k is None,
            seqused_q is None,
            seqused_k is None,
            score_mod_hash,
            score_mod_bwd_hash,
            mask_mod_hash,
            num_aux_tensors,
            use_block_sparsity,
            block_sparse_broadcast_pattern,
            get_broadcast_dims(q),
            get_broadcast_dims(k),
            get_broadcast_dims(v),
            get_broadcast_dims(dout),
            (seqlen_q_rounded // m_block_size == 1),
            (seqlen_k_rounded // n_block_size == 1),
        )
    else:
        compile_key = (
            arch,
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            causal,
            window_size_left is not None,
            window_size_right is not None,
            m_block_size,
            n_block_size,
            num_threads,
            pack_gqa,
            cluster_size,
            use_2cta_instrs,
            deterministic,
            score_mod_hash,
            score_mod_bwd_hash,
            mask_mod_hash,
            num_aux_tensors,
            use_block_sparsity,
            block_sparse_broadcast_pattern,
            cu_seqlens_q is None,
            cu_seqlens_k is None,
            seqused_q is None,
            seqused_k is None,
            get_broadcast_dims(q),
            get_broadcast_dims(k),
            get_broadcast_dims(v),
            get_broadcast_dims(dout),
            (seqlen_q_rounded // m_block_size == 1),
            (seqlen_k_rounded // n_block_size == 1),
        )
    if compile_key not in _flash_attn_bwd.compile_cache:
        q_tensor, k_tensor, v_tensor, do_tensor, dq_tensor, dk_tensor, dv_tensor = [
            to_cute_tensor(t) for t in (q, k, v, dout, dq, dk, dv)
        ]
        lse_log2_tensor, dpsum_tensor = [to_cute_tensor(t) for t in (lse_log2, dpsum)]
        dq_accum_tensor = to_cute_tensor(dq_accum) if dq_accum is not None else None
        if dKV_postprocess:
            dk_accum_tensor, dv_accum_tensor = [
                to_cute_tensor(t) for t in (dk_accum, dv_accum)
            ]
        cu_seqlens_q_tensor, cu_seqlens_k_tensor, seqused_q_tensor, seqused_k_tensor = [
            to_cute_tensor(t, assumed_align=4) if t is not None else None
            for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
        ]
        dQ_semaphore_tensor, dK_semaphore_tensor, dV_semaphore_tensor = [
            utils.convert_from_dlpack_leading_static(
                t.detach(), leading_dim=3, alignment=4, stride_order=_dim_order(t)
            )
            if t is not None else None
            for t in (dQ_semaphore, dK_semaphore, dV_semaphore)
        ]
        if arch // 10 in [8, 12]:
            flash_bwd_obj_cls = FlashAttentionBackwardSm120 if arch // 10 == 12 else FlashAttentionBackwardSm80
            fa_bwd_obj = flash_bwd_obj_cls(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                m_block_size,
                n_block_size,
                num_stages_Q,
                num_stages_dO,
                num_threads,
                pack_gqa,
                causal,
                SdP_swapAB,
                dKV_swapAB,
                dQ_swapAB,
                AtomLayoutMSdP,
                AtomLayoutNdKV,
                AtomLayoutMdQ,
                V_in_regs=V_in_regs,
                score_mod=score_mod,
                score_mod_bwd=score_mod_bwd,
            )
        elif arch // 10 == 9:
            fa_bwd_obj = FlashAttentionBackwardSm90(
                dtype,
                head_dim,
                head_dim_v,
                qhead_per_kvhead,
                causal,
                is_local=local,
                deterministic=deterministic,
                tile_m=m_block_size,
                tile_n=n_block_size,
                Q_stage=num_stages_Q,
                dO_stage=num_stages_dO,
                PdS_stage=num_stages_PdS,
                SdP_swapAB=SdP_swapAB,
                dKV_swapAB=dKV_swapAB,
                dQ_swapAB=dQ_swapAB,
                AtomLayoutMSdP=AtomLayoutMSdP,
                AtomLayoutNdKV=AtomLayoutNdKV,
                AtomLayoutMdQ=AtomLayoutMdQ,
                num_threads=num_threads,
                V_in_regs=V_in_regs,
                score_mod=score_mod,
                score_mod_bwd=score_mod_bwd,
                mask_mod=mask_mod,
                has_aux_tensors=aux_tensors is not None,
                subtile_factor=subtile_factor,
                dQ_single_wg=dQ_single_wg,
            )
        else:
            if use_dedicated_hd256_kernel:
                assert softcap == 0.0, "SM100 backward with head_dim=256 does not support softcap"
                assert block_sparse_tensors is None, \
                    "SM100 backward with head_dim=256 does not support block sparsity"
                assert dlse is None, \
                    "SM100 backward with head_dim=256 does not support dlse"
                assert seqused_q is None and seqused_k is None, \
                    "SM100 backward with head_dim=256 does not support seqused_q/seqused_k"

                dq_tile_mn = (128, 128)
                dkdv_tile_mn = (128, 64)
                fa_bwd_obj = BlackwellFusedMultiHeadAttentionBackward(
                    head_dim,
                    head_dim_v,
                    is_causal=causal,
                    is_local=local,
                    qhead_per_kvhead=qhead_per_kvhead,
                    is_persistent=False,
                    deterministic=deterministic,
                    cluster_size=cluster_size,
                    use_2cta_instrs=use_2cta_instrs,
                    score_mod=score_mod,
                    score_mod_bwd=score_mod_bwd,
                    mask_mod=mask_mod,
                    has_aux_tensors=aux_tensors is not None,
                    subtile_factor=subtile_factor,
                    tile_m_dq=dq_tile_mn[0],
                    tile_n_dq=dq_tile_mn[1],
                    tile_m_dkdv=dkdv_tile_mn[0],
                    tile_n_dkdv=dkdv_tile_mn[1],
                )
            else:
                fa_bwd_obj = FlashAttentionBackwardSm100(
                    head_dim,
                    head_dim_v,
                    is_causal=causal,
                    is_local=local,
                    qhead_per_kvhead=qhead_per_kvhead,
                    tile_m=m_block_size,
                    tile_n=n_block_size,
                    cluster_size=cluster_size,
                    use_2cta_instrs=use_2cta_instrs,
                    deterministic=deterministic,
                    score_mod=score_mod,
                    score_mod_bwd=score_mod_bwd,
                    mask_mod=mask_mod,
                    has_aux_tensors=aux_tensors is not None,
                    subtile_factor=subtile_factor,
                )

        sparse_tensors_compile = None
        if normalized_block_sparse_tensors is not None:
            sparse_tensors_compile = to_cute_block_sparse_tensors(normalized_block_sparse_tensors)

        # For hd=256, pass dq (bf16/fp16) directly at the dQ_accum slot.
        dq_accum_tensor = dq_tensor if use_dedicated_hd256_kernel else dq_accum_tensor

        _flash_attn_bwd.compile_cache[compile_key] = cute.compile(
            fa_bwd_obj,
            q_tensor,
            k_tensor,
            v_tensor,
            do_tensor,
            lse_log2_tensor,
            dpsum_tensor,
            dq_accum_tensor,
            dk_tensor if not dKV_postprocess else dk_accum_tensor,
            dv_tensor if not dKV_postprocess else dv_accum_tensor,
            softmax_scale,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            window_size_left,
            window_size_right,
            dQ_semaphore_tensor,
            dK_semaphore_tensor,
            dV_semaphore_tensor,
            cute_aux_tensors,
            sparse_tensors_compile,
            current_stream,
            options="--enable-tvm-ffi",
        )
    if not is_fake_mode():
        # For hd=256, dq_accum is None; pass dq directly at the dQ_accum slot.
        dq_accum = dq if use_dedicated_hd256_kernel else dq_accum
        _flash_attn_bwd.compile_cache[compile_key](
            q.detach(),
            k.detach(),
            v.detach(),
            dout,
            lse_log2,
            dpsum,
            dq_accum,
            dk if not dKV_postprocess else dk_accum,
            dv if not dKV_postprocess else dv_accum,
            softmax_scale,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            window_size_left,
            window_size_right,
            dQ_semaphore,
            dK_semaphore,
            dV_semaphore,
            aux_tensors,
            normalized_block_sparse_tensors[:4] if normalized_block_sparse_tensors is not None else None,
        )

    # Postprocess: convert dq_accum from float32 to dq in bf16/fp16
    # hd=256 2CTA backward has its own internal postprocess, skip here.
    if not use_dedicated_hd256_kernel:
        if arch // 10 == 9:
            num_threads_post_dQ = 128 if dQ_single_wg else cfg.num_wg * 128
            num_threads_post_dKV = cfg.num_wg * 128
        else:
            num_threads_post_dQ = 128
            num_threads_post_dKV = 128

        _bwd_postprocess_convert(
            dq_accum, dq, softmax_scale,
            cu_seqlens_q, seqused_q,
            arch, dtype, head_dim, m_block_size, num_threads_post_dQ,
            AtomLayoutMdQ, dQ_swapAB,
            use_2cta_instrs=use_2cta_instrs, cluster_size=1,
        )

        if dKV_postprocess:
            _bwd_postprocess_convert(
                dk_accum, dk, softmax_scale,
                cu_seqlens_k, seqused_k,
                arch, dtype, head_dim, n_block_size, num_threads_post_dKV,
                AtomLayoutNdKV, dKV_swapAB,
                cluster_size=cluster_size,
            )
            _bwd_postprocess_convert(
                dv_accum, dv, 1.0,
                cu_seqlens_k, seqused_k,
                arch, dtype, head_dim_v, n_block_size, num_threads_post_dKV,
                AtomLayoutNdKV, dKV_swapAB,
                cluster_size=cluster_size,
            )

    return dq, dk, dv


_flash_attn_bwd.compile_cache = get_jit_cache("bwd")


class FlashAttnFunc(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
        ctx,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        learnable_sink: Optional[paddle.Tensor] = None,
        softcap: float = 0.0,
        num_splits: int = 1,
        pack_gqa: Optional[bool] = None,
        deterministic: bool = False,
        mask_mod: Optional[Callable] = None,
        full_block_cnt: Optional[paddle.Tensor] = None,
        full_block_idx: Optional[paddle.Tensor] = None,
        mask_block_cnt: Optional[paddle.Tensor] = None,
        mask_block_idx: Optional[paddle.Tensor] = None,
        block_size: Optional[Tuple[int, int]] = None,
        return_lse: bool = False,
    ):
        block_sparse_tensors = None
        if any(t is not None for t in [full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx]):
            block_sparse_tensors = BlockSparseTensorsPaddle(
                full_block_cnt=full_block_cnt,
                full_block_idx=full_block_idx,
                mask_block_cnt=mask_block_cnt,
                mask_block_idx=mask_block_idx,
                block_size=block_size,
            )
        out, lse = _flash_attn_fwd(
            q, k, v,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            learnable_sink=learnable_sink,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            mask_mod=mask_mod,
            block_sparse_tensors=block_sparse_tensors,
            return_lse=return_lse,
        )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.return_lse = return_lse
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        q, k, v, out, lse = ctx.saved_tensor()
        if not ctx.return_lse:
            dlse = None
        if dout is None:
            dout = paddle.zeros_like(out)
        dq, dk, dv = _flash_attn_bwd(
            q, k, v, out, dout, lse,
            ctx.softmax_scale, ctx.causal, ctx.softcap,
            window_size_left=ctx.window_size[0],
            window_size_right=ctx.window_size[1],
            deterministic=ctx.deterministic,
            dlse=dlse,
        )
        # 18 inputs to forward (excluding ctx): q,k,v + 15 non-differentiable args
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None


class FlashAttnVarlenFunc(paddle.autograd.PyLayer):
    @staticmethod
    def forward(
        ctx,
        q: paddle.Tensor,
        k: paddle.Tensor,
        v: paddle.Tensor,
        cu_seqlens_q: Optional[paddle.Tensor],
        cu_seqlens_k: Optional[paddle.Tensor],
        seqused_q: Optional[paddle.Tensor] = None,
        seqused_k: Optional[paddle.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
        page_table: Optional[paddle.Tensor] = None,
        softmax_scale: Optional[float] = None,
        causal: bool = False,
        window_size: Tuple[Optional[int], Optional[int]] = (None, None),
        learnable_sink: Optional[paddle.Tensor] = None,
        softcap: float = 0.0,
        num_splits: int = 1,
        pack_gqa: Optional[bool] = None,
        deterministic: bool = False,
        score_mod: Optional[Callable] = None,
        aux_tensors: Optional[list] = None,
        return_lse: bool = False,
    ):
        out, lse = _flash_attn_fwd(
            q, k, v,
            cu_seqlens_q, cu_seqlens_k,
            seqused_q, seqused_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            page_table=page_table,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            learnable_sink=learnable_sink,
            softcap=softcap,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            score_mod=score_mod,
            aux_tensors=aux_tensors,
            return_lse=return_lse,
        )
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.return_lse = return_lse
        # Track which optional Tensor inputs are non-None so backward can
        # return the correct number of gradients (Paddle PyLayer only counts
        # actual Tensor inputs, unlike PyTorch autograd.Function).
        ctx.has_seqused_q = seqused_q is not None
        ctx.has_seqused_k = seqused_k is not None
        ctx.has_page_table = page_table is not None
        ctx.has_learnable_sink = learnable_sink is not None
        return out, lse

    @staticmethod
    def backward(ctx, dout, dlse):
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = ctx.saved_tensor()
        if not ctx.return_lse:
            dlse = None
        if dout is None:
            dout = paddle.zeros_like(out)
        dq, dk, dv = _flash_attn_bwd(
            q, k, v, out, dout, lse,
            ctx.softmax_scale, ctx.causal, ctx.softcap,
            window_size_left=ctx.window_size[0],
            window_size_right=ctx.window_size[1],
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            max_seqlen_q=ctx.max_seqlen_q,
            max_seqlen_k=ctx.max_seqlen_k,
            deterministic=ctx.deterministic,
            dlse=dlse,
        )
        # Paddle PyLayer backward must return exactly one gradient per *Tensor*
        # input to forward (scalars, bools, tuples, and None values are excluded
        # from the count, unlike PyTorch autograd.Function).
        # Required tensors in forward arg order: q, k, v, cu_seqlens_q, cu_seqlens_k
        # Optional tensors (also in forward arg order):
        #   seqused_q, seqused_k, page_table, learnable_sink
        grads = [dq, dk, dv, None, None]
        if ctx.has_seqused_q:
            grads.append(None)
        if ctx.has_seqused_k:
            grads.append(None)
        if ctx.has_page_table:
            grads.append(None)
        if ctx.has_learnable_sink:
            grads.append(None)
        return tuple(grads)


def flash_attn_func(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[paddle.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    mask_mod: Optional[Callable] = None,
    full_block_cnt: Optional[paddle.Tensor] = None,
    full_block_idx: Optional[paddle.Tensor] = None,
    mask_block_cnt: Optional[paddle.Tensor] = None,
    mask_block_idx: Optional[paddle.Tensor] = None,
    block_size: Optional[Tuple[int, int]] = None,
    return_lse: bool = False,
):
    return FlashAttnFunc.apply(
        q, k, v,
        softmax_scale, causal, window_size,
        learnable_sink, softcap, num_splits, pack_gqa, deterministic,
        mask_mod, full_block_cnt, full_block_idx, mask_block_cnt, mask_block_idx,
        block_size, return_lse,
    )


def flash_attn_varlen_func(
    q: paddle.Tensor,
    k: paddle.Tensor,
    v: paddle.Tensor,
    cu_seqlens_q: Optional[paddle.Tensor] = None,
    cu_seqlens_k: Optional[paddle.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    seqused_q: Optional[paddle.Tensor] = None,
    seqused_k: Optional[paddle.Tensor] = None,
    page_table: Optional[paddle.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[Optional[int], Optional[int]] = (None, None),
    learnable_sink: Optional[paddle.Tensor] = None,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    score_mod: Optional[Callable] = None,
    aux_tensors: Optional[list] = None,
    return_lse: bool = False,
):
    return FlashAttnVarlenFunc.apply(
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        seqused_q, seqused_k,
        max_seqlen_q, max_seqlen_k,
        page_table,
        softmax_scale, causal, window_size,
        learnable_sink, softcap, num_splits, pack_gqa, deterministic,
        score_mod, aux_tensors, return_lse,
    )


def _compile_fwd_combine(
    dtype, dtype_partial, head_dim, tile_m, k_block_size, log_max_splits,
    has_cu_seqlens, has_seqused, has_lse, has_varlen_batch_idx,
):
    """Compile fwd combine kernel using cute fake tensors (no real GPU tensors needed)."""
    sym = cute.sym_int
    div = 128 // dtype_partial.width  # 16-byte alignment in elements

    fa_combine = FlashAttentionForwardCombine(
        dtype=dtype,
        dtype_partial=dtype_partial,
        head_dim=head_dim,
        tile_m=tile_m,
        k_block_size=k_block_size,
        log_max_splits=log_max_splits,
    )
    if not fa_combine.can_implement(
        dtype, dtype_partial, head_dim, tile_m, k_block_size, log_max_splits,
        num_threads=256,
    ):
        raise RuntimeError(
            "FlashAttention combine kernel cannot be implemented with given parameters"
        )

    if has_cu_seqlens:
        num_splits, total_q, nheads = sym(), sym(), sym()
        mO_partial = fake_tensor(dtype_partial, (num_splits, total_q, nheads, head_dim), divisibility=div)
        mLSE_partial = fake_tensor(Float32, (num_splits, total_q, nheads), divisibility=1, leading_dim=1)
        mO = fake_tensor(dtype, (total_q, nheads, head_dim), divisibility=div)
        mLSE = fake_tensor(Float32, (total_q, nheads), divisibility=1, leading_dim=0) if has_lse else None
    else:
        num_splits, batch, seqlen, nheads = sym(), sym(), sym(), sym()
        mO_partial = fake_tensor(dtype_partial, (num_splits, batch, seqlen, nheads, head_dim), divisibility=div)
        mLSE_partial = fake_tensor(Float32, (num_splits, batch, seqlen, nheads), divisibility=1, leading_dim=2)
        mO = fake_tensor(dtype, (batch, seqlen, nheads, head_dim), divisibility=div)
        mLSE = fake_tensor(Float32, (batch, seqlen, nheads), divisibility=1, leading_dim=1) if has_lse else None
        batch = mO_partial.shape[1]

    batch_for_1d = batch if not has_cu_seqlens else sym()
    batchp1 = sym()
    mCuSeqlens = fake_tensor(Int32, (batchp1,), divisibility=1) if has_cu_seqlens else None
    mSeqused = fake_tensor(Int32, (batch_for_1d,), divisibility=1) if has_seqused else None
    mNumSplitsDynamic = None
    mVarlenBatchIdx = fake_tensor(Int32, (batch_for_1d,), divisibility=1) if has_varlen_batch_idx else None
    mSemaphore = None

    return cute.compile(
        fa_combine,
        mO_partial, mLSE_partial, mO, mLSE,
        mCuSeqlens, mSeqused, mNumSplitsDynamic, mVarlenBatchIdx, mSemaphore,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _flash_attn_fwd_combine(
    out_partial: paddle.Tensor,
    lse_partial: paddle.Tensor,
    out: paddle.Tensor,
    lse: Optional[paddle.Tensor] = None,
    cu_seqlens: Optional[paddle.Tensor] = None,
    seqused: Optional[paddle.Tensor] = None,
    num_splits_dynamic_ptr: Optional[paddle.Tensor] = None,
    varlen_batch_idx: Optional[paddle.Tensor] = None,
    semaphore_to_reset: Optional[paddle.Tensor] = None,
) -> None:
    """Forward combine kernel for split attention computation."""
    assert out_partial.dtype in [paddle.float16, paddle.bfloat16, paddle.float32], (
        "out_partial must be fp16, bf16, or fp32"
    )
    if not is_fake_mode():
        assert out_partial.place.is_gpu_place() and lse_partial.place.is_gpu_place(), (
            "tensors must be on CUDA device"
        )
    is_varlen = len(out_partial.shape) == 4
    for t, name in [
        (cu_seqlens, "cu_seqlens"),
        (seqused, "seqused"),
        (num_splits_dynamic_ptr, "num_splits_dynamic_ptr"),
    ]:
        if t is not None:
            if not is_fake_mode():
                assert t.place.is_gpu_place(), f"{name} must be on CUDA device"
            assert t.is_contiguous(), f"{name} must be contiguous"
    head_dim = out_partial.shape[-1]
    num_splits = out_partial.shape[0]
    assert num_splits <= 256
    k_block_size = 64 if head_dim <= 64 else 128
    tile_m = 8 if k_block_size % 128 == 0 else (16 if k_block_size % 64 == 0 else 32)
    log_max_splits = max(math.ceil(math.log2(num_splits)), 4)
    if tile_m == 8:
        log_max_splits = max(log_max_splits, 5)

    dtype = paddle2cute_dtype_map[out.dtype]
    dtype_partial = paddle2cute_dtype_map[out_partial.dtype]
    compile_key = (
        dtype, dtype_partial, head_dim, tile_m, k_block_size, log_max_splits,
        cu_seqlens is not None, seqused is not None, lse is not None,
        varlen_batch_idx is not None,
    )
    if compile_key not in _flash_attn_fwd_combine.compile_cache:
        _flash_attn_fwd_combine.compile_cache[compile_key] = _compile_fwd_combine(*compile_key)
    if not is_fake_mode():
        _flash_attn_fwd_combine.compile_cache[compile_key](
            out_partial, lse_partial, out, lse,
            cu_seqlens, seqused, num_splits_dynamic_ptr, varlen_batch_idx,
            semaphore_to_reset,
        )


_flash_attn_fwd_combine.compile_cache = get_jit_cache("fwd_combine")


def flash_attn_combine(
    out_partial: paddle.Tensor,
    lse_partial: paddle.Tensor,
    out: Optional[paddle.Tensor] = None,
    out_dtype=None,
    cu_seqlens: Optional[paddle.Tensor] = None,
    seqused: Optional[paddle.Tensor] = None,
    varlen_batch_idx: Optional[paddle.Tensor] = None,
    return_lse: bool = True,
) -> Tuple[paddle.Tensor, Optional[paddle.Tensor]]:
    """Flash Attention combine function for split attention computation."""
    assert len(out_partial.shape) in [4, 5], "out_partial must have 4 or 5 dimensions"
    is_varlen = len(out_partial.shape) == 4
    if is_varlen:
        num_splits, total_q, num_heads, head_size = out_partial.shape
        batch_size = 1
        seqlen = total_q
    else:
        num_splits, batch_size, seqlen, num_heads, head_size = out_partial.shape
    if out_dtype is None:
        out_dtype = out_partial.dtype
    if out is None:
        if is_varlen:
            out = paddle.empty([total_q, num_heads, head_size], dtype=out_dtype)
        else:
            out = paddle.empty([batch_size, seqlen, num_heads, head_size], dtype=out_dtype)
    if return_lse:
        if is_varlen:
            lse = paddle.empty([num_heads, total_q], dtype=paddle.float32)
        else:
            lse = paddle.empty([batch_size, num_heads, seqlen], dtype=paddle.float32)
        lse = _swap_last_two_dims(lse)
    else:
        lse = None
    _flash_attn_fwd_combine(
        out_partial, lse_partial, out, lse,
        cu_seqlens, seqused,
        varlen_batch_idx=varlen_batch_idx,
    )
    return out, lse
