"""Quantization core: MXFP4 E2M1, NVFP4 E2M1, and FP8 E4M3.

MXFP4: Block size 32, E8M0 power-of-2 scale, MSE-optimal scale selection,
       activation-aware γ-weighting.
NVFP4: Block size 16, two-level scaling — per-block E4M3 × per-tensor FP32 global,
       amax-based (NVIDIA reference recipe).
FP8:   Per-channel or per-tensor scales, simple amax-based.
"""

import torch

BLOCK_SIZE = 32

# NVFP4: 16-element blocks, E4M3 block scale, per-tensor FP32 global scale.
NVFP4_BLOCK_SIZE = 16
FP4_E2M1_MAX = 6.0    # max representable E2M1 magnitude
FP8_E4M3_MAX = 448.0  # max representable E4M3 magnitude (= torch.finfo(float8_e4m3fn).max)

# MXFP4 E2M1 representable positive values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
# Boundaries are midpoints between consecutive values.
_POS_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
_BOUNDARIES = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)


def compute_block_hessian(
    X: torch.Tensor,
    block_size: int = BLOCK_SIZE,
    damp: float = 1e-6,
) -> torch.Tensor:
    """Compute block-diagonal Hessian H_b = X_b^T @ X_b / N for each block.

    Used for AWQ-style scale selection: minimizing reconstruction error
    trace(dW @ H @ dW^T) instead of simple MSE.

    Args:
        X: [N, in_features] activation matrix (tokens routed to this weight).
        block_size: Block size for MXFP4 quantization (default 32).
        damp: Dampening added to diagonal for numerical stability when N is small.

    Returns:
        H: [num_blocks, block_size, block_size] symmetric positive semi-definite.
    """
    assert X.ndim == 2, f"Expected 2D activation matrix, got shape {X.shape}"
    N, K = X.shape
    assert K % block_size == 0, f"in_features {K} not divisible by block_size {block_size}"

    num_blocks = K // block_size
    X_b = X.float().reshape(N, num_blocks, block_size)
    H = torch.einsum('nbs,nbt->bst', X_b, X_b) / max(N, 1)
    H.diagonal(dim1=-2, dim2=-1).add_(damp)
    return H


def _round_to_mxfp4(scaled: torch.Tensor) -> torch.Tensor:
    """Round values in [-6, 6] to the nearest MXFP4 representable value.

    Returns dequantized floats (not codes). Shape-preserving.
    """
    abs_scaled = scaled.abs().clamp(max=6.0)
    bucket = torch.searchsorted(_BOUNDARIES.to(scaled.device), abs_scaled.reshape(-1))
    dequant_abs = _POS_VALUES.to(scaled.device)[bucket].reshape_as(abs_scaled)
    return dequant_abs * scaled.sign()


def _pack_e2m1(scaled: torch.Tensor, leading, K: int) -> torch.Tensor:
    """Round scaled values in [-6, 6] to E2M1 codes and pack two nibbles per byte.

    Shared by MXFP4 and NVFP4 — only the scale differs, the element encoding does not.

    Args:
        scaled:  [*leading, num_blocks, block_size] values already divided by scale.
        leading: the leading dims (everything before the block axes).
        K:       num_blocks * block_size (the un-blocked trailing size).
    Returns:
        [*leading, K//2] uint8 — codes interleaved as (lo | hi << 4).
    """
    abs_s = scaled.abs().clamp(max=6.0)
    bucket = torch.searchsorted(_BOUNDARIES.to(scaled.device), abs_s.reshape(-1))
    sign_mask = (scaled.reshape(-1) < 0).to(torch.uint8) * 8
    codes = (bucket.to(torch.uint8) + sign_mask).reshape(*leading, K)
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)


def dequant_mxfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Dequantize MXFP4 packed weights back to float32.

    Args:
        packed: [..., in//2] uint8 — two 4-bit codes per byte.
        scales: [..., in//BLOCK_SIZE] uint8 — e8m0 biased exponents.
        shape: Original weight shape (e.g., [out, in]).

    Returns:
        Reconstructed float32 tensor with the given shape.
    """
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    codes = torch.stack([lo, hi], dim=-1).reshape(shape)
    sign = ((codes >> 3) & 1).float() * -2 + 1
    mag_idx = (codes & 0x07).long().reshape(-1)
    magnitude = _POS_VALUES.to(packed.device)[mag_idx].reshape(codes.shape)
    raw_exp = scales.float() - 127
    scale_vals = torch.pow(2.0, raw_exp)
    scale_expanded = scale_vals.repeat_interleave(BLOCK_SIZE, dim=-1)
    return sign * magnitude * scale_expanded


def quantize_mxfp4(
    tensor: torch.Tensor,
    scale_percentile: float = 99.5,
    gamma: torch.Tensor | None = None,
    hessian: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a float tensor to MXFP4 format with MSE-optimal scale selection.

    For each block of BLOCK_SIZE input channels, three candidate scale exponents
    {floor-1, floor, floor+1} are evaluated and the one minimizing per-block
    quantization error is selected.

    Scale selection priority:
        1. hessian: Hessian-weighted reconstruction error (AWQ-style).
           Minimizes trace(dW @ H @ dW^T) — output-space error.
        2. gamma: γ²-weighted MSE (activation magnitude proxy).
        3. Neither: unweighted MSE.

    Args:
        tensor: Shape [out_features, in_features]. Last dim divisible by BLOCK_SIZE.
        scale_percentile: Percentile used to anchor the initial block_max estimate.
                          Still matters as the center of the candidate range.
        gamma: Shape [in_features]. Per-channel activation magnitude proxy.
               When provided, MSE is weighted by γ² so high-activation channels
               drive scale selection.
        hessian: Shape [num_blocks, BLOCK_SIZE, BLOCK_SIZE]. Block-diagonal Hessian
                 from compute_block_hessian(). Mutually exclusive with gamma.

    Returns:
        packed: [out, in//2] uint8 — two 4-bit codes per byte.
        scales: [out, in//BLOCK_SIZE] uint8 — e8m0 biased exponents.
    """
    if hessian is not None and gamma is not None:
        raise ValueError("hessian and gamma are mutually exclusive")
    assert tensor.shape[-1] % BLOCK_SIZE == 0, (
        f"Last dim {tensor.shape[-1]} not divisible by BLOCK_SIZE={BLOCK_SIZE}"
    )

    t = tensor.to(torch.float32)
    *leading, K = t.shape
    num_blocks = K // BLOCK_SIZE

    if hessian is not None:
        assert hessian.shape == (num_blocks, BLOCK_SIZE, BLOCK_SIZE), (
            f"hessian shape {hessian.shape} != expected ({num_blocks}, {BLOCK_SIZE}, {BLOCK_SIZE})"
        )

    t_blocked = t.reshape(*leading, num_blocks, BLOCK_SIZE)  # [..., B, 32]

    # --- Anchor estimate of block_max for candidate generation ---
    # Always use unweighted amax so outlier channels are never suppressed by gamma.
    # gamma (if provided) only influences the MSE error weighting below.
    if gamma is not None:
        gamma_b = gamma.reshape(num_blocks, BLOCK_SIZE)                  # [B, 32]
    if scale_percentile >= 100.0:
        block_max = t_blocked.abs().amax(dim=-1)
    else:
        block_max = torch.quantile(t_blocked.abs(), scale_percentile / 100.0, dim=-1)

    # Zero out near-zero blocks: structured sparsity in expert weights produces
    # blocks with max ~1e-7 that quantize to nonzero MXFP4 codes (noise).
    near_zero = block_max < 1e-6
    t_blocked = torch.where(near_zero.unsqueeze(-1), torch.zeros_like(t_blocked), t_blocked)
    block_max = block_max.clamp(min=1e-6)

    # --- MSE-optimal exponent selection over 3 candidates ---
    exp_floor = torch.floor(torch.log2(block_max / 6.0))               # [..., B]
    candidates = torch.stack(
        [exp_floor - 1, exp_floor, exp_floor + 1], dim=-1
    )                                                                    # [..., B, 3]

    # Vectorized: evaluate all 3 candidates simultaneously
    t_cand = t_blocked.unsqueeze(-2)                                    # [..., B, 1, 32]
    scales = torch.pow(2.0, candidates).unsqueeze(-1)                   # [..., B, 3, 1]
    scaled = (t_cand / scales).clamp(-6.0, 6.0)                        # [..., B, 3, 32]

    dequant_orig = _round_to_mxfp4(scaled) * scales                    # [..., B, 3, 32]
    dW = dequant_orig - t_cand                                          # [..., B, 3, 32]

    if hessian is not None:
        # Hessian-weighted reconstruction error: trace(dW @ H @ dW^T) per row
        # dW: [out, B, 3, 32], hessian: [B, 32, 32]
        weighted = torch.einsum('obcs,bsd->obcd', dW, hessian.to(t.device))
        err = (weighted * dW).sum(dim=-1)                               # [out, B, 3]
    elif gamma is not None:
        # γ²-weighted MSE
        err = (dW ** 2 * (gamma_b ** 2)[None, :, None, :]).mean(dim=-1)
    else:
        err = (dW ** 2).mean(dim=-1)                                    # [..., B, 3]

    best_idx = err.argmin(dim=-1, keepdim=True)                        # [..., B, 1]
    raw_exp = candidates.gather(-1, best_idx).squeeze(-1)               # [..., B]

    # Safety: only correct overflow when the safe exponent lies OUTSIDE the candidate
    # range (i.e. block_max was grossly underestimated by gamma/percentile so all
    # three candidates are too small).  When exp_floor+1 is already in the candidate
    # set the 3-candidate MSE already accounts for saturation correctly and may
    # legitimately prefer a slightly-overflowing exp_floor (benign clamping gives the
    # same reconstruction as the rounded result at exp_floor+1).
    actual_scale = torch.pow(2.0, raw_exp)
    has_overflow = (t_blocked.abs() / actual_scale.unsqueeze(-1) > 6.0).any(dim=-1)
    if has_overflow.any():
        actual_max = t_blocked.abs().amax(dim=-1)                          # [..., B]
        safe_exp = torch.ceil(torch.log2(actual_max.clamp(min=1e-12) / 6.0))
        # Benign overflow: raw_exp == safe_exp - 1 (one step below safe).
        # The block_max is at most 12× the scale; saturation gives the same
        # FP4 reconstruction as rounding at safe_exp, so the MSE decision is correct.
        # Catastrophic overflow: raw_exp < safe_exp - 1 (γ²-MSE or gamma-block_max
        # pushed candidates too low; outlier is 2× or more out of FP4 range).
        needs_correction = has_overflow & (raw_exp < safe_exp - 1)
        raw_exp = torch.where(needs_correction, safe_exp, raw_exp)

    biased_exp = (raw_exp + 127).clamp(0, 254).to(torch.uint8)
    actual_scale = torch.pow(2.0, raw_exp)

    scaled_final = (t_blocked / actual_scale.unsqueeze(-1)).clamp(-6.0, 6.0)
    packed = _pack_e2m1(scaled_final, leading, K)

    return packed, biased_exp


# ---------------------------------------------------------------------------
# NVFP4 E2M1 quantization (compressed-tensors nvfp4-pack-quantized)
# ---------------------------------------------------------------------------


def _nvfp4_block_scale(
    block_amax: torch.Tensor, global_scale_b: torch.Tensor
) -> torch.Tensor:
    """Per-block E4M3 scale from block amax and the per-tensor global scale.

    Reference recipe (vLLM ``ref_nvfp4_quant``): ``scale = global · amax / 6``,
    clamped to the E4M3 range and cast to ``float8_e4m3fn``. v1 is amax-based; a
    continuous / γ-weighted block-scale search would replace this helper without
    touching any caller.

    Args:
        block_amax:     [..., num_blocks] per-block max abs value.
        global_scale_b: broadcastable to block_amax (per-tensor global scale).
    Returns:
        float8_e4m3fn block scales, same shape as block_amax.
    """
    scale = global_scale_b * (block_amax / FP4_E2M1_MAX)
    scale = scale.clamp(min=-FP8_E4M3_MAX, max=FP8_E4M3_MAX)
    return scale.to(torch.float8_e4m3fn)


def quantize_nvfp4(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a float tensor to NVFP4 (compressed-tensors ``nvfp4-pack-quantized``).

    Two-level scaling: a per-tensor FP32 global scale and a per-16-block E4M3 scale.
    The global scale is stored in the **CT reciprocal convention** ``(6·448)/amax``;
    vLLM applies ``1/x`` on load. Amax-based (NVIDIA reference recipe) — ``gamma`` is
    intentionally unused at this layer (see _nvfp4_block_scale for the extension point).

    Args:
        tensor: Shape [..., out, in] — 2D, or 3D batched over experts. Last dim
                divisible by NVFP4_BLOCK_SIZE (16). The global scale is reduced over
                ``(out, in)`` per leading "expert" index: one scalar for a 2D weight,
                one per expert for a 3D [E, out, in] weight.

    Returns:
        packed:       [..., out, in//2] uint8 — two E2M1 codes per byte.
        block_scale:  [..., out, in//16] float8_e4m3fn — per-block scales.
        global_scale: [..., 1] float32 — per-tensor global, stored as (6·448)/amax
                      (2D → shape [1]; 3D → shape [E, 1], one per expert).
    """
    assert tensor.shape[-1] % NVFP4_BLOCK_SIZE == 0, (
        f"Last dim {tensor.shape[-1]} not divisible by NVFP4_BLOCK_SIZE={NVFP4_BLOCK_SIZE}"
    )
    assert tensor.ndim >= 2, f"Expected 2D or 3D tensor, got {tensor.ndim}D"

    t = tensor.to(torch.float32)
    *batch, OUT, K = t.shape
    num_blocks = K // NVFP4_BLOCK_SIZE

    # Per-tensor (per-expert) global scale: amax over (out, in). Stored as 2688/amax
    # (CT reciprocal convention). Shape [*batch, 1] so a 2D weight yields [1] and a
    # 3D weight yields [E, 1] — exactly one scalar per emitted logical weight.
    amax = t.abs().amax(dim=(-2, -1)).clamp(min=1e-12)               # [*batch]
    global_scale = ((FP4_E2M1_MAX * FP8_E4M3_MAX) / amax).reshape(*batch, 1)

    t_blocked = t.reshape(*batch, OUT, num_blocks, NVFP4_BLOCK_SIZE)
    block_amax = t_blocked.abs().amax(dim=-1)                        # [*batch, OUT, nb]
    near_zero = block_amax < 1e-12

    gs_block = global_scale.reshape(*batch, 1, 1)                    # broadcast over OUT, nb
    block_scale = _nvfp4_block_scale(block_amax, gs_block)          # e4m3 [*batch, OUT, nb]

    bs_f32 = block_scale.to(torch.float32)
    safe_bs = torch.where(bs_f32 == 0, torch.ones_like(bs_f32), bs_f32)
    output_scale = gs_block / safe_bs                               # [*batch, OUT, nb]
    scaled = (t_blocked * output_scale.unsqueeze(-1)).clamp(-FP4_E2M1_MAX, FP4_E2M1_MAX)
    scaled = torch.where(near_zero.unsqueeze(-1), torch.zeros_like(scaled), scaled)

    packed = _pack_e2m1(scaled, (*batch, OUT), K)
    return packed, block_scale, global_scale.to(torch.float32)


def dequant_nvfp4(
    packed: torch.Tensor,
    block_scale: torch.Tensor,
    global_scale: torch.Tensor,
    shape: tuple[int, ...],
) -> torch.Tensor:
    """Dequantize NVFP4 packed weights back to float32.

    Reconstruction: ``e2m1(code) · block_scale / global_scale`` (global stored as
    2688/amax, so we divide). Mirrors vLLM's ``dequantize_to_dtype`` with
    ``weight_global_scale = 1/global`` applied on load.

    Args:
        packed:       [..., out, in//2] uint8 — two 4-bit codes per byte.
        block_scale:  [..., out, in//16] float8_e4m3fn — per-block scales.
        global_scale: [..., 1] float32 — per-tensor global (2688/amax).
        shape:        Original weight shape (e.g. [out, in] or [E, out, in]).
    """
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    codes = torch.stack([lo, hi], dim=-1).reshape(shape)
    sign = ((codes >> 3) & 1).float() * -2 + 1
    mag_idx = (codes & 0x07).long().reshape(-1)
    magnitude = _POS_VALUES.to(packed.device)[mag_idx].reshape(codes.shape)

    bs_expanded = block_scale.to(torch.float32).repeat_interleave(NVFP4_BLOCK_SIZE, dim=-1)
    gs = global_scale.to(torch.float32).reshape(*global_scale.shape[:-1], 1, 1)
    return sign * magnitude * bs_expanded / gs


# ---------------------------------------------------------------------------
# FP8 E4M3 quantization
# ---------------------------------------------------------------------------

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448.0


def quantize_fp8(
    tensor: torch.Tensor,
    per_channel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a float tensor to FP8 E4M3 format.

    Args:
        tensor: Shape [out_features, in_features] (2D weight matrix).
        per_channel: If True, compute one scale per output row (per-channel).
                     If False, compute a single scalar scale (per-tensor).

    Returns:
        quantized: [out, in] float8_e4m3fn weights.
        scale:     [out, 1] float32 per-channel scales, or [1, 1] per-tensor scale.
                   Weight ≈ quantized * scale (dequantization).
    """
    t = tensor.float()
    if per_channel:
        amax = t.abs().amax(dim=-1, keepdim=True)  # [out, 1]
    else:
        amax = t.abs().amax().unsqueeze(0).unsqueeze(0)  # [1, 1]
    scale = (amax / FP8_MAX).clamp(min=1e-12)
    quantized = (t / scale).to(torch.float8_e4m3fn)
    # Keep [out, 1] shape — vLLM's ChannelQuantScaleParameter expects it
    return quantized, scale
