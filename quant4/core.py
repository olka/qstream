"""MXFP4 E2M1 quantization core.

Block size: 32 elements per scale (e8m0 biased exponent).
Scale selection: MSE-optimal over {floor-1, floor, floor+1} candidate exponents.
Activation-aware: optional γ-weighted MSE (γ = input_layernorm.weight).
"""

import torch

BLOCK_SIZE = 32  # elements per MXFP4 scale

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

    # --- Final quantization with selected exponents ---
    biased_exp = (raw_exp + 127).clamp(0, 254).to(torch.uint8)
    actual_scale = torch.pow(2.0, raw_exp)

    scaled_final = (t_blocked / actual_scale.unsqueeze(-1)).clamp(-6.0, 6.0)
    abs_final = scaled_final.abs().clamp(max=6.0)
    bucket = torch.searchsorted(_BOUNDARIES.to(t.device), abs_final.reshape(-1))
    sign_mask = (scaled_final.reshape(-1) < 0).to(torch.uint8) * 8
    codes = (bucket.to(torch.uint8) + sign_mask).reshape(*leading, num_blocks, BLOCK_SIZE)

    codes_flat = codes.reshape(*leading, K)
    packed = codes_flat[..., 0::2] | (codes_flat[..., 1::2] << 4)

    return packed.to(torch.uint8), biased_exp
