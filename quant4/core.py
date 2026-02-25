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


def _round_to_mxfp4(scaled: torch.Tensor) -> torch.Tensor:
    """Round values in [-6, 6] to the nearest MXFP4 representable value.

    Returns dequantized floats (not codes). Shape-preserving.
    """
    abs_scaled = scaled.abs().clamp(max=6.0)
    bucket = torch.searchsorted(_BOUNDARIES.to(scaled.device), abs_scaled.reshape(-1))
    dequant_abs = _POS_VALUES.to(scaled.device)[bucket].reshape_as(abs_scaled)
    return dequant_abs * scaled.sign()


def quantize_mxfp4(
    tensor: torch.Tensor,
    scale_percentile: float = 99.5,
    gamma: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a float tensor to MXFP4 format with MSE-optimal scale selection.

    For each block of BLOCK_SIZE input channels, three candidate scale exponents
    {floor-1, floor, floor+1} are evaluated and the one minimizing per-block
    quantization MSE is selected. This outperforms the round(log2(...)) heuristic
    for heavy-tailed blocks where rounding overestimates the needed scale.

    Args:
        tensor: Shape [out_features, in_features]. Last dim divisible by BLOCK_SIZE.
        scale_percentile: Percentile used to anchor the initial block_max estimate.
                          Still matters as the center of the candidate range.
        gamma: Shape [in_features]. input_layernorm.weight of the preceding layer.
               When provided, MSE is weighted by γ² so high-activation channels
               drive scale selection. Only valid when gamma.shape[0] == in_features.

    Returns:
        packed: [out, in//2] uint8 — two 4-bit codes per byte.
        scales: [out, in//BLOCK_SIZE] uint8 — e8m0 biased exponents.
    """
    assert tensor.shape[-1] % BLOCK_SIZE == 0, (
        f"Last dim {tensor.shape[-1]} not divisible by BLOCK_SIZE={BLOCK_SIZE}"
    )

    t = tensor.to(torch.float32)
    *leading, K = t.shape
    num_blocks = K // BLOCK_SIZE
    t_blocked = t.reshape(*leading, num_blocks, BLOCK_SIZE)  # [..., B, 32]

    # --- Anchor estimate of block_max for candidate generation ---
    if gamma is not None:
        gamma_b = gamma.reshape(num_blocks, BLOCK_SIZE)                  # [B, 32]
        gamma_b_mean = gamma_b.mean(dim=-1).clamp(min=1e-12)            # [B]
        abs_w = t_blocked.abs() * gamma_b
        if scale_percentile >= 100.0:
            block_max = abs_w.amax(dim=-1) / gamma_b_mean
        else:
            block_max = torch.quantile(abs_w, scale_percentile / 100.0, dim=-1) / gamma_b_mean
    else:
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
    sq_err = (dequant_orig - t_cand) ** 2                               # [..., B, 3, 32]

    if gamma is not None:
        # γ² weighting: [B, 32] → [1, B, 1, 32] for broadcasting
        sq_err = sq_err * (gamma_b ** 2)[None, :, None, :]

    mse = sq_err.mean(dim=-1)                                           # [..., B, 3]
    best_idx = mse.argmin(dim=-1, keepdim=True)                        # [..., B, 1]
    raw_exp = candidates.gather(-1, best_idx).squeeze(-1)               # [..., B]

    # Safety: if selected exponent still causes overflow (can happen when
    # scale_percentile < 100 and actual max exceeds the percentile significantly)
    actual_scale = torch.pow(2.0, raw_exp)
    has_overflow = (t_blocked.abs() / actual_scale.unsqueeze(-1) > 6.0).any(dim=-1)
    raw_exp = torch.where(has_overflow, raw_exp + 1, raw_exp)

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
