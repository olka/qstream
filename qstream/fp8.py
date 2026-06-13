"""FP8 (float8_e4m3fn) dequantization with block scales."""

import torch


def dequant_fp8_block(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """Dequantize FP8 weights to BF16.

    Auto-dispatches on `scale_inv` rank:
      - 0-d scalar → per-tensor static FP8 (Mistral-style)
      - matching rank → block-scaled FP8 (DeepSeek/Qwen-style, `block_size` × `block_size`
        on the trailing two dims; any number of leading dims, e.g. expert dim, are preserved)

    Args:
        weight_fp8: [..., out, in] float8_e4m3fn
        scale_inv:  [] (per-tensor) or [..., out//block_size, in//block_size] inverse scales
        block_size: FP8 quantization block size (128 for most models, ignored for per-tensor)

    Returns:
        [..., out, in] bfloat16
    """
    if scale_inv.ndim == 0:
        return (weight_fp8.to(torch.float32) * scale_inv.to(torch.float32)).to(torch.bfloat16)

    lead = weight_fp8.shape[:-2]
    out_f, in_f = weight_fp8.shape[-2:]
    n_blocks_out, n_blocks_in = scale_inv.shape[-2:]
    assert scale_inv.shape[:-2] == lead, (
        f"scale_inv leading dims {scale_inv.shape[:-2]} must match weight {lead}"
    )

    weight_f32 = weight_fp8.to(torch.float32)

    if out_f == n_blocks_out * block_size and in_f == n_blocks_in * block_size:
        # Perfect tiling — reshape + broadcast (memory efficient)
        weight_blocked = weight_f32.reshape(*lead, n_blocks_out, block_size, n_blocks_in, block_size)
        weight_blocked = weight_blocked * scale_inv.unsqueeze(-1).unsqueeze(-3)
        weight_f32 = weight_blocked.reshape(*lead, out_f, in_f)
    else:
        # Imperfect tiling — fallback loop (trailing two dims only)
        for i in range(n_blocks_out):
            out_s, out_e = i * block_size, min((i + 1) * block_size, out_f)
            for j in range(n_blocks_in):
                in_s, in_e = j * block_size, min((j + 1) * block_size, in_f)
                weight_f32[..., out_s:out_e, in_s:in_e] *= scale_inv[..., i, j].reshape(
                    *lead, 1, 1
                )

    return weight_f32.to(torch.bfloat16)
