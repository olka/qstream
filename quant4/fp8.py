"""FP8 (float8_e4m3fn) dequantization with block scales."""

import torch


def dequant_fp8_block(
    weight_fp8: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """Dequantize FP8 weights with block scales to BF16.

    Args:
        weight_fp8: [out, in] float8_e4m3fn
        scale_inv:  [out//block_size, in//block_size] float32 inverse scales
        block_size: FP8 quantization block size (128 for most models)

    Returns:
        [out, in] bfloat16
    """
    out_f, in_f = weight_fp8.shape
    n_blocks_out, n_blocks_in = scale_inv.shape

    weight_f32 = weight_fp8.to(torch.float32)

    if out_f == n_blocks_out * block_size and in_f == n_blocks_in * block_size:
        # Perfect tiling — reshape + broadcast (memory efficient)
        weight_blocked = weight_f32.reshape(n_blocks_out, block_size, n_blocks_in, block_size)
        weight_blocked = weight_blocked * scale_inv.unsqueeze(1).unsqueeze(3)
        weight_f32 = weight_blocked.reshape(out_f, in_f)
    else:
        # Imperfect tiling — fallback loop
        for i in range(n_blocks_out):
            out_s, out_e = i * block_size, min((i + 1) * block_size, out_f)
            for j in range(n_blocks_in):
                in_s, in_e = j * block_size, min((j + 1) * block_size, in_f)
                weight_f32[out_s:out_e, in_s:in_e] *= scale_inv[i, j]

    return weight_f32.to(torch.bfloat16)
