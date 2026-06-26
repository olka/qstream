"""Tests for qstream.fp8 — FP8 block dequantization."""

import torch

from qstream.fp8 import dequant_fp8_block


class TestDequantFp8Block:
    def test_identity_scale(self):
        """Scale of 1.0 everywhere should preserve values (within FP8 precision)."""
        weight = torch.ones(128, 128, dtype=torch.float8_e4m3fn)
        scale_inv = torch.ones(1, 1, dtype=torch.float32)
        result = dequant_fp8_block(weight, scale_inv, block_size=128)
        assert result.dtype == torch.bfloat16
        assert result.shape == (128, 128)
        torch.testing.assert_close(result.float(), torch.ones(128, 128), atol=0.01, rtol=0.01)

    def test_perfect_tiling(self):
        """Dimensions perfectly divisible by block_size."""
        weight = torch.ones(256, 256, dtype=torch.float8_e4m3fn)
        scale_inv = torch.ones(2, 2, dtype=torch.float32) * 2.0
        result = dequant_fp8_block(weight, scale_inv, block_size=128)
        assert result.shape == (256, 256)
        # All values should be ~2.0 (1.0 * 2.0)
        assert (result.float() - 2.0).abs().max() < 0.1

    def test_imperfect_tiling(self):
        """Dimensions not perfectly divisible by block_size — fallback path."""
        weight = torch.ones(100, 100, dtype=torch.float8_e4m3fn)
        scale_inv = torch.ones(1, 1, dtype=torch.float32) * 3.0
        result = dequant_fp8_block(weight, scale_inv, block_size=128)
        assert result.shape == (100, 100)
        assert result.dtype == torch.bfloat16

    def test_output_dtype(self):
        weight = torch.zeros(128, 128, dtype=torch.float8_e4m3fn)
        scale_inv = torch.ones(1, 1, dtype=torch.float32)
        result = dequant_fp8_block(weight, scale_inv)
        assert result.dtype == torch.bfloat16

    def test_scale_varies_per_block(self):
        """Different scales for different blocks."""
        weight = torch.ones(256, 128, dtype=torch.float8_e4m3fn)
        scale_inv = torch.tensor([[1.0], [2.0]], dtype=torch.float32)
        result = dequant_fp8_block(weight, scale_inv, block_size=128)
        # First 128 rows scaled by 1.0, next 128 by 2.0
        assert (result[:128].float() - 1.0).abs().max() < 0.1
        assert (result[128:].float() - 2.0).abs().max() < 0.1

    def test_per_tensor_scalar_scale(self):
        """0-d scalar scale_inv (Mistral-style per-tensor static FP8)."""
        weight = torch.ones(1024, 12288, dtype=torch.float8_e4m3fn)
        scale_inv = torch.tensor(2.5, dtype=torch.bfloat16)
        result = dequant_fp8_block(weight, scale_inv, block_size=128)
        assert result.dtype == torch.bfloat16
        assert result.shape == (1024, 12288)
        assert (result.float() - 2.5).abs().max() < 0.05

    def test_per_tensor_irregular_shape(self):
        """Per-tensor scaling works for any shape, including non-divisible-by-128."""
        weight = torch.ones(1024, 12288, dtype=torch.float8_e4m3fn)
        scale_inv = torch.tensor(0.000125, dtype=torch.bfloat16)
        result = dequant_fp8_block(weight, scale_inv)
        # bfloat16 representation of 0.000125 ~ 0.0001249
        assert result.shape == (1024, 12288)
        assert (result.float() - float(scale_inv)).abs().max() < 1e-5

    def test_stacked_experts_3d(self):
        """Stacked MoE tensor [E, out, in] with [E, out//B, in//B] scale (Step-3.7 layout)."""
        E, out_f, in_f = 4, 256, 384
        weight = torch.ones(E, out_f, in_f, dtype=torch.float8_e4m3fn)
        scale_inv = torch.arange(1, E + 1, dtype=torch.float32).reshape(E, 1, 1).expand(E, 2, 3).contiguous()
        result = dequant_fp8_block(weight, scale_inv, block_size=128)
        assert result.shape == (E, out_f, in_f)
        assert result.dtype == torch.bfloat16
        for e in range(E):
            assert (result[e].float() - float(e + 1)).abs().max() < 0.05

    def test_stacked_experts_3d_per_block(self):
        """Stacked MoE tensor with truly per-block-varying scales (no broadcast cheating)."""
        E, out_f, in_f, B = 2, 256, 256, 128
        weight = torch.ones(E, out_f, in_f, dtype=torch.float8_e4m3fn)
        scale_inv = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0]],
             [[5.0, 6.0], [7.0, 8.0]]], dtype=torch.float32,
        )
        result = dequant_fp8_block(weight, scale_inv, block_size=B).float()
        assert result.shape == (E, out_f, in_f)
        for e in range(E):
            for i in range(2):
                for j in range(2):
                    block = result[e, i * B:(i + 1) * B, j * B:(j + 1) * B]
                    assert (block - float(scale_inv[e, i, j])).abs().max() < 0.05

    def test_stacked_experts_shape_mismatch_raises(self):
        """Scale leading dims must match weight leading dims."""
        weight = torch.ones(4, 256, 256, dtype=torch.float8_e4m3fn)
        scale_inv = torch.ones(3, 2, 2, dtype=torch.float32)  # wrong E
        try:
            dequant_fp8_block(weight, scale_inv, block_size=128)
        except AssertionError:
            return
        raise AssertionError("expected AssertionError on mismatched leading dims")
