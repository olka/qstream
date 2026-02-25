"""Tests for quant4.fp8 — FP8 block dequantization."""

import torch

from quant4.fp8 import dequant_fp8_block


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
