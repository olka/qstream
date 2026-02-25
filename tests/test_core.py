"""Tests for quant4.core — MXFP4 quantization kernel."""

import torch
import pytest

from quant4.core import BLOCK_SIZE, quantize_mxfp4, _round_to_mxfp4


class TestRoundToMxfp4:
    def test_zero(self):
        t = torch.zeros(8)
        result = _round_to_mxfp4(t)
        assert (result == 0).all()

    def test_positive_values_snap_to_grid(self):
        # Each MXFP4 value should round to itself
        grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        result = _round_to_mxfp4(grid)
        torch.testing.assert_close(result, grid)

    def test_negative_values(self):
        t = torch.tensor([-1.0, -3.0, -6.0])
        result = _round_to_mxfp4(t)
        torch.testing.assert_close(result, t)

    def test_midpoint_rounding(self):
        # 0.25 is boundary between 0 and 0.5
        t = torch.tensor([0.3, 0.7, 1.2])
        result = _round_to_mxfp4(t)
        assert result[0] == 0.5  # 0.3 > 0.25 → 0.5
        assert result[1] == 0.5  # 0.7 < 0.75 → 0.5
        assert result[2] == 1.0  # 1.2 < 1.25 → 1.0

    def test_clamps_at_6(self):
        t = torch.tensor([10.0, 100.0])
        result = _round_to_mxfp4(t)
        assert (result == 6.0).all()


class TestQuantizeMxfp4:
    def test_basic_2d(self):
        t = torch.randn(64, 128, dtype=torch.bfloat16)
        packed, scales = quantize_mxfp4(t)
        assert packed.shape == (64, 64)  # 128 / 2
        assert scales.shape == (64, 128 // BLOCK_SIZE)
        assert packed.dtype == torch.uint8
        assert scales.dtype == torch.uint8

    def test_3d_fused_experts(self):
        n_experts, out_dim, in_dim = 4, 32, 64
        t = torch.randn(n_experts, out_dim, in_dim, dtype=torch.bfloat16)
        packed, scales = quantize_mxfp4(t)
        assert packed.shape == (n_experts, out_dim, in_dim // 2)
        assert scales.shape == (n_experts, out_dim, in_dim // BLOCK_SIZE)

    def test_last_dim_not_divisible_raises(self):
        t = torch.randn(64, 33)
        with pytest.raises(AssertionError, match="not divisible"):
            quantize_mxfp4(t)

    def test_with_gamma(self):
        t = torch.randn(64, 128, dtype=torch.bfloat16)
        gamma = torch.rand(128)
        packed, scales = quantize_mxfp4(t, gamma=gamma)
        assert packed.shape == (64, 64)
        assert scales.shape == (64, 128 // BLOCK_SIZE)

    def test_zero_tensor(self):
        t = torch.zeros(32, 64, dtype=torch.bfloat16)
        packed, scales = quantize_mxfp4(t)
        # All codes should be 0 (zero value)
        assert (packed == 0).all()

    def test_scale_percentile_100(self):
        t = torch.randn(32, 64, dtype=torch.bfloat16)
        packed, scales = quantize_mxfp4(t, scale_percentile=100.0)
        assert packed.shape == (32, 32)

    def test_dequant_reconstruction_quality(self):
        """Verify quantization error is reasonable — MSE < 5% of variance."""
        torch.manual_seed(42)
        t = torch.randn(256, 256, dtype=torch.float32)
        packed, scales = quantize_mxfp4(t)

        # Reconstruct: unpack codes and apply scales
        from quant4.core import _POS_VALUES, BLOCK_SIZE
        lo = packed & 0x0F
        hi = (packed >> 4) & 0x0F
        codes = torch.stack([lo, hi], dim=-1).reshape(256, 256)
        sign = ((codes >> 3) & 1).float() * -2 + 1
        mag_idx = (codes & 0x07).long().reshape(-1)
        magnitude = _POS_VALUES[mag_idx].reshape(codes.shape)
        raw_exp = scales.float() - 127
        scale_vals = torch.pow(2.0, raw_exp)
        scale_expanded = scale_vals.repeat_interleave(BLOCK_SIZE, dim=-1)
        reconstructed = sign * magnitude * scale_expanded

        mse = ((t - reconstructed) ** 2).mean()
        variance = t.var()
        assert mse / variance < 0.05, f"MSE/var = {mse/variance:.4f}, expected < 0.05"

    def test_packed_code_range(self):
        """Each nibble should be 0-15."""
        t = torch.randn(32, 64, dtype=torch.bfloat16)
        packed, _ = quantize_mxfp4(t)
        lo = packed & 0x0F
        hi = (packed >> 4) & 0x0F
        assert (lo <= 15).all()
        assert (hi <= 15).all()

    def test_biased_exponent_range(self):
        """Scales should be valid e8m0 biased exponents (0-254)."""
        t = torch.randn(32, 64, dtype=torch.bfloat16)
        _, scales = quantize_mxfp4(t)
        assert (scales <= 254).all()
