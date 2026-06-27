"""Tests for qstream.core — MXFP4 quantization kernel."""

import torch
import pytest

from qstream.core import (
    BLOCK_SIZE,
    NVFP4_BLOCK_SIZE,
    _POS_VALUES,
    _round_to_mxfp4,
    dequant_nvfp4,
    quantize_mxfp4,
    quantize_nvfp4,
)


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


class TestQuantizeNvfp4:
    def test_basic_2d(self):
        t = torch.randn(64, 128, dtype=torch.bfloat16)
        packed, block_scale, global_scale = quantize_nvfp4(t)
        assert packed.shape == (64, 64)                       # 128 / 2
        assert packed.dtype == torch.uint8
        assert block_scale.shape == (64, 128 // NVFP4_BLOCK_SIZE)  # group_size 16
        assert block_scale.dtype == torch.float8_e4m3fn
        assert global_scale.shape == (1,)                     # per-tensor scalar
        assert global_scale.dtype == torch.float32

    def test_3d_fused_experts(self):
        n_experts, out_dim, in_dim = 4, 32, 64
        t = torch.randn(n_experts, out_dim, in_dim, dtype=torch.bfloat16)
        packed, block_scale, global_scale = quantize_nvfp4(t)
        assert packed.shape == (n_experts, out_dim, in_dim // 2)
        assert block_scale.shape == (n_experts, out_dim, in_dim // NVFP4_BLOCK_SIZE)
        assert global_scale.shape == (n_experts, 1)           # one global per expert

    def test_last_dim_not_divisible_raises(self):
        t = torch.randn(64, 20)  # 20 % 16 != 0
        with pytest.raises(AssertionError, match="not divisible"):
            quantize_nvfp4(t)

    def test_global_scale_recipe(self):
        """Global scale is stored as (6·448)/amax (CT reciprocal convention)."""
        torch.manual_seed(0)
        t = torch.randn(32, 64, dtype=torch.float32)
        _, _, global_scale = quantize_nvfp4(t)
        expected = (6.0 * 448.0) / t.abs().amax()
        torch.testing.assert_close(global_scale.squeeze(), expected, rtol=1e-5, atol=0)

    def test_dequant_reconstruction_quality(self):
        """16-elem blocks + E4M3 scales should reconstruct within MSE < 2% of variance."""
        torch.manual_seed(42)
        t = torch.randn(256, 256, dtype=torch.float32)
        packed, block_scale, global_scale = quantize_nvfp4(t)
        rec = dequant_nvfp4(packed, block_scale, global_scale, t.shape)
        mse = ((t - rec) ** 2).mean()
        assert mse / t.var() < 0.02, f"MSE/var = {mse / t.var():.4f}, expected < 0.02"

    def test_dequant_roundtrip_3d(self):
        torch.manual_seed(1)
        t = torch.randn(4, 32, 64, dtype=torch.float32)
        packed, block_scale, global_scale = quantize_nvfp4(t)
        rec = dequant_nvfp4(packed, block_scale, global_scale, t.shape)
        assert rec.shape == t.shape
        assert ((t - rec) ** 2).mean() / t.var() < 0.02

    def test_zero_tensor(self):
        t = torch.zeros(32, 64, dtype=torch.bfloat16)
        packed, block_scale, _ = quantize_nvfp4(t)
        assert (packed == 0).all()

    def test_packed_code_range(self):
        t = torch.randn(32, 64, dtype=torch.bfloat16)
        packed, _, _ = quantize_nvfp4(t)
        assert (packed & 0x0F <= 15).all()
        assert ((packed >> 4) & 0x0F <= 15).all()


# Optional cross-check against vLLM's NVFP4 reference dequant — the strongest
# correctness gate. Skipped when vLLM is not importable.
try:
    from vllm.model_executor.layers.quantization.utils.nvfp4_emulation_utils import (
        dequantize_to_dtype as _vllm_dequant_nvfp4,
    )
    _HAS_VLLM = True
except Exception:
    _HAS_VLLM = False


@pytest.mark.skipif(not _HAS_VLLM, reason="vLLM not importable")
def test_nvfp4_matches_vllm_reference():
    torch.manual_seed(7)
    t = torch.randn(64, 128, dtype=torch.float32)
    packed, block_scale, global_scale = quantize_nvfp4(t)
    ours = dequant_nvfp4(packed, block_scale, global_scale, t.shape)
    # vLLM applies `1/x` to the stored global on load (CT convention), so pass 1/global.
    theirs = _vllm_dequant_nvfp4(
        packed, block_scale, (1.0 / global_scale).reshape(()),
        torch.float32, block_size=NVFP4_BLOCK_SIZE, swizzle=False,
    )
    torch.testing.assert_close(ours, theirs, rtol=1e-3, atol=1e-3)
