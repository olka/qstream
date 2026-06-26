"""Tests for MXFP8 (e8m0, [1,32]-block) dequantization.

MiniMax-M3 ships weights as float8_e4m3fn with a `weight_scale_inv` that is a
uint8 E8M0 biased exponent, one per 32 input channels (block [1, 32]).
Dequant is  w.float() * 2**(scale - 127), broadcast over the 32-col block.
"""

import torch

from qstream.fp8 import dequant_fp8_block, dequant_mxfp8


class TestDequantMxfp8:
    def test_unit_scale_is_identity(self):
        """scale == 127 (bias) → factor 2**0 == 1, values preserved within FP8 precision."""
        w = torch.ones(4, 64, dtype=torch.float8_e4m3fn)
        s = torch.full((4, 2), 127, dtype=torch.uint8)
        out = dequant_mxfp8(w, s, block_size=32)
        assert out.dtype == torch.bfloat16
        assert out.shape == (4, 64)
        torch.testing.assert_close(out.float(), torch.ones(4, 64), atol=0.01, rtol=0.01)

    def test_power_of_two_factors(self):
        """scale 128 → ×2, scale 126 → ×0.5."""
        w = torch.ones(1, 64, dtype=torch.float8_e4m3fn)
        s = torch.tensor([[128, 126]], dtype=torch.uint8)
        out = dequant_mxfp8(w, s, block_size=32).float()
        assert (out[0, :32] - 2.0).abs().max() < 1e-3
        assert (out[0, 32:] - 0.5).abs().max() < 1e-3

    def test_real_m3_block(self):
        """The verified M3 example: fp8 absmax 448 with scale 114 → 0.0546875."""
        w = torch.full((1, 32), 448.0, dtype=torch.float8_e4m3fn)
        s = torch.full((1, 1), 114, dtype=torch.uint8)
        out = dequant_mxfp8(w, s, block_size=32).float()
        assert (out - 0.0546875).abs().max() < 1e-6

    def test_block_size_inferred_from_shape(self):
        """in=192, scale cols=6 → block 32; per-block scales applied independently."""
        w = torch.ones(2, 192, dtype=torch.float8_e4m3fn)
        s = torch.arange(127, 127 + 6, dtype=torch.uint8).repeat(2, 1)  # [2,6]
        out = dequant_mxfp8(w, s).float()  # default block_size=32
        for b in range(6):
            expected = 2.0 ** (127 + b - 127)
            assert (out[:, b * 32:(b + 1) * 32] - expected).abs().max() < 1e-3

    def test_3d_stacked(self):
        """Leading expert dim preserved: [E, out, in] with [E, out, in//32] scale."""
        w = torch.ones(3, 4, 32, dtype=torch.float8_e4m3fn)
        s = torch.tensor([127, 128, 129], dtype=torch.uint8).reshape(3, 1, 1).expand(3, 4, 1).contiguous()
        out = dequant_mxfp8(w, s, block_size=32).float()
        assert out.shape == (3, 4, 32)
        for e, exp in enumerate([1.0, 2.0, 4.0]):
            assert (out[e] - exp).abs().max() < 1e-3


class TestDequantFp8BlockDispatch:
    def test_uint8_scale_routes_to_mxfp8(self):
        """dequant_fp8_block must treat a uint8 scale as e8m0 (NOT a float multiplier)."""
        w = torch.ones(2, 64, dtype=torch.float8_e4m3fn)
        s = torch.full((2, 2), 128, dtype=torch.uint8)  # → ×2 if e8m0; ×128 if float
        out = dequant_fp8_block(w, s, block_size=128).float()
        assert (out - 2.0).abs().max() < 1e-3  # e8m0 interpretation

    def test_float_scale_path_unchanged(self):
        """Float block scales still use the legacy multiply path (regression guard)."""
        w = torch.ones(256, 256, dtype=torch.float8_e4m3fn)
        s = torch.ones(2, 2, dtype=torch.float32) * 2.0
        out = dequant_fp8_block(w, s, block_size=128).float()
        assert (out - 2.0).abs().max() < 0.1
