"""Tests for qstream.handlers — tensor handler abstraction."""

import torch

from qstream.handlers import (
    StandardWeightHandler,
    FusedExpertHandler,
    get_handler,
    _passes_exclude_filter,
)


class TestPassesExcludeFilter:
    def test_no_patterns_passes(self):
        assert _passes_exclude_filter("model.layers.0.mlp.up_proj.weight", []) is True

    def test_matching_pattern_excludes(self):
        assert _passes_exclude_filter(
            "model.layers.0.self_attn.q_proj.weight",
            ["*self_attn*"],
        ) is False

    def test_non_matching_passes(self):
        assert _passes_exclude_filter(
            "model.layers.0.mlp.up_proj.weight",
            ["*self_attn*"],
        ) is True

    def test_gate_pattern(self):
        # *.mlp.gate. should match router but not gate_proj
        assert _passes_exclude_filter(
            "model.layers.0.mlp.gate.weight",
            ["*.mlp.gate.*"],
        ) is False
        assert _passes_exclude_filter(
            "model.layers.0.mlp.experts.gate_up_proj",
            ["*.mlp.gate.*"],
        ) is True


class TestStandardWeightHandler:
    handler = StandardWeightHandler()

    def test_quantizes_2d_weight(self):
        t = torch.randn(64, 128)
        assert self.handler.should_quantize("model.layers.0.mlp.up_proj.weight", t, []) is True

    def test_skips_non_weight_key(self):
        t = torch.randn(64, 128)
        assert self.handler.should_quantize("model.layers.0.mlp.up_proj.bias", t, []) is False

    def test_skips_3d(self):
        t = torch.randn(4, 64, 128)
        assert self.handler.should_quantize("model.layers.0.mlp.up_proj.weight", t, []) is False

    def test_skips_1d(self):
        t = torch.randn(128)
        assert self.handler.should_quantize("model.layers.0.input_layernorm.weight", t, []) is False

    def test_respects_exclude(self):
        t = torch.randn(64, 128)
        assert self.handler.should_quantize(
            "model.layers.0.self_attn.q_proj.weight", t, ["*self_attn*"]
        ) is False

    def test_output_keys(self):
        packed_key, scale_key = self.handler.output_keys("model.layers.0.mlp.up_proj.weight")
        assert packed_key == "model.layers.0.mlp.up_proj.weight_packed"
        assert scale_key == "model.layers.0.mlp.up_proj.weight_scale"

    def test_prepare_weight_bf16(self):
        t = torch.randn(32, 64, dtype=torch.float16)
        result = self.handler.prepare_weight(
            "k", t, "cpu", "fp16", 128, {}, None
        )
        assert result.dtype == torch.bfloat16


class TestFusedExpertHandler:
    handler = FusedExpertHandler()

    def test_quantizes_3d_gate_up_proj(self):
        t = torch.randn(4, 64, 128)
        assert self.handler.should_quantize(
            "model.layers.0.mlp.experts.gate_up_proj", t, []
        ) is True

    def test_quantizes_3d_down_proj(self):
        t = torch.randn(4, 128, 64)
        assert self.handler.should_quantize(
            "model.layers.0.mlp.experts.down_proj", t, []
        ) is True

    def test_skips_2d(self):
        t = torch.randn(64, 128)
        assert self.handler.should_quantize(
            "model.layers.0.mlp.experts.gate_up_proj", t, []
        ) is False

    def test_skips_non_expert_3d(self):
        t = torch.randn(4, 64, 128)
        assert self.handler.should_quantize(
            "model.layers.0.some_random_tensor", t, []
        ) is False

    def test_output_keys_gate_up(self):
        packed_key, scale_key = self.handler.output_keys(
            "model.layers.0.mlp.experts.gate_up_proj"
        )
        assert packed_key == "model.layers.0.mlp.experts.w13_weight"
        assert scale_key == "model.layers.0.mlp.experts.w13_weight_scale"

    def test_output_keys_down(self):
        packed_key, scale_key = self.handler.output_keys(
            "model.layers.0.mlp.experts.down_proj"
        )
        assert packed_key == "model.layers.0.mlp.experts.w2_weight"
        assert scale_key == "model.layers.0.mlp.experts.w2_weight_scale"

    def test_quantizes_step_moe_gate_proj(self):
        """Step-3.7 layout: `.moe.gate_proj.weight` with 3D shape."""
        t = torch.randn(4, 64, 128)
        assert self.handler.should_quantize(
            "model.layers.8.moe.gate_proj.weight", t, []
        ) is True

    def test_quantizes_step_moe_down_proj(self):
        t = torch.randn(4, 128, 64)
        assert self.handler.should_quantize(
            "model.layers.8.moe.down_proj.weight", t, []
        ) is True

    def test_step_moe_skips_2d_router_gate(self):
        """`.moe.gate.weight` (router) is 2D and must not match the fused pattern."""
        t = torch.randn(288, 4096)
        assert self.handler.should_quantize(
            "model.layers.8.moe.gate.weight", t, []
        ) is False

    def test_prepare_weight_fp8_3d_dequant(self):
        """3D FP8 input with 128x128 block scales should be dequanted to BF16."""
        import tempfile, os
        from safetensors.torch import save_file
        from safetensors import safe_open

        E, out_f, in_f = 2, 256, 256
        w = torch.ones(E, out_f, in_f, dtype=torch.float8_e4m3fn)
        s = torch.full((E, 2, 2), 1.5, dtype=torch.float32)
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            save_file({"k.weight": w, "k.weight_scale_inv": s}, f.name)
            path = f.name
        try:
            with safe_open(path, framework="pt") as sf:
                result = self.handler.prepare_weight(
                    "k.weight", sf.get_tensor("k.weight"), "cpu", "fp8", 128,
                    {"k.weight": "k.weight_scale_inv"}, sf,
                )
            assert result.dtype == torch.bfloat16
            assert result.shape == (E, out_f, in_f)
            assert (result.float() - 1.5).abs().max() < 0.05
        finally:
            os.unlink(path)

    def test_interleave_gate_up_roundtrip(self):
        # [E, 2*N, K] with first N=gate, last N=up
        E, N, K = 4, 8, 16
        gate = torch.arange(E * N * K).reshape(E, N, K).float()
        up = torch.arange(100, 100 + E * N * K).reshape(E, N, K).float()
        contiguous = torch.cat([gate, up], dim=1)  # [E, 2N, K]

        interleaved = FusedExpertHandler.interleave_gate_up(contiguous)
        assert interleaved.shape == contiguous.shape

        # Verify interleaved pattern: row 0=gate0, row 1=up0, row 2=gate1, ...
        assert torch.equal(interleaved[:, 0, :], gate[:, 0, :])
        assert torch.equal(interleaved[:, 1, :], up[:, 0, :])
        assert torch.equal(interleaved[:, 2, :], gate[:, 1, :])
        assert torch.equal(interleaved[:, 3, :], up[:, 1, :])

        recovered_gate = interleaved[:, ::2, :]
        recovered_up = interleaved[:, 1::2, :]
        recovered = torch.cat([recovered_gate, recovered_up], dim=1)
        assert torch.equal(recovered, contiguous)


class TestGetHandler:
    def test_returns_standard_for_2d_weight(self):
        t = torch.randn(64, 128)
        h = get_handler("model.layers.0.mlp.up_proj.weight", t)
        assert isinstance(h, StandardWeightHandler)

    def test_returns_fused_for_3d_expert(self):
        t = torch.randn(4, 64, 128)
        h = get_handler("model.layers.0.mlp.experts.gate_up_proj", t)
        assert isinstance(h, FusedExpertHandler)

    def test_returns_none_for_bias(self):
        t = torch.randn(128)
        h = get_handler("model.layers.0.mlp.up_proj.bias", t)
        assert h is None

    def test_returns_none_for_layernorm(self):
        t = torch.randn(128)
        h = get_handler("model.layers.0.input_layernorm.weight", t)
        assert h is None
