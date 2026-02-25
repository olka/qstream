"""Tests for quant4.handlers — tensor handler abstraction."""

import torch

from quant4.handlers import (
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
            ["*.mlp.gate."],
        ) is False
        assert _passes_exclude_filter(
            "model.layers.0.mlp.experts.gate_up_proj",
            ["*.mlp.gate."],
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

    def test_output_keys(self):
        packed_key, scale_key = self.handler.output_keys(
            "model.layers.0.mlp.experts.gate_up_proj"
        )
        assert packed_key == "model.layers.0.mlp.experts.gate_up_proj_packed"
        assert scale_key == "model.layers.0.mlp.experts.gate_up_proj_scale"


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
