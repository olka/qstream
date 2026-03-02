"""Tests for qstream.gamma — layernorm weight loading."""

from qstream.gamma import extract_layer_index


class TestExtractLayerIndex:
    def test_standard_key(self):
        assert extract_layer_index("model.layers.42.mlp.up_proj.weight") == 42

    def test_layer_zero(self):
        assert extract_layer_index("model.layers.0.self_attn.q_proj.weight") == 0

    def test_no_layer_returns_none(self):
        assert extract_layer_index("model.embed_tokens.weight") is None
        assert extract_layer_index("lm_head.weight") is None

    def test_expert_key(self):
        assert extract_layer_index(
            "model.layers.5.block_sparse_moe.experts.3.w1.weight"
        ) == 5

    def test_fused_expert_key(self):
        assert extract_layer_index(
            "model.layers.10.mlp.experts.gate_up_proj"
        ) == 10
