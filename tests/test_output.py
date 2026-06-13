"""Tests for qstream.output — index generation and mixed-precision config."""

import os
import re
import tempfile

import torch
from safetensors.torch import save_file

from qstream.output import build_index_from_shards, build_quantization_config


def _re_match(targets, name):
    """True if any 're:' target matches name (mirrors vLLM _is_equal_or_regex_match)."""
    for t in targets:
        if t.startswith("re:") and re.match(t[3:], name):
            return True
        if t == name:
            return True
    return False


class TestBuildIndexFromShards:
    def test_maps_tensors_to_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_file({"a.weight": torch.zeros(2, 2), "b.weight": torch.zeros(2, 2)},
                      os.path.join(tmp, "model-00001-of-00002.safetensors"))
            save_file({"c.weight": torch.zeros(2, 2)},
                      os.path.join(tmp, "model-00002-of-00002.safetensors"))
            idx = build_index_from_shards(
                tmp, ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])
            wm = idx["weight_map"]
            assert wm["a.weight"] == "model-00001-of-00002.safetensors"
            assert wm["b.weight"] == "model-00001-of-00002.safetensors"
            assert wm["c.weight"] == "model-00002-of-00002.safetensors"
            assert "metadata" in idx and idx["metadata"]["total_size"] > 0


class TestBuildQuantizationConfig:
    def _cfg(self):
        L = "language_model.model.layers"
        mxfp4 = [
            f"{L}.7.block_sparse_moe.experts.0.w1",
            f"{L}.7.block_sparse_moe.experts.0.w2",
            f"{L}.7.block_sparse_moe.experts.1.w1",
        ]
        mxfp8 = [
            f"{L}.7.self_attn.q_proj",
            f"{L}.7.self_attn.o_proj",
            f"{L}.7.block_sparse_moe.shared_experts.gate_proj",
            f"{L}.0.mlp.gate_proj",
        ]
        ignore = [
            f"{L}.7.block_sparse_moe.gate",
            "language_model.lm_head",
            "language_model.model.embed_tokens",
            "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj",
        ]
        return build_quantization_config(mxfp4, mxfp8, ignore)

    def test_top_level_shape(self):
        c = self._cfg()
        assert c["quant_method"] == "compressed-tensors"
        assert c["format"] == "mixed-precision"
        assert set(c["config_groups"]) == {"group_0", "group_1"}
        assert "ignore" in c

    def test_mxfp4_group_schema(self):
        g = self._cfg()["config_groups"]["group_0"]
        assert g["format"] == "mxfp4-pack-quantized"
        w = g["weights"]
        assert w["num_bits"] == 4
        assert w["type"] == "float"
        assert w["strategy"] == "group"
        assert w["group_size"] == 32
        assert w["symmetric"] is True

    def test_mxfp8_group_schema(self):
        g = self._cfg()["config_groups"]["group_1"]
        assert g["format"] == "mxfp8-quantized"
        w = g["weights"]
        assert w["num_bits"] == 8
        assert w["type"] == "float"
        assert w["strategy"] == "group"
        assert w["group_size"] == 32
        assert w["symmetric"] is True
        # MUST be present and round-trip to torch.uint8 for vLLM _is_mxfp8
        assert w["scale_dtype"] == "torch.uint8"

    def test_expert_target_matches_vllm_probe(self):
        """vLLM probes `{moe}.0.gate_proj` — the MXFP4 target must match that."""
        g0 = self._cfg()["config_groups"]["group_0"]["targets"]
        probe = "language_model.model.layers.7.block_sparse_moe.experts.0.gate_proj"
        assert _re_match(g0, probe)
        # ...and the down/up probes
        assert _re_match(g0, probe.replace("gate_proj", "down_proj"))

    def test_groups_are_disjoint(self):
        """A shared-expert / attention module must NOT match the MXFP4 group, and an
        expert probe must NOT match the MXFP8 group."""
        c = self._cfg()
        g0 = c["config_groups"]["group_0"]["targets"]
        g1 = c["config_groups"]["group_1"]["targets"]
        shared = "language_model.model.layers.7.block_sparse_moe.shared_experts.gate_proj"
        attn = "language_model.model.layers.7.self_attn.q_proj"
        expert_probe = "language_model.model.layers.7.block_sparse_moe.experts.0.gate_proj"
        assert not _re_match(g0, shared)
        assert not _re_match(g0, attn)
        assert _re_match(g1, shared)
        assert _re_match(g1, attn)
        assert not _re_match(g1, expert_probe)

    def test_mxfp8_target_generalizes_across_layers(self):
        """A target built from layer 7 must also match the same module in layer 42."""
        g1 = self._cfg()["config_groups"]["group_1"]["targets"]
        assert _re_match(g1, "language_model.model.layers.42.self_attn.q_proj")

    def test_targets_match_vllm_merged_modules(self):
        """vLLM fuses q/k/v->qkv_proj and gate/up->gate_up_proj and resolves the
        scheme against the MERGED name; the config must target those or the
        merged linears load unquantized (the M3 zero-weights bug)."""
        c = self._cfg()
        g1 = c["config_groups"]["group_1"]["targets"]
        L = "language_model.model.layers.7"
        assert _re_match(g1, f"{L}.self_attn.qkv_proj"), "qkv_proj must match"
        assert _re_match(g1, f"{L}.mlp.gate_up_proj"), "gate_up_proj must match"
        # and the separate names still match (non-merged paths / loaders vary)
        assert _re_match(g1, f"{L}.self_attn.q_proj")
        assert _re_match(g1, f"{L}.mlp.gate_proj")

    def test_merged_targets_scoped_not_global(self):
        """Merged targets are prefix-scoped (won't hit an unquantized vision tower)."""
        c = self._cfg()
        g1 = c["config_groups"]["group_1"]["targets"]
        # vision tower has its own self_attn.qkv_proj that must NOT be quantized
        assert not _re_match(g1, "vision_tower.vision_model.encoder.layers.0.self_attn.qkv_proj")

    def test_ignore_contains_router_and_heads(self):
        c = self._cfg()
        ig = c["ignore"]
        assert _re_match(ig, "language_model.model.layers.7.block_sparse_moe.gate")
        assert _re_match(ig, "language_model.model.layers.42.block_sparse_moe.gate")
        assert _re_match(ig, "language_model.lm_head")
        # ignored layers must not leak into a quantized group
        g0 = c["config_groups"]["group_0"]["targets"]
        g1 = c["config_groups"]["group_1"]["targets"]
        assert not _re_match(g0, "language_model.lm_head")
        assert not _re_match(g1, "language_model.lm_head")
