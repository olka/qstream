"""Tests for quant4.shard — shard processing and classification."""

import json
import os
import tempfile

import torch
from safetensors.torch import save_file

from quant4.shard import (
    activation_type_from_key,
    should_quantize,
    detect_input_format,
    classify_shard,
    process_shard,
)


class TestActivationTypeFromKey:
    def test_qkv_proj(self):
        assert activation_type_from_key("model.layers.0.self_attn.q_proj.weight") == "pre_attn"
        assert activation_type_from_key("model.layers.0.self_attn.k_proj.weight") == "pre_attn"
        assert activation_type_from_key("model.layers.0.self_attn.v_proj.weight") == "pre_attn"

    def test_o_proj(self):
        assert activation_type_from_key("model.layers.0.self_attn.o_proj.weight") == "post_attn"

    def test_gate_up_proj(self):
        assert activation_type_from_key("model.layers.0.mlp.gate_proj.weight") == "pre_mlp"
        assert activation_type_from_key("model.layers.0.mlp.up_proj.weight") == "pre_mlp"

    def test_minimax_w1_w3(self):
        assert activation_type_from_key("model.layers.0.block_sparse_moe.experts.0.w1.weight") == "pre_mlp"
        assert activation_type_from_key("model.layers.0.block_sparse_moe.experts.0.w3.weight") == "pre_mlp"

    def test_down_proj(self):
        assert activation_type_from_key("model.layers.0.mlp.down_proj.weight") == "pre_down"
        assert activation_type_from_key("model.layers.0.block_sparse_moe.experts.0.w2.weight") == "pre_down"

    def test_fused_gate_up_proj(self):
        assert activation_type_from_key("model.layers.0.mlp.experts.gate_up_proj") == "pre_mlp"

    def test_router_returns_none(self):
        assert activation_type_from_key("model.layers.0.mlp.gate.weight") is None

    def test_layernorm_returns_none(self):
        assert activation_type_from_key("model.layers.0.input_layernorm.weight") is None

    def test_embed_returns_none(self):
        assert activation_type_from_key("model.embed_tokens.weight") is None


class TestShouldQuantize:
    def test_quantizes_linear_weight(self):
        assert should_quantize("model.layers.0.mlp.up_proj.weight", []) is True

    def test_skips_non_weight(self):
        assert should_quantize("model.layers.0.mlp.up_proj.bias", []) is False

    def test_skips_scale_inv(self):
        assert should_quantize("model.layers.0.mlp.up_proj.weight_scale_inv", []) is False

    def test_respects_exclude(self):
        assert should_quantize(
            "model.layers.0.self_attn.q_proj.weight",
            ["*self_attn*"],
        ) is False

    def test_embed_excluded(self):
        assert should_quantize(
            "model.embed_tokens.weight",
            ["*embed_tokens*"],
        ) is False


class TestDetectInputFormat:
    def test_fp16_model(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            save_file({"model.layers.0.weight": torch.randn(32, 64)}, f.name)
            assert detect_input_format(f.name) == "fp16"
            os.unlink(f.name)

    def test_fp8_model(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            save_file({
                "model.layers.0.weight": torch.randn(32, 64),
                "model.layers.0.weight_scale_inv": torch.ones(1, 1),
            }, f.name)
            assert detect_input_format(f.name) == "fp8"
            os.unlink(f.name)


class TestClassifyShard:
    def test_all_quantizable(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            save_file({
                "model.layers.0.mlp.up_proj.weight": torch.randn(64, 128),
                "model.layers.0.mlp.down_proj.weight": torch.randn(128, 64),
            }, f.name)
            q, p = classify_shard(f.name, ["*self_attn*"])
            assert len(q) == 2
            assert len(p) == 0
            os.unlink(f.name)

    def test_mixed_shard(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            save_file({
                "model.layers.0.mlp.up_proj.weight": torch.randn(64, 128),
                "model.layers.0.input_layernorm.weight": torch.randn(128),
            }, f.name)
            q, p = classify_shard(f.name, [])
            assert len(q) == 1
            assert len(p) == 1
            os.unlink(f.name)

    def test_all_passthrough(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            save_file({
                "model.embed_tokens.weight": torch.randn(1000, 128),
            }, f.name)
            q, p = classify_shard(f.name, ["*embed_tokens*"])
            assert len(q) == 0
            assert len(p) == 1
            os.unlink(f.name)

    def test_3d_fused_expert(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            save_file({
                "model.layers.0.mlp.experts.gate_up_proj": torch.randn(4, 64, 128),
            }, f.name)
            q, p = classify_shard(f.name, [])
            assert len(q) == 1
            assert len(p) == 0
            os.unlink(f.name)


class TestProcessShard:
    def test_quantizes_bf16_shard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.safetensors")
            output_path = os.path.join(tmpdir, "output.safetensors")

            save_file({
                "model.layers.0.mlp.up_proj.weight": torch.randn(64, 128, dtype=torch.bfloat16),
                "model.layers.0.input_layernorm.weight": torch.randn(128, dtype=torch.bfloat16),
            }, input_path)

            result = process_shard(
                input_path, output_path,
                exclude_patterns=[], input_format="fp16",
            )

            assert "model.layers.0.mlp.up_proj.weight_packed" in result
            assert "model.layers.0.mlp.up_proj.weight_scale" in result
            assert "model.layers.0.input_layernorm.weight" in result
            assert os.path.exists(output_path)

    def test_quantizes_3d_fused_experts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.safetensors")
            output_path = os.path.join(tmpdir, "output.safetensors")

            save_file({
                "model.layers.0.mlp.experts.gate_up_proj": torch.randn(4, 64, 128, dtype=torch.bfloat16),
            }, input_path)

            result = process_shard(
                input_path, output_path,
                exclude_patterns=[], input_format="fp16",
            )

            assert "model.layers.0.mlp.experts.gate_up_proj" in result
            assert "model.layers.0.mlp.experts.gate_up_proj_scale" in result

    def test_passthrough_excluded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.safetensors")
            output_path = os.path.join(tmpdir, "output.safetensors")

            save_file({
                "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 128, dtype=torch.bfloat16),
            }, input_path)

            result = process_shard(
                input_path, output_path,
                exclude_patterns=["*self_attn*"], input_format="fp16",
            )

            # Should pass through as-is, not quantized
            assert "model.layers.0.self_attn.q_proj.weight" in result
            assert "model.layers.0.self_attn.q_proj.weight_packed" not in result

    def test_quantize_only_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.safetensors")
            output_path = os.path.join(tmpdir, "output.safetensors")

            save_file({
                "model.layers.0.mlp.up_proj.weight": torch.randn(64, 128, dtype=torch.bfloat16),
                "model.layers.0.mlp.down_proj.weight": torch.randn(128, 64, dtype=torch.bfloat16),
            }, input_path)

            result = process_shard(
                input_path, output_path,
                exclude_patterns=[], input_format="fp16",
                quantize_only_keys=["model.layers.0.mlp.up_proj.weight"],
            )

            assert "model.layers.0.mlp.up_proj.weight_packed" in result
            assert "model.layers.0.mlp.down_proj.weight_packed" not in result
