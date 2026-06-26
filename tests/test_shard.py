"""Tests for qstream.shard — shard processing and classification."""

import json
import os
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from qstream.shard import (
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

    def test_shared_expert_gate_excluded(self):
        # shared_expert_gate is a [1, hidden] gating weight — must stay BF16.
        # The plain ".mlp.gate." pattern does NOT cover it, so the exclude list
        # must carry "*shared_expert*" explicitly.
        key = "model.language_model.layers.0.mlp.shared_expert_gate.weight"
        assert should_quantize(key, ["*.mlp.gate."]) is True
        assert should_quantize(key, ["*shared_expert*"]) is False

    def test_mlp_gate_still_excluded_alongside_shared_expert(self):
        # Adding "*shared_expert*" must not stop "*.mlp.gate." from matching
        # the routing gate.
        key = "model.language_model.layers.0.mlp.gate.weight"
        assert should_quantize(key, ["*.mlp.gate.", "*shared_expert*"]) is False


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

    def test_quantizes_3d_fused_experts_ct(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.safetensors")
            output_path = os.path.join(tmpdir, "output.safetensors")

            save_file({
                "model.layers.0.mlp.experts.gate_up_proj": torch.randn(4, 64, 128, dtype=torch.bfloat16),
            }, input_path)

            result = process_shard(
                input_path, output_path,
                exclude_patterns=[], input_format="fp16",
                output_format="ct",
            )

            # CT format: per-expert separate gate/up tensors
            for i in range(4):
                assert f"model.layers.0.mlp.experts.{i}.gate_proj.weight_packed" in result
                assert f"model.layers.0.mlp.experts.{i}.gate_proj.weight_scale" in result
                assert f"model.layers.0.mlp.experts.{i}.up_proj.weight_packed" in result
                assert f"model.layers.0.mlp.experts.{i}.up_proj.weight_scale" in result
            assert "model.layers.0.mlp.experts.w13_weight" not in result

    def test_quantizes_3d_down_proj_ct(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.safetensors")
            output_path = os.path.join(tmpdir, "output.safetensors")

            save_file({
                "model.layers.0.mlp.experts.down_proj": torch.randn(4, 128, 64, dtype=torch.bfloat16),
            }, input_path)

            result = process_shard(
                input_path, output_path,
                exclude_patterns=[], input_format="fp16",
                output_format="ct",
            )

            for i in range(4):
                assert f"model.layers.0.mlp.experts.{i}.down_proj.weight_packed" in result
                assert f"model.layers.0.mlp.experts.{i}.down_proj.weight_scale" in result
            assert "model.layers.0.mlp.experts.w2_weight" not in result

    def test_quantizes_3d_fused_experts_fused(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.safetensors")
            output_path = os.path.join(tmpdir, "output.safetensors")

            save_file({
                "model.layers.0.mlp.experts.gate_up_proj": torch.randn(4, 64, 128, dtype=torch.bfloat16),
            }, input_path)

            result = process_shard(
                input_path, output_path,
                exclude_patterns=[], input_format="fp16",
                output_format="fused",
            )

            # Fused format: interleaved w13/w2
            assert "model.layers.0.mlp.experts.w13_weight" in result
            assert "model.layers.0.mlp.experts.w13_weight_scale" in result

    def test_3d_fused_excluded_split_per_expert_ct(self):
        # When a 3D fused expert tensor is excluded from quantization in CT mode,
        # it must be split into per-expert .weight BF16 tensors so vLLM's
        # per-expert / MTP loaders can consume it.
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.safetensors")
            output_path = os.path.join(tmpdir, "output.safetensors")

            save_file({
                "mtp.layers.0.mlp.experts.gate_up_proj": torch.randn(4, 64, 128, dtype=torch.bfloat16),
                "mtp.layers.0.mlp.experts.down_proj": torch.randn(4, 128, 32, dtype=torch.bfloat16),
            }, input_path)

            result = process_shard(
                input_path, output_path,
                exclude_patterns=["*mtp*"], input_format="fp16",
                output_format="ct",
            )

            # Per-expert BF16 .weight tensors — not the 3D fused passthrough.
            for i in range(4):
                assert f"mtp.layers.0.mlp.experts.{i}.gate_proj.weight" in result
                assert f"mtp.layers.0.mlp.experts.{i}.up_proj.weight" in result
                assert f"mtp.layers.0.mlp.experts.{i}.down_proj.weight" in result
                # Must NOT be quantized.
                assert f"mtp.layers.0.mlp.experts.{i}.gate_proj.weight_packed" not in result
            # Original 3D keys must be gone.
            assert "mtp.layers.0.mlp.experts.gate_up_proj" not in result
            assert "mtp.layers.0.mlp.experts.down_proj" not in result

    def test_quantizes_step_moe_ct(self):
        """Step-3.7 layout: `.moe.gate_proj.weight` 3D FP8 → CT per-expert under `.moe.experts.{E}.`."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.safetensors")
            output_path = os.path.join(tmpdir, "output.safetensors")

            E, out_f, in_f = 4, 256, 384
            w = torch.ones(E, out_f, in_f, dtype=torch.float8_e4m3fn)
            s = torch.full((E, 2, 3), 1.0, dtype=torch.float32)
            save_file({
                "model.layers.8.moe.gate_proj.weight": w,
                "model.layers.8.moe.gate_proj.weight_scale_inv": s,
                "model.layers.8.moe.down_proj.weight":
                    torch.ones(E, in_f, out_f, dtype=torch.float8_e4m3fn),
                "model.layers.8.moe.down_proj.weight_scale_inv":
                    torch.full((E, 3, 2), 1.0, dtype=torch.float32),
            }, input_path)

            result = process_shard(
                input_path, output_path,
                exclude_patterns=[], input_format="fp8",
                output_format="ct",
            )

            # CT per-expert keys with the inserted `.experts.` segment — matches
            # what FusedMoE.make_expert_params_mapping expects.
            for i in range(E):
                assert f"model.layers.8.moe.experts.{i}.gate_proj.weight_packed" in result
                assert f"model.layers.8.moe.experts.{i}.gate_proj.weight_scale" in result
                assert f"model.layers.8.moe.experts.{i}.down_proj.weight_packed" in result
                assert f"model.layers.8.moe.experts.{i}.down_proj.weight_scale" in result
            assert "model.layers.8.moe.gate_proj.weight" not in result
            assert "model.layers.8.moe.gate_proj.weight_scale_inv" not in result

    def test_quantizes_qwen_experts_ct_unchanged(self):
        """Qwen3.5-style `.experts.gate_up_proj` input must still emit `.experts.{E}.PROJ.weight_packed`."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.safetensors")
            output_path = os.path.join(tmpdir, "output.safetensors")

            save_file({
                "model.layers.0.mlp.experts.gate_up_proj":
                    torch.randn(4, 64, 128, dtype=torch.bfloat16),
            }, input_path)

            result = process_shard(
                input_path, output_path,
                exclude_patterns=[], input_format="fp16",
                output_format="ct",
            )

            for i in range(4):
                assert f"model.layers.0.mlp.experts.{i}.gate_proj.weight_packed" in result
                assert f"model.layers.0.mlp.experts.{i}.up_proj.weight_packed" in result
            # Must not double-insert `.experts.experts.`
            assert "model.layers.0.mlp.experts.experts.0.gate_proj.weight_packed" not in result

    def test_classify_shard_step_moe(self):
        """Step-3.7 `.moe.PROJ.weight` 3D keys must be picked up by classify_shard."""
        from qstream.shard import classify_shard
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            save_file({
                "model.layers.8.moe.gate_proj.weight": torch.randn(4, 64, 128).to(torch.float8_e4m3fn),
                "model.layers.8.moe.gate_proj.weight_scale_inv": torch.ones(4, 1, 1, dtype=torch.float32),
                "model.layers.8.moe.gate.weight": torch.randn(4, 64, dtype=torch.bfloat16),  # router 2D
            }, f.name)
            q, p = classify_shard(f.name, ["*moe.gate.weight"])
            assert "model.layers.8.moe.gate_proj.weight" in q
            assert "model.layers.8.moe.gate.weight" in p
            os.unlink(f.name)

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

            assert "model.layers.0.self_attn.q_proj.weight" in result
            assert "model.layers.0.self_attn.q_proj.weight_packed" not in result

    def test_fp8_passthrough_preserves_format(self):
        """FP8 tensors excluded from quantization keep weight + renamed scale companion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.safetensors")
            output_path = os.path.join(tmpdir, "output.safetensors")

            w_fp8 = torch.randn(64, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            s_block = torch.ones(1, 1, dtype=torch.float32)

            save_file({
                "model.layers.0.self_attn.q_proj.weight": w_fp8,
                "model.layers.0.self_attn.q_proj.weight_scale_inv": s_block,
            }, input_path)

            result = process_shard(
                input_path, output_path,
                exclude_patterns=["*self_attn*"], input_format="fp8",
            )

            assert "model.layers.0.self_attn.q_proj.weight" in result
            assert "model.layers.0.self_attn.q_proj.weight_scale" in result
            assert "model.layers.0.self_attn.q_proj.weight_scale_inv" not in result
            assert "model.layers.0.self_attn.q_proj.weight_packed" not in result

            with safe_open(output_path, framework="pt") as f:
                assert f.get_tensor("model.layers.0.self_attn.q_proj.weight").dtype == torch.float8_e4m3fn

    def test_fp8_passthrough_per_tensor_scale(self):
        """Per-tensor scalar scale (Mistral-style) survives passthrough as weight_scale."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.safetensors")
            output_path = os.path.join(tmpdir, "output.safetensors")

            w_fp8 = torch.randn(64, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            s_scalar = torch.tensor(0.000125, dtype=torch.bfloat16)

            save_file({
                "model.layers.0.self_attn.q_proj.weight": w_fp8,
                "model.layers.0.self_attn.q_proj.weight_scale_inv": s_scalar,
            }, input_path)

            result = process_shard(
                input_path, output_path,
                exclude_patterns=["*self_attn*"], input_format="fp8",
            )

            with safe_open(output_path, framework="pt") as f:
                assert f.get_tensor("model.layers.0.self_attn.q_proj.weight").dtype == torch.float8_e4m3fn
                preserved_scale = f.get_tensor("model.layers.0.self_attn.q_proj.weight_scale")
                assert preserved_scale.shape == (1,)
                assert preserved_scale.dtype == torch.float32

    def test_fp8_passthrough_renames_activation_scale(self):
        """activation_scale companion is renamed to input_scale on passthrough."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.safetensors")
            output_path = os.path.join(tmpdir, "output.safetensors")

            w_fp8 = torch.randn(64, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            ws = torch.tensor(0.000125, dtype=torch.bfloat16)
            ascale = torch.tensor(0.5, dtype=torch.bfloat16)

            save_file({
                "model.layers.0.self_attn.q_proj.weight": w_fp8,
                "model.layers.0.self_attn.q_proj.weight_scale_inv": ws,
                "model.layers.0.self_attn.q_proj.activation_scale": ascale,
            }, input_path)

            result = process_shard(
                input_path, output_path,
                exclude_patterns=["*self_attn*"], input_format="fp8",
            )

            assert "model.layers.0.self_attn.q_proj.input_scale" in result
            assert "model.layers.0.self_attn.q_proj.activation_scale" not in result
            with safe_open(output_path, framework="pt") as f:
                preserved_act = f.get_tensor("model.layers.0.self_attn.q_proj.input_scale")
                assert preserved_act.shape == (1,)
                assert preserved_act.dtype == torch.float32
                assert (preserved_act - 0.5).abs().item() < 1e-3

    def test_expert_quantize_set_selective(self):
        """Selective per-expert: quantize one expert, keep another in FP8."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.safetensors")
            output_path = os.path.join(tmpdir, "output.safetensors")

            save_file({
                # Expert 0: should be quantized to MXFP4
                "model.layers.0.block_sparse_moe.experts.0.w1.weight":
                    torch.randn(64, 128, dtype=torch.bfloat16),
                # Expert 1: should stay FP8 (not in quantize set)
                "model.layers.0.block_sparse_moe.experts.1.w1.weight":
                    torch.randn(64, 128, dtype=torch.bfloat16),
            }, input_path)

            # Only quantize expert 0 in layer 0
            result = process_shard(
                input_path, output_path,
                exclude_patterns=[], input_format="fp16",
                expert_quantize_set={(0, 0)},
            )

            # Expert 0: MXFP4 quantized
            assert "model.layers.0.block_sparse_moe.experts.0.w1.weight_packed" in result
            assert "model.layers.0.block_sparse_moe.experts.0.w1.weight_scale" in result

            # Expert 1: kept as-is (BF16 passthrough since input is fp16)
            assert "model.layers.0.block_sparse_moe.experts.1.w1.weight" in result
            assert "model.layers.0.block_sparse_moe.experts.1.w1.weight_packed" not in result

    def test_expert_quantize_set_fp8_passthrough(self):
        """FP8 experts not in quantize set keep weight + scale_inv."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.safetensors")
            output_path = os.path.join(tmpdir, "output.safetensors")

            # Simulate FP8 expert weights
            w = torch.randn(64, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            s = torch.ones(1, 1, dtype=torch.float32)  # block scale

            save_file({
                "model.layers.0.block_sparse_moe.experts.0.w1.weight": w,
                "model.layers.0.block_sparse_moe.experts.0.w1.weight_scale_inv": s,
            }, input_path)

            # Empty set = no experts to quantize
            result = process_shard(
                input_path, output_path,
                exclude_patterns=[], input_format="fp8",
                expert_quantize_set=set(),
            )

            assert "model.layers.0.block_sparse_moe.experts.0.w1.weight" in result
            assert "model.layers.0.block_sparse_moe.experts.0.w1.weight_scale" in result
            assert "model.layers.0.block_sparse_moe.experts.0.w1.weight_scale_inv" not in result
            assert "model.layers.0.block_sparse_moe.experts.0.w1.weight_packed" not in result

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
