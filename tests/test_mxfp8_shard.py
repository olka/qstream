"""Shard-level tests for the mixed MXFP4 (experts) + MXFP8 (rest) path.

MiniMax-M3 recipe: routed experts -> MXFP4; every other FP8 layer kept as
native MXFP8 (uint8 e8m0 scale preserved, key renamed); BF16/F32 copied.
"""

import os
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from qstream.shard import detect_input_format, process_shard


def _mxfp8(out, in_, block=32):
    """A fake MXFP8 weight + e8m0 scale_inv pair."""
    w = torch.randn(out, in_, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    s = torch.full((out, in_ // block), 127, dtype=torch.uint8)
    return w, s


class TestDetectMxfp8:
    def test_uint8_scale_inv_is_mxfp8(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            w, s = _mxfp8(64, 128)
            save_file({
                "model.layers.0.self_attn.q_proj.weight": w,
                "model.layers.0.self_attn.q_proj.weight_scale_inv": s,
            }, f.name)
            assert detect_input_format(f.name) == "mxfp8"
            os.unlink(f.name)

    def test_float_scale_inv_still_fp8(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            w = torch.randn(64, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
            save_file({
                "model.layers.0.self_attn.q_proj.weight": w,
                "model.layers.0.self_attn.q_proj.weight_scale_inv": torch.ones(1, 1),
            }, f.name)
            assert detect_input_format(f.name) == "fp8"
            os.unlink(f.name)


class TestMixedMxfp4Mxfp8Shard:
    def _run(self, tmpdir):
        input_path = os.path.join(tmpdir, "in.safetensors")
        output_path = os.path.join(tmpdir, "out.safetensors")

        ew, es = _mxfp8(64, 128)        # routed expert -> MXFP4
        sw, ss = _mxfp8(64, 128)        # shared expert -> MXFP8 passthrough
        aw, as_ = _mxfp8(64, 128)       # attention -> MXFP8 passthrough
        tensors = {
            "language_model.model.layers.7.block_sparse_moe.experts.0.w1.weight": ew,
            "language_model.model.layers.7.block_sparse_moe.experts.0.w1.weight_scale_inv": es,
            "language_model.model.layers.7.block_sparse_moe.shared_experts.gate_proj.weight": sw,
            "language_model.model.layers.7.block_sparse_moe.shared_experts.gate_proj.weight_scale_inv": ss,
            "language_model.model.layers.7.self_attn.q_proj.weight": aw,
            "language_model.model.layers.7.self_attn.q_proj.weight_scale_inv": as_,
            "language_model.lm_head.weight": torch.randn(100, 64, dtype=torch.bfloat16),
        }
        save_file(tensors, input_path)

        result = process_shard(
            input_path, output_path,
            exclude_patterns=[], input_format="mxfp8",
            include_patterns=["block_sparse_moe.experts"],
        )
        return result, output_path

    def test_only_routed_experts_become_mxfp4(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, _ = self._run(tmp)
            exp = "language_model.model.layers.7.block_sparse_moe.experts.0.w1"
            assert f"{exp}.weight_packed" in result
            assert f"{exp}.weight_scale" in result
            assert f"{exp}.weight" not in result

    def test_shared_experts_kept_mxfp8(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, out_path = self._run(tmp)
            sh = "language_model.model.layers.7.block_sparse_moe.shared_experts.gate_proj"
            assert f"{sh}.weight" in result          # fp8 weight kept
            assert f"{sh}.weight_scale" in result    # renamed from _scale_inv
            assert f"{sh}.weight_packed" not in result
            assert f"{sh}.weight_scale_inv" not in result
            with safe_open(out_path, framework="pt") as f:
                assert f.get_tensor(f"{sh}.weight").dtype == torch.float8_e4m3fn
                # e8m0 scale MUST stay uint8 (float-casting breaks vLLM _is_mxfp8)
                assert f.get_tensor(f"{sh}.weight_scale").dtype == torch.uint8

    def test_attention_kept_mxfp8(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, out_path = self._run(tmp)
            q = "language_model.model.layers.7.self_attn.q_proj"
            assert f"{q}.weight" in result
            assert f"{q}.weight_scale" in result
            assert f"{q}.weight_packed" not in result
            with safe_open(out_path, framework="pt") as f:
                assert f.get_tensor(f"{q}.weight_scale").dtype == torch.uint8

    def test_bf16_copied_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, out_path = self._run(tmp)
            assert "language_model.lm_head.weight" in result
            with safe_open(out_path, framework="pt") as f:
                assert f.get_tensor("language_model.lm_head.weight").dtype == torch.bfloat16
