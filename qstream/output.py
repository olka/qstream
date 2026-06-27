"""Output-side helpers: safetensors index generation and quantization config.

These are pure functions (no model loading) so the quantize CLI can assemble a
valid output directory — an index when the source lacks one, and a mixed-precision
``quantization_config`` describing per-group MXFP4 / MXFP8 schemes.
"""

import json
import os
import re
import struct


def build_index_from_shards(model_dir: str, shard_files: list[str]) -> dict:
    """Build a `model.safetensors.index.json` payload by reading shard headers.

    Some checkpoints (e.g. MiniMax-M3) ship without an index. We reconstruct the
    `weight_map` (tensor → shard filename) and `metadata.total_size` (sum of tensor
    bytes) by parsing only the safetensors header of each shard — no tensor data.
    """
    weight_map: dict[str, str] = {}
    total_size = 0
    for shard in shard_files:
        path = os.path.join(model_dir, shard)
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            weight_map[key] = shard
            start, end = meta["data_offsets"]
            total_size += end - start
    return {"metadata": {"total_size": total_size}, "weight_map": weight_map}


# --- quantization config -------------------------------------------------------

_EXPERT_SEGMENT = re.compile(r"\.experts\.\d+\.")
# vLLM probes `{moe}.0.gate_proj/.up_proj/.down_proj`; this matches all of them
# while staying disjoint from `.shared_experts.` (preceded by `_`, not `.`).
_EXPERT_TARGET = r"re:.*\.experts\.\d+\..*"


def _module_to_regex(module: str) -> str:
    """Turn a concrete module path into a layer-index-agnostic `re:` target.

    `...layers.7.self_attn.q_proj` → `re:.*...layers\\.\\d+\\.self_attn\\.q_proj$`,
    so one target covers the module across every layer. vLLM matches with
    `re.match` (start-anchored), hence the leading `.*` and trailing `$`.
    """
    parts = (r"\d+" if p.isdigit() else re.escape(p) for p in module.split("."))
    return "re:.*" + r"\.".join(parts) + "$"


def _dedup(seq) -> list[str]:
    seen: dict[str, None] = {}
    for x in seq:
        seen.setdefault(x, None)
    return list(seen)


def _mxfp4_targets(modules: list[str]) -> list[str]:
    targets: list[str] = []
    for m in modules:
        targets.append(_EXPERT_TARGET if _EXPERT_SEGMENT.search(m) else _module_to_regex(m))
    return _dedup(targets)


def _targets(modules: list[str]) -> list[str]:
    return _dedup(_module_to_regex(m) for m in modules)


# vLLM fuses these checkpoint projections into a single merged module at load
# time and resolves the quant scheme against the *merged* name. The config must
# therefore also target the merged names, or those linears silently fall back to
# unquantized (weights never load -> zero output). See the M3 debugging saga.
_MERGE_MAP = {
    "q_proj": "qkv_proj", "k_proj": "qkv_proj", "v_proj": "qkv_proj",
    "index_q_proj": "qkv_proj", "index_k_proj": "qkv_proj",
    "gate_proj": "gate_up_proj", "up_proj": "gate_up_proj",
}


def _merged_targets(modules: list[str]) -> list[str]:
    """Derive vLLM merged-module targets (qkv_proj/gate_up_proj) from the
    separate checkpoint projections, scoped to each module's own prefix (so
    they never reach an unquantized vision tower)."""
    out: list[str] = []
    for m in modules:
        prefix, _, leaf = m.rpartition(".")
        merged = _MERGE_MAP.get(leaf)
        if merged and prefix:
            out.append(_module_to_regex(f"{prefix}.{merged}"))
    return _dedup(out)


def _fp8_group(modules: list[str], fp8_kind: str, fp8_block_size: int) -> dict:
    """Config group for FP8 layers kept at native 8-bit precision.

    `fp8_kind="mxfp8"` → OCP MXFP8 (group-32, uint8 e8m0 scales; M3-style). The
    `scale_dtype` is what makes vLLM's `_is_mxfp8` pick the MXFP8 scheme.
    `fp8_kind="block"` → DeepSeek/Step-style block FP8 (128×128 float scales).
    """
    targets = _dedup(_targets(modules) + _merged_targets(modules))
    if fp8_kind == "mxfp8":
        return {
            "targets": targets,
            "format": "mxfp8-quantized",
            "weights": {
                "num_bits": 8, "type": "float", "strategy": "group",
                "group_size": 32, "symmetric": True, "scale_dtype": "torch.uint8",
            },
        }
    return {  # block FP8
        "targets": targets,
        "format": "float-quantized",
        "weights": {
            "num_bits": 8, "type": "float", "strategy": "block",
            "block_structure": [fp8_block_size, fp8_block_size],
            "symmetric": True, "dynamic": False,
        },
    }


def build_quantization_config(
    mxfp4_modules: list[str],
    fp8_modules: list[str],
    ignore_modules: list[str],
    *,
    fp4_kind: str = "mxfp4",
    fp8_kind: str = "mxfp8",
    fp8_block_size: int = 128,
) -> dict:
    """Assemble a compressed-tensors mixed-precision quantization config.

    Two config groups are emitted:
      group_0 — packed FP4 (num_bits 4) for the routed experts. `fp4_kind` selects MXFP4
                (group_size 32, E8M0 scales) or NVFP4 (group_size 16, tensor_group, E4M3
                block scales + per-tensor global). The group's `format` + `weights` block
                come from the QuantScheme (single source of truth).
      group_1 — FP8 (num_bits 8) for the layers kept at native precision (attention,
                dense MLP, shared experts). `fp8_kind` selects MXFP8 (M3, e8m0) or
                block FP8 (Step-3.7/DeepSeek, 128×128 float scales).

    `targets` are layer-index-agnostic regexes and include vLLM's *merged* runtime
    modules (`qkv_proj`, `gate_up_proj`) — without those the fused linears load
    unquantized. The two groups are disjoint; `ignore` covers everything left
    unquantized (router gate, lm_head, embeddings, vision tower, projector).
    """
    # Imported here to avoid a module-load cycle (schemes -> core; output is pure).
    from .schemes import get_scheme

    config_groups: dict[str, dict] = {}
    if mxfp4_modules:
        fp4_format, fp4_weights = get_scheme(fp4_kind).weights_descriptor()
        config_groups["group_0"] = {
            "targets": _dedup(_mxfp4_targets(mxfp4_modules) + _merged_targets(mxfp4_modules)),
            "format": fp4_format,
            "weights": fp4_weights,
        }
    if fp8_modules:
        config_groups["group_1"] = _fp8_group(fp8_modules, fp8_kind, fp8_block_size)
    return {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": config_groups,
        "ignore": _targets(ignore_modules),
    }
