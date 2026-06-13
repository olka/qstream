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


def build_quantization_config(
    mxfp4_modules: list[str],
    mxfp8_modules: list[str],
    ignore_modules: list[str],
) -> dict:
    """Assemble a compressed-tensors mixed-precision quantization config.

    Two config groups are emitted:
      group_0 — MXFP4 (num_bits 4, group_size 32) for the routed experts.
      group_1 — MXFP8 (num_bits 8, group_size 32, scale_dtype torch.uint8) for the
                FP8 layers kept at native precision (attention, dense MLP, shared
                experts). The `scale_dtype` field is what makes vLLM's `_is_mxfp8`
                select the MXFP8 scheme rather than plain block FP8.

    `targets` are layer-index-agnostic regexes; the two groups are disjoint and
    `ignore` covers everything left unquantized (router gate, lm_head, embeddings,
    vision tower, projector).
    """
    config_groups: dict[str, dict] = {}
    if mxfp4_modules:
        config_groups["group_0"] = {
            "targets": _dedup(_mxfp4_targets(mxfp4_modules) + _merged_targets(mxfp4_modules)),
            "format": "mxfp4-pack-quantized",
            "weights": {
                "num_bits": 4,
                "type": "float",
                "strategy": "group",
                "group_size": 32,
                "symmetric": True,
            },
        }
    if mxfp8_modules:
        config_groups["group_1"] = {
            "targets": _dedup(_targets(mxfp8_modules) + _merged_targets(mxfp8_modules)),
            "format": "mxfp8-quantized",
            "weights": {
                "num_bits": 8,
                "type": "float",
                "strategy": "group",
                "group_size": 32,
                "symmetric": True,
                "scale_dtype": "torch.uint8",
            },
        }
    return {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": config_groups,
        "ignore": _targets(ignore_modules),
    }
