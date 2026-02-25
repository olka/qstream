"""Activation proxy from layernorm weights.

input_layernorm.weight (γ) is the per-channel scale applied after LayerNorm
normalization. It serves as a proxy for per-channel activation magnitude:
    x_normalized = LayerNorm(x) * γ
so E[|x_j|] ≈ γ_j.

γ only applies to tensors whose in_features == hidden_size (gate_proj, up_proj).
Tensors with different in_features (down_proj, attention projections) are skipped.
"""

import re
from pathlib import Path

import torch
from safetensors import safe_open


def extract_layer_index(key: str) -> int | None:
    """Extract transformer layer index from a tensor key.

    Works for keys like:
        model.layers.42.mlp.experts.0.gate_proj.weight
        transformer.layers.7.ffn.gate_proj.weight
    """
    m = re.search(r'\.layers\.(\d+)\.', key)
    return int(m.group(1)) if m else None


def load_layernorm_gammas(
    model_dir: Path,
    shard_files: list[str],
) -> dict[int, torch.Tensor]:
    """Pre-scan shards and collect input_layernorm.weight tensors by layer index.

    Returns {layer_idx: gamma_tensor} where gamma is float32, shape [hidden_size].
    These are small tensors — loading all layers is cheap even for large models.
    """
    gammas: dict[int, torch.Tensor] = {}
    for shard_file in shard_files:
        with safe_open(str(model_dir / shard_file), framework="pt", device="cpu") as f:
            for k in f.keys():
                if "input_layernorm.weight" in k:
                    layer_idx = extract_layer_index(k)
                    if layer_idx is not None and layer_idx not in gammas:
                        gammas[layer_idx] = f.get_tensor(k).to(torch.float32)
    return gammas
