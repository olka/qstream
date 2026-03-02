"""Safetensors shard processing: load, quantize, save."""

import ctypes
import gc
import json
import os
import struct

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_libc = ctypes.CDLL("libc.so.6")

from .core import BLOCK_SIZE, quantize_mxfp4
from .fp8 import dequant_fp8_block
from .gamma import extract_layer_index
from .handlers import (
    FusedExpertHandler,
    _FUSED_EXPERT_PATTERNS,
    _passes_exclude_filter,
    get_handler,
)


def activation_type_from_key(key: str) -> str | None:
    """Map a weight tensor key to its activation stat type.

    Returns one of: "pre_attn", "post_attn", "pre_mlp", "pre_down", or None.
    These correspond to the keys produced by Qwen3LayerRunner.run_layer().
    """
    if ".q_proj" in key or ".k_proj" in key or ".v_proj" in key:
        return "pre_attn"
    if ".o_proj" in key:
        return "post_attn"
    if ".gate_proj" in key or ".up_proj" in key or ".w1." in key or ".w3." in key:
        return "pre_mlp"
    if ".down_proj" in key or ".w2." in key:
        return "pre_down"
    # Fused expert keys: gate_up_proj contains both gate and up
    if ".gate_up_proj" in key:
        return "pre_mlp"
    return None


def should_quantize(key: str, exclude_patterns: list[str]) -> bool:
    """Return True if this tensor key should be MXFP4 quantized.

    Rules:
    - Must end in .weight
    - Must not match any exclude pattern (substring match, vLLM compatible)

    Note: This only handles 2D .weight tensors. For 3D fused expert tensors,
    use get_handler() from handlers.py instead.
    """
    if not key.endswith(".weight"):
        return False
    for pattern in exclude_patterns:
        if pattern.strip("*") in key:
            return False
    return True


def detect_input_format(shard_path: str) -> str:
    """Return 'fp8' if the shard has weight_scale_inv keys, else 'fp16'."""
    with safe_open(shard_path, framework="pt") as f:
        for k in f.keys():
            if k.endswith(".weight_scale_inv"):
                return "fp8"
    return "fp16"


def classify_shard(
    input_path: str,
    exclude_patterns: list[str],
) -> tuple[list[str], list[str]]:
    """Classify tensors in a shard by reading only the safetensors header.

    Returns (keys_to_quantize, keys_passthrough).
    No tensor data is loaded — only the JSON header (a few KB).
    """
    with open(input_path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))

    keys_to_quantize = []
    keys_passthrough = []

    for key, meta in header.items():
        if key == "__metadata__":
            continue
        if key.endswith(".weight_scale_inv"):
            continue  # consumed during FP8 dequant, never in output

        shape = meta["shape"]
        ndim = len(shape)

        # Mirror the handler logic from handlers.py
        would_quantize = False
        if ndim == 3 and _FUSED_EXPERT_PATTERNS.search(key):
            would_quantize = _passes_exclude_filter(key, exclude_patterns)
        elif ndim == 2 and key.endswith(".weight"):
            would_quantize = _passes_exclude_filter(key, exclude_patterns)

        if would_quantize:
            keys_to_quantize.append(key)
        else:
            keys_passthrough.append(key)

    return keys_to_quantize, keys_passthrough


def process_shard(
    input_path: str,
    output_path: str,
    exclude_patterns: list[str],
    input_format: str,
    fp8_block_size: int = 128,
    threads_per_worker: int = 2,
    scale_percentile: float = 99.5,
    gamma_by_layer: dict[int, torch.Tensor] | None = None,
    calibration_stats_path: str | None = None,
    device: str = "cpu",
    quantize_only_keys: list[str] | None = None,
    output_format: str = "ct",
) -> dict[str, str]:
    """Quantize all eligible weights in one safetensors shard to MXFP4.

    Args:
        input_path:      Source shard path.
        output_path:     Destination shard path.
        exclude_patterns: Substring patterns for tensors to skip.
        input_format:    'fp8' or 'fp16'.
        fp8_block_size:  FP8 dequantization block size.
        threads_per_worker: torch thread count.
        scale_percentile: Anchor percentile for MSE candidate generation.
        gamma_by_layer:  {layer_idx: γ tensor} from load_layernorm_gammas.
                         None disables activation-aware scaling.
        calibration_stats_path: Path to calibration_stats.json. Loaded per-worker to
                          avoid cross-process tensor sharing (fd limit issues).
                          Takes priority over gamma_by_layer when provided.
        quantize_only_keys: If set, only read and quantize these keys.
                          Passthrough tensors are skipped (assumed to be in a
                          symlinked original shard). Output contains only
                          packed+scale tensors.

    Returns:
        {tensor_name: shard_filename} for the output index.
    """
    torch.set_num_threads(threads_per_worker)
    shard_name = os.path.basename(input_path)

    # Load calibration stats inside the worker — avoids cross-process tensor fd sharing.
    activation_stats = None
    if calibration_stats_path:
        from .calibrate import load_calibration_stats
        activation_stats = load_calibration_stats(calibration_stats_path)
    output_tensors = {}
    n_quantized = 0
    n_gamma_used = 0

    with safe_open(input_path, framework="pt") as f:
        keys = list(f.keys())

        # Build weight → scale_inv lookup for FP8 shards
        scale_inv_map: dict[str, str] = {}
        if input_format == "fp8":
            for k in keys:
                if k.endswith(".weight_scale_inv"):
                    scale_inv_map[k.replace(".weight_scale_inv", ".weight")] = k

        quantize_only_set = set(quantize_only_keys) if quantize_only_keys is not None else None

        for k in keys:
            if k.endswith(".weight_scale_inv"):
                continue

            # In quantize-only mode, skip keys we don't need to process
            if quantize_only_set is not None and k not in quantize_only_set:
                continue

            t = f.get_tensor(k)
            handler = get_handler(k, t)

            if handler is not None and handler.should_quantize(k, t, exclude_patterns):
                # Prepare weight: dequant FP8 or cast to BF16
                weight_bf16 = handler.prepare_weight(
                    k, t, device, input_format, fp8_block_size, scale_inv_map, f,
                )

                # Pad last dim to BLOCK_SIZE if needed
                in_features = weight_bf16.shape[-1]
                if in_features % BLOCK_SIZE != 0:
                    pad = BLOCK_SIZE - (in_features % BLOCK_SIZE)
                    weight_bf16 = torch.nn.functional.pad(weight_bf16, (0, pad))
                    print(f"  WARNING: Padded {k} by {pad} for BLOCK_SIZE alignment")

                # Resolve γ: activation_stats takes priority over gamma_by_layer.
                # γ is used only for MSE weighting; block_max always uses unweighted
                # amax (see core.py) so outlier channels are never suppressed.
                gamma = None
                layer_idx = extract_layer_index(k)
                if activation_stats and layer_idx is not None:
                    act_type = activation_type_from_key(k)
                    g = activation_stats.get(layer_idx, {}).get(act_type)
                    if g is not None and g.shape[0] == weight_bf16.shape[-1]:
                        gamma = g
                        n_gamma_used += 1
                elif gamma_by_layer and layer_idx is not None:
                    if layer_idx in gamma_by_layer:
                        g = gamma_by_layer[layer_idx]
                        if g.shape[0] == weight_bf16.shape[-1]:
                            gamma = g
                            n_gamma_used += 1

                g_dev = gamma.to(device) if gamma is not None else None

                if weight_bf16.ndim == 3:
                    # Quantize expert-by-expert to avoid 3× FP32 blowup on
                    # the full [n_experts, out, in] tensor (~64GB peak for 256 experts).
                    packed_list, scales_list = [], []
                    for i in range(weight_bf16.shape[0]):
                        p, s = quantize_mxfp4(weight_bf16[i], scale_percentile, gamma=g_dev)
                        packed_list.append(p.cpu())
                        scales_list.append(s.cpu())
                    packed = torch.stack(packed_list)
                    scales = torch.stack(scales_list)
                    del packed_list, scales_list
                else:
                    packed, scales = quantize_mxfp4(weight_bf16, scale_percentile, gamma=g_dev)

                if isinstance(handler, FusedExpertHandler) and output_format == "ct":
                    # CT format: output per-expert separate tensors
                    # Base: "model.layers.X.mlp.experts.gate_up_proj" or ".down_proj"
                    base = re.sub(r"\.(gate_up_proj|down_proj|gate_proj|up_proj)$", "", k)
                    n_experts = packed.shape[0]

                    if "gate_up_proj" in k:
                        # packed shape: [E, 2*N, K//2], scales: [E, 2*N, K//32]
                        N = packed.shape[1] // 2
                        for i in range(n_experts):
                            gate_p = packed[i, :N, :].cpu()
                            gate_s = scales[i, :N, :].cpu()
                            up_p = packed[i, N:, :].cpu()
                            up_s = scales[i, N:, :].cpu()
                            output_tensors[f"{base}.{i}.gate_proj.weight_packed"] = gate_p
                            output_tensors[f"{base}.{i}.gate_proj.weight_scale"] = gate_s
                            output_tensors[f"{base}.{i}.up_proj.weight_packed"] = up_p
                            output_tensors[f"{base}.{i}.up_proj.weight_scale"] = up_s
                    elif "down_proj" in k:
                        for i in range(n_experts):
                            output_tensors[f"{base}.{i}.down_proj.weight_packed"] = packed[i].cpu()
                            output_tensors[f"{base}.{i}.down_proj.weight_scale"] = scales[i].cpu()
                    elif "gate_proj" in k:
                        for i in range(n_experts):
                            output_tensors[f"{base}.{i}.gate_proj.weight_packed"] = packed[i].cpu()
                            output_tensors[f"{base}.{i}.gate_proj.weight_scale"] = scales[i].cpu()
                    elif "up_proj" in k:
                        for i in range(n_experts):
                            output_tensors[f"{base}.{i}.up_proj.weight_packed"] = packed[i].cpu()
                            output_tensors[f"{base}.{i}.up_proj.weight_scale"] = scales[i].cpu()
                    n_quantized += 1
                else:
                    # Fused format: interleave gate_up and output as w13/w2
                    if isinstance(handler, FusedExpertHandler) and "gate_up_proj" in k:
                        packed = FusedExpertHandler.interleave_gate_up(packed)
                        scales = FusedExpertHandler.interleave_gate_up(scales)

                    packed_key, scale_key = handler.output_keys(k)
                    output_tensors[packed_key] = packed.cpu()
                    output_tensors[scale_key] = scales.cpu()
                    n_quantized += 1

            else:
                # Pass through, dequanting FP8 where needed
                if input_format == "fp8" and t.dtype == torch.float8_e4m3fn:
                    scale_key = scale_inv_map.get(k)
                    if scale_key is not None:
                        output_tensors[k] = dequant_fp8_block(t, f.get_tensor(scale_key), fp8_block_size)
                    else:
                        output_tensors[k] = t.to(torch.bfloat16)
                else:
                    output_tensors[k] = t

    result_map = {name: os.path.basename(output_path) for name in output_tensors}

    save_file(output_tensors, output_path)
    output_tensors.clear()
    gc.collect()
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    _libc.malloc_trim(0)

    in_mb = os.path.getsize(input_path) / 1e6
    out_mb = os.path.getsize(output_path) / 1e6
    gamma_note = f", {n_gamma_used} with γ" if (gamma_by_layer or activation_stats) else ""
    print(f"  {shard_name}: {in_mb:.0f}MB → {out_mb:.0f}MB ({n_quantized} quantized{gamma_note})")

    return result_map
