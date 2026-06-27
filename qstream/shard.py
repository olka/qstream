"""Safetensors shard processing: load, quantize, save."""

import ctypes
import gc
import json
import os
import re
import struct

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# glibc malloc_trim() releases freed arenas back to the OS between shards (Linux).
# Not available on macOS (no libc.so.6 / no malloc_trim) — degrade to a no-op there.
try:
    _libc = ctypes.CDLL("libc.so.6")
    _malloc_trim = _libc.malloc_trim
except (OSError, AttributeError):
    _malloc_trim = None

from .calibrate import load_calibration_stats
from .core import BLOCK_SIZE, quantize_fp8
from .gamma import extract_expert_index, extract_layer_index
from .handlers import (
    FusedExpertHandler,
    _FUSED_EXPERT_PATTERNS,
    _passes_exclude_filter,
    get_handler,
    should_quantize_key,
)
from .schemes import get_scheme


def _scheme_quantize(scheme, weight, gamma, scale_percentile):
    """Quantize a 2D weight, or a 3D [E, out, in] expert stack, with the named-scale
    contract of QuantScheme.quantize.

    3D is looped per-expert (not batched) to bound peak memory — the MSE candidate
    tensors are large — and, for NVFP4, this yields the correct per-expert global scale.
    The expert dim is preserved in every returned tensor (stacked), so callers index
    ``t[i]`` uniformly across suffixes.
    """
    if weight.ndim == 3:
        per = [
            scheme.quantize(weight[i], gamma=gamma, scale_percentile=scale_percentile)
            for i in range(weight.shape[0])
        ]
        return {suffix: torch.stack([p[suffix] for p in per]) for suffix in per[0]}
    return scheme.quantize(weight, gamma=gamma, scale_percentile=scale_percentile)


def _normalize_fp8_scale(scale: torch.Tensor) -> torch.Tensor:
    """Normalize an FP8 scale to vLLM's compressed-tensors expectation: float32, rank-1.

    vLLM's PerTensorScaleParameter handles both scalar and shape (1,) sources, but FP32
    is the consistent expected dtype across loader paths.
    """
    s = scale.to(torch.float32)
    if s.ndim == 0:
        s = s.unsqueeze(0)
    return s


def _smooth_norm_key(key: str) -> str | None:
    """Return the preceding layernorm weight key for SmoothQuant, or None.

    SmoothQuant can only be applied to layers with a preceding norm whose
    weights can absorb the inverse smoothing factor.
    """
    # MolmoAct: att_proj ← attn_norm, ff_proj ← ff_norm
    if ".att_proj.weight" in key:
        return key.replace(".att_proj.weight", ".attn_norm.weight")
    if ".ff_proj.weight" in key:
        return key.replace(".ff_proj.weight", ".ff_norm.weight")
    # Standard (Qwen, Llama): q/k/v_proj ← input_layernorm, gate/up_proj ← post_attention_layernorm
    if any(f".{p}.weight" in key for p in ("q_proj", "k_proj", "v_proj", "qkv_proj")):
        # Find the layer prefix and append input_layernorm
        parts = key.split(".")
        for i, p in enumerate(parts):
            if p in ("self_attn", "attention"):
                return ".".join(parts[:i] + ["input_layernorm", "weight"])
        return None
    if any(f".{p}.weight" in key for p in ("gate_proj", "up_proj", "gate_up_proj")):
        parts = key.split(".")
        for i, p in enumerate(parts):
            if p == "mlp":
                return ".".join(parts[:i] + ["post_attention_layernorm", "weight"])
        return None
    return None


def activation_type_from_key(key: str) -> str | None:
    """Map a weight tensor key to its activation stat type.

    Returns one of: "pre_attn", "post_attn", "pre_mlp", "pre_down", or None.
    These correspond to the keys produced by LayerRunner.run_layer().
    """
    # Standard (Qwen, Llama, etc.)
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
    # MolmoAct naming: att_proj (fused QKV), attn_out, ff_proj (fused gate+up), ff_out
    if ".att_proj" in key:
        return "pre_attn"
    if ".attn_out" in key:
        return "post_attn"
    if ".ff_proj" in key:
        return "pre_mlp"
    if ".ff_out" in key:
        return "pre_down"
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
    """Classify a shard's weight format from its scale tensors.

    Returns:
        'mxfp8' — has uint8 `weight_scale_inv` (OCP MXFP8 e8m0, block [1,32]; MiniMax-M3).
        'fp8'   — has float `weight_scale_inv` (DeepSeek/Qwen block FP8, or per-tensor).
        'fp16'  — no scale tensors (plain BF16/FP16).
    """
    with safe_open(shard_path, framework="pt") as f:
        for k in f.keys():
            if k.endswith(".weight_scale_inv"):
                if f.get_slice(k).get_dtype() == "U8":
                    return "mxfp8"
                return "fp8"
    return "fp16"


def classify_shard(
    input_path: str,
    exclude_patterns: list[str],
    include_patterns: list[str] | None = None,
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

        would_quantize = False
        if ndim == 3 and _FUSED_EXPERT_PATTERNS.search(key):
            would_quantize = should_quantize_key(key, exclude_patterns, include_patterns)
        elif ndim == 2 and key.endswith(".weight"):
            would_quantize = should_quantize_key(key, exclude_patterns, include_patterns)

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
    include_patterns: list[str] | None = None,
    quant_format: str = "mxfp4",
    expert_quantize_set: set[tuple[int, int]] | None = None,
) -> dict[str, str]:
    """Quantize all eligible weights in one safetensors shard.

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

    # Packed-FP4 formats (mxfp4/nvfp4) are handled by a QuantScheme; fp8 stays a
    # special case below. The scheme owns encoding, the named scale set, and group size.
    scheme = get_scheme(quant_format) if quant_format != "fp8" else None
    if quant_format == "nvfp4" and output_format == "fused":
        raise ValueError(
            "NVFP4 with output_format='fused' is not supported; use output_format='ct' "
            "(the compressed-tensors per-expert layout)."
        )

    # Load calibration stats inside the worker — avoids cross-process tensor fd sharing.
    activation_stats = None
    if calibration_stats_path:
        activation_stats = load_calibration_stats(calibration_stats_path)
    output_tensors = {}
    smoothed_norms: dict[str, torch.Tensor] = {}  # SmoothQuant modified norms
    n_quantized = 0
    n_gamma_used = 0
    n_smoothed = 0
    n_fp8_kept = 0

    with safe_open(input_path, framework="pt") as f:
        keys = list(f.keys())

        # Build weight → scale_inv and weight → activation_scale lookups for FP8 shards
        scale_inv_map: dict[str, str] = {}
        act_scale_map: dict[str, str] = {}
        if input_format in ("fp8", "mxfp8"):
            for k in keys:
                if k.endswith(".weight_scale_inv"):
                    scale_inv_map[k.replace(".weight_scale_inv", ".weight")] = k
                elif k.endswith(".activation_scale"):
                    act_scale_map[k.replace(".activation_scale", ".weight")] = k
                elif k.endswith(".input_scale"):
                    act_scale_map[k.replace(".input_scale", ".weight")] = k

        quantize_only_set = set(quantize_only_keys) if quantize_only_keys is not None else None

        for k in keys:
            if k.endswith(".weight_scale_inv") or k.endswith(".activation_scale"):
                continue

            # In quantize-only mode, skip keys we don't need to process
            if quantize_only_set is not None and k not in quantize_only_set:
                continue

            t = f.get_tensor(k)

            # Selective per-expert quantization: if expert_quantize_set is
            # provided, only quantize experts in the set. Others keep FP8.
            if expert_quantize_set is not None:
                expert_idx = extract_expert_index(k)
                layer_idx = extract_layer_index(k)
                if expert_idx is not None and layer_idx is not None:
                    if (layer_idx, expert_idx) not in expert_quantize_set:
                        output_tensors[k] = t
                        scale_key = scale_inv_map.get(k)
                        if scale_key:
                            output_tensors[k.replace(".weight", ".weight_scale")] = (
                                _normalize_fp8_scale(f.get_tensor(scale_key))
                            )
                            act_key = act_scale_map.get(k)
                            if act_key is not None:
                                output_tensors[k.replace(".weight", ".input_scale")] = (
                                    _normalize_fp8_scale(f.get_tensor(act_key))
                                )
                        n_fp8_kept += 1
                        continue

            handler = get_handler(k, t)

            if handler is not None and handler.should_quantize(k, t, exclude_patterns, include_patterns):
                weight_bf16 = handler.prepare_weight(
                    k, t, device, input_format, fp8_block_size, scale_inv_map, f,
                )

                align = scheme.group_size if scheme is not None else BLOCK_SIZE
                in_features = weight_bf16.shape[-1]
                if in_features % align != 0:
                    pad = align - (in_features % align)
                    weight_bf16 = torch.nn.functional.pad(weight_bf16, (0, pad))
                    print(f"  WARNING: Padded {k} by {pad} for block alignment ({align})")

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

                if quant_format == "fp8":
                    # SmoothQuant: smooth weight columns by activation magnitudes
                    # and absorb inverse into the preceding layernorm.
                    if g_dev is not None and weight_bf16.ndim == 2:
                        norm_key = _smooth_norm_key(k)
                        if norm_key is not None and norm_key in keys:
                            norm_w = f.get_tensor(norm_key).float().to(device)
                            w_col_max = weight_bf16.abs().amax(dim=0).to(device)
                            alpha = 0.5
                            s = (g_dev.clamp(min=1e-12) ** alpha) / (
                                w_col_max.clamp(min=1e-12) ** (1 - alpha)
                            )
                            s = s.clamp(min=1e-6)
                            weight_bf16 = weight_bf16.to(device) / s.unsqueeze(0)
                            smoothed_norms[norm_key] = (norm_w * s).cpu()
                            weight_bf16 = weight_bf16.cpu()
                            n_smoothed += 1

                    # FP8 E4M3 per-channel quantization
                    if weight_bf16.ndim == 3:
                        q_list, s_list = [], []
                        for i in range(weight_bf16.shape[0]):
                            q, s = quantize_fp8(weight_bf16[i], per_channel=True)
                            q_list.append(q.cpu())
                            s_list.append(s.cpu())
                        quantized = torch.stack(q_list)
                        q_scales = torch.stack(s_list)
                        del q_list, s_list
                    else:
                        quantized, q_scales = quantize_fp8(weight_bf16, per_channel=True)

                    weight_key, scale_key = handler.output_keys(k, quant_format="fp8")
                    output_tensors[weight_key] = quantized.cpu()
                    output_tensors[scale_key] = q_scales.cpu()
                    n_quantized += 1
                else:
                    # Packed-FP4 (mxfp4 / nvfp4) via QuantScheme. The scheme returns a
                    # {suffix: tensor} dict (e.g. weight_packed/weight_scale, plus
                    # weight_global_scale for nvfp4); shard.py only routes the named
                    # tensors to keys — it never branches on the format.
                    if isinstance(handler, FusedExpertHandler) and output_format == "ct":
                        # CT format: per-expert separate tensors. Keys live under
                        # `<prefix>.experts.{i}.PROJ.<suffix>`, matching vLLM's
                        # make_expert_params_mapping (assembled into w13_/w2_ params).
                        #
                        # Split gate/up into halves BEFORE quantizing: NVFP4's per-tensor
                        # global scale must be per emitted weight (gate and up get
                        # separate globals — vLLM's w13_weight_global_scale is [E, 2]).
                        # For MXFP4 (per-block scales, independent rows) splitting first is
                        # bit-identical to quantizing the fused blob and slicing.
                        m = re.match(
                            r"(.+?)\.(experts|moe)\.(gate_up_proj|down_proj|gate_proj|up_proj)(\.weight)?$",
                            k,
                        )
                        prefix, container = m.group(1), m.group(2)
                        base = f"{prefix}.experts" if container == "experts" else f"{prefix}.moe.experts"

                        if "gate_up_proj" in k:
                            half = weight_bf16.shape[1] // 2
                            parts = [
                                ("gate_proj", weight_bf16[:, :half, :]),
                                ("up_proj", weight_bf16[:, half:, :]),
                            ]
                        elif "down_proj" in k:
                            parts = [("down_proj", weight_bf16)]
                        elif "gate_proj" in k:
                            parts = [("gate_proj", weight_bf16)]
                        else:  # up_proj
                            parts = [("up_proj", weight_bf16)]

                        for emit_proj, w in parts:
                            res = _scheme_quantize(scheme, w.contiguous(), g_dev, scale_percentile)
                            for i in range(w.shape[0]):
                                for suffix, t in res.items():
                                    output_tensors[f"{base}.{i}.{emit_proj}.{suffix}"] = (
                                        t[i].contiguous().cpu()
                                    )
                        n_quantized += 1
                    else:
                        res = _scheme_quantize(scheme, weight_bf16, g_dev, scale_percentile)
                        if isinstance(handler, FusedExpertHandler):
                            # vLLM-fork fused (w13) layout — mxfp4 only (nvfp4 guarded out).
                            if "gate_up_proj" in k:
                                res = {
                                    s: FusedExpertHandler.interleave_gate_up(t)
                                    for s, t in res.items()
                                }
                            packed_key, scale_key = handler.output_keys(k)
                            output_tensors[packed_key] = res["weight_packed"].cpu()
                            output_tensors[scale_key] = res["weight_scale"].cpu()
                        else:
                            # Standard 2D: emit `<base>.<suffix>` for every scale tensor.
                            base = k[: -len(".weight")] if k.endswith(".weight") else k
                            for suffix, t in res.items():
                                output_tensors[f"{base}.{suffix}"] = t.contiguous().cpu()
                        n_quantized += 1

            else:
                if isinstance(handler, FusedExpertHandler) and output_format == "ct":
                    # 3D fused expert tensor excluded from quantization.
                    # CT consumers (stock vLLM, MTP draft loader) expect per-expert
                    # 2D .weight tensors — not the [E, N, K] 3D fused layout.
                    base = re.sub(r"\.(gate_up_proj|down_proj|gate_proj|up_proj)$", "", k)
                    n_experts = t.shape[0]
                    t_bf16 = t.to(torch.bfloat16)
                    if "gate_up_proj" in k:
                        N = t.shape[1] // 2
                        for i in range(n_experts):
                            output_tensors[f"{base}.{i}.gate_proj.weight"] = t_bf16[i, :N, :].contiguous()
                            output_tensors[f"{base}.{i}.up_proj.weight"] = t_bf16[i, N:, :].contiguous()
                    elif "down_proj" in k:
                        for i in range(n_experts):
                            output_tensors[f"{base}.{i}.down_proj.weight"] = t_bf16[i].contiguous()
                    elif "gate_proj" in k:
                        for i in range(n_experts):
                            output_tensors[f"{base}.{i}.gate_proj.weight"] = t_bf16[i].contiguous()
                    elif "up_proj" in k:
                        for i in range(n_experts):
                            output_tensors[f"{base}.{i}.up_proj.weight"] = t_bf16[i].contiguous()
                elif k in smoothed_norms:
                    # SmoothQuant: use the smoothed layernorm weights
                    output_tensors[k] = smoothed_norms[k]
                elif input_format == "mxfp8" and t.dtype == torch.float8_e4m3fn:
                    # MXFP8 passthrough: keep the fp8 weight and its uint8 e8m0 scale
                    # bit-exact, renaming weight_scale_inv → weight_scale. The scale
                    # MUST stay uint8 — vLLM's _is_mxfp8 requires scale_dtype == uint8.
                    scale_key = scale_inv_map.get(k)
                    if scale_key is not None:
                        output_tensors[k] = t
                        output_tensors[k.replace(".weight", ".weight_scale")] = (
                            f.get_tensor(scale_key)
                        )
                        n_fp8_kept += 1
                    else:
                        output_tensors[k] = t.to(torch.bfloat16)
                elif input_format == "fp8" and t.dtype == torch.float8_e4m3fn:
                    scale_key = scale_inv_map.get(k)
                    if scale_key is not None:
                        output_tensors[k] = t
                        output_tensors[k.replace(".weight", ".weight_scale")] = (
                            _normalize_fp8_scale(f.get_tensor(scale_key))
                        )
                        act_key = act_scale_map.get(k)
                        if act_key is not None:
                            output_tensors[k.replace(".weight", ".input_scale")] = (
                                _normalize_fp8_scale(f.get_tensor(act_key))
                            )
                    else:
                        output_tensors[k] = t.to(torch.bfloat16)
                else:
                    output_tensors[k] = t

    # Apply any remaining SmoothQuant norms that weren't in the passthrough loop
    # (norm key in a different shard or processed before the linear weight)
    for norm_key, smoothed in smoothed_norms.items():
        if norm_key in output_tensors:
            output_tensors[norm_key] = smoothed

    result_map = {name: os.path.basename(output_path) for name in output_tensors}

    save_file(output_tensors, output_path)
    output_tensors.clear()
    gc.collect()
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if _malloc_trim is not None:
        _malloc_trim(0)

    in_mb = os.path.getsize(input_path) / 1e6
    out_mb = os.path.getsize(output_path) / 1e6
    gamma_note = f", {n_gamma_used} with γ" if (gamma_by_layer or activation_stats) else ""
    smooth_note = f", {n_smoothed} smoothed" if n_smoothed else ""
    fp8_note = f", {n_fp8_kept} FP8 kept" if n_fp8_kept else ""
    print(f"  {shard_name}: {in_mb:.0f}MB → {out_mb:.0f}MB ({n_quantized} quantized{fp8_note}{gamma_note}{smooth_note})")

    return result_map
