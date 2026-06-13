"""Tensor handler abstraction for different weight storage formats.

Each handler encapsulates the logic for:
- Deciding whether a tensor should be quantized
- Preparing the tensor for quantization (dtype conversion, device placement)
- Naming the output keys (packed weights + scales)

New weight formats (e.g. future fused layouts) can be added by implementing
a new handler and registering it in get_handler().
"""

import re
from fnmatch import fnmatch

import torch

from .fp8 import dequant_fp8_block

# Patterns that identify 3D fused/stacked expert tensors.
# Covers two layouts:
#   - Qwen3.5: `...experts.{gate_up_proj|down_proj|...}`     (no .weight suffix)
#   - Step-3.7: `...moe.{gate_proj|up_proj|down_proj}.weight` (with .weight suffix)
_FUSED_EXPERT_PATTERNS = re.compile(
    r"(\.experts|\.moe)\.(gate_up_proj|down_proj|gate_proj|up_proj)(\.weight)?$"
)


def _match(key: str, pattern: str) -> bool:
    """Match key against pattern: fnmatch glob if '*' present, else substring."""
    if "*" in pattern:
        return fnmatch(key, pattern)
    return pattern in key


def _passes_exclude_filter(key: str, exclude_patterns: list[str]) -> bool:
    """Return True if key is NOT excluded (should be kept/quantized)."""
    for pattern in exclude_patterns:
        if _match(key, pattern):
            return False  # excluded
    return True  # not excluded


def _passes_include_filter(key: str, include_patterns: list[str]) -> bool:
    """Return True if key matches any include pattern (should be quantized)."""
    for pattern in include_patterns:
        if _match(key, pattern):
            return True
    return False


def should_quantize_key(
    key: str,
    exclude_patterns: list[str],
    include_patterns: list[str] | None = None,
) -> bool:
    """Unified filter: include_patterns overrides exclude_patterns when set."""
    if include_patterns is not None:
        return _passes_include_filter(key, include_patterns)
    return _passes_exclude_filter(key, exclude_patterns)


class StandardWeightHandler:
    """Handles 2D tensors with .weight suffix.

    Covers: attention projections, shared expert weights, individual expert
    weights (MiniMax-style .experts.{j}.w1.weight), and any other 2D Linear.
    """

    def should_quantize(
        self, key: str, tensor: torch.Tensor, exclude_patterns: list[str],
        include_patterns: list[str] | None = None,
    ) -> bool:
        if not key.endswith(".weight"):
            return False
        if tensor.ndim != 2:
            return False
        return should_quantize_key(key, exclude_patterns, include_patterns)

    def prepare_weight(
        self,
        key: str,
        tensor: torch.Tensor,
        device: str,
        input_format: str,
        fp8_block_size: int,
        scale_inv_map: dict[str, str],
        shard_file,
    ) -> torch.Tensor:
        """Dequantize FP8 or cast to BF16, return tensor on device."""
        if input_format == "fp8":
            scale_key = scale_inv_map.get(key)
            if scale_key is None:
                return tensor.to(device, torch.bfloat16)
            return dequant_fp8_block(
                tensor.to(device),
                shard_file.get_tensor(scale_key).to(device),
                fp8_block_size,
            )
        return tensor.to(device, torch.bfloat16)

    def output_keys(self, key: str, quant_format: str = "mxfp4") -> tuple[str, str]:
        """Return (weight_key, scale_key) for the quantized output."""
        if quant_format == "fp8":
            return (key, key.replace(".weight", ".weight_scale"))
        return (
            key.replace(".weight", ".weight_packed"),
            key.replace(".weight", ".weight_scale"),
        )


class FusedExpertHandler:
    """Handles 3D stacked expert tensors.

    Two source layouts supported:
      - Qwen3.5: `...experts.{gate_up_proj|down_proj}` (no .weight, always BF16)
      - Step-3.7: `...moe.{gate_proj|up_proj|down_proj}.weight` (FP8 with 128x128 blocks)

    quantize_mxfp4 handles the 3D shape natively via *leading dims.
    """

    def should_quantize(
        self, key: str, tensor: torch.Tensor, exclude_patterns: list[str],
        include_patterns: list[str] | None = None,
    ) -> bool:
        if tensor.ndim != 3:
            return False
        if not _FUSED_EXPERT_PATTERNS.search(key):
            return False
        return should_quantize_key(key, exclude_patterns, include_patterns)

    def prepare_weight(
        self,
        key: str,
        tensor: torch.Tensor,
        device: str,
        input_format: str,
        fp8_block_size: int,
        scale_inv_map: dict[str, str],
        shard_file,
    ) -> torch.Tensor:
        """Dequant 3D FP8 (Step-3.7) or cast to BF16 (Qwen3.5)."""
        if input_format == "fp8" and tensor.dtype == torch.float8_e4m3fn:
            scale_key = scale_inv_map.get(key)
            if scale_key is not None:
                return dequant_fp8_block(
                    tensor.to(device),
                    shard_file.get_tensor(scale_key).to(device),
                    fp8_block_size,
                )
        return tensor.to(device, torch.bfloat16)

    @staticmethod
    def interleave_gate_up(tensor: torch.Tensor) -> torch.Tensor:
        """Convert contiguous [gate | up] to gpt-oss interleaved [g0,u0,g1,u1,...].

        Input shape: [E, 2*N, K] with first N rows = gate, last N = up.
        Output shape: [E, 2*N, K] with rows alternating gate/up.
        """
        gate, up = tensor.chunk(2, dim=1)
        return torch.stack([gate, up], dim=2).reshape(tensor.shape)

    def output_keys(self, key: str, quant_format: str = "mxfp4") -> tuple[str, str]:
        """Return (weight_key, scale_key) for the fused vLLM-fork output format.

        CT per-expert key construction is handled inline in process_shard since it
        needs to fan out one input tensor into n_experts output tensors.
        """
        if quant_format == "fp8":
            return (key, key + "_scale")
        if "gate_up_proj" in key:
            base = key.replace("gate_up_proj", "w13_weight")
            return (base, base + "_scale")
        elif "down_proj" in key:
            base = key.replace("down_proj", "w2_weight")
            return (base, base + "_scale")
        else:
            return (key, key + "_scale")


# Handler registry — checked in order, first match wins.
_HANDLERS: list = [FusedExpertHandler(), StandardWeightHandler()]


def get_handler(
    key: str, tensor: torch.Tensor
) -> StandardWeightHandler | FusedExpertHandler | None:
    """Select the appropriate handler for a tensor, or None if it shouldn't be quantized.

    This is a lookup only — does NOT check exclude patterns.
    Call handler.should_quantize() separately for the full decision.
    """
    if tensor.ndim == 3 and _FUSED_EXPERT_PATTERNS.search(key):
        return _HANDLERS[0]  # FusedExpertHandler
    if key.endswith(".weight") and tensor.ndim == 2:
        return _HANDLERS[1]  # StandardWeightHandler
    return None
