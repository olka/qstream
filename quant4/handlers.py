"""Tensor handler abstraction for different weight storage formats.

Each handler encapsulates the logic for:
- Deciding whether a tensor should be quantized
- Preparing the tensor for quantization (dtype conversion, device placement)
- Naming the output keys (packed weights + scales)

New weight formats (e.g. future fused layouts) can be added by implementing
a new handler and registering it in get_handler().
"""

import re

import torch

from .fp8 import dequant_fp8_block

# Patterns that identify 3D fused expert tensors (no .weight suffix).
_FUSED_EXPERT_PATTERNS = re.compile(
    r"\.experts\.(gate_up_proj|down_proj|gate_proj|up_proj)$"
)


def _passes_exclude_filter(key: str, exclude_patterns: list[str]) -> bool:
    """Return True if key is NOT excluded (should be kept/quantized)."""
    for pattern in exclude_patterns:
        if pattern.strip("*") in key:
            return False  # excluded
    return True  # not excluded


class StandardWeightHandler:
    """Handles 2D tensors with .weight suffix.

    Covers: attention projections, shared expert weights, individual expert
    weights (MiniMax-style .experts.{j}.w1.weight), and any other 2D Linear.
    """

    def should_quantize(
        self, key: str, tensor: torch.Tensor, exclude_patterns: list[str]
    ) -> bool:
        if not key.endswith(".weight"):
            return False
        if tensor.ndim != 2:
            return False
        return _passes_exclude_filter(key, exclude_patterns)

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

    def output_keys(self, key: str) -> tuple[str, str]:
        """Return (packed_key, scale_key) for the quantized output."""
        return (
            key.replace(".weight", ".weight_packed"),
            key.replace(".weight", ".weight_scale"),
        )


class FusedExpertHandler:
    """Handles 3D fused expert tensors without .weight suffix.

    Covers: Qwen3.5-style stacked expert tensors like
    experts.gate_up_proj [n_experts, out_dim, in_dim] and
    experts.down_proj [n_experts, out_dim, in_dim].

    These are always BF16 (no FP8 variant exists yet).
    quantize_mxfp4 handles 3D natively via *leading dims.
    """

    def should_quantize(
        self, key: str, tensor: torch.Tensor, exclude_patterns: list[str]
    ) -> bool:
        if tensor.ndim != 3:
            return False
        if not _FUSED_EXPERT_PATTERNS.search(key):
            return False
        return _passes_exclude_filter(key, exclude_patterns)

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
        """Cast to BF16 on device. FP8 not supported for fused experts."""
        return tensor.to(device, torch.bfloat16)

    @staticmethod
    def interleave_gate_up(tensor: torch.Tensor) -> torch.Tensor:
        """Convert contiguous [gate | up] to gpt-oss interleaved [g0,u0,g1,u1,...].

        Input shape: [E, 2*N, K] with first N rows = gate, last N = up.
        Output shape: [E, 2*N, K] with rows alternating gate/up.
        """
        gate, up = tensor.chunk(2, dim=1)
        return torch.stack([gate, up], dim=2).reshape(tensor.shape)

    def output_keys(self, key: str) -> tuple[str, str]:
        """Return (packed_key, scale_key) using vLLM mxfp4 parameter names."""
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
