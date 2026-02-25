"""Lightweight per-layer calibration for activation-aware quantization.

Architecture
------------
Real activation statistics require running the model forward pass. We do this
one transformer layer at a time to stay within CPU RAM:

    tokens → embeddings
           → [load layer 0 weights, run layer 0, capture stats, unload]
           → [load layer 1 weights, run layer 1, capture stats, unload]
           → ...
           → save stats JSON

Stats format (per layer)
------------------------
{layer_idx: {"pre_attn":  tensor[hidden_size],       # input to q/k/v_proj
              "post_attn": tensor[hidden_size],       # input to o_proj
              "pre_mlp":   tensor[hidden_size],       # input to expert gate/up_proj
              "pre_down":  tensor[moe_intermediate_size]}}  # input to down_proj

All values are mean(|x|, dim=tokens) over calibration tokens.

These stats replace the γ proxy (input_layernorm.weight) with real activation
magnitudes, enabling proper γ-weighted MSE quantization.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Protocol

import torch
import torch.nn.functional as F
from safetensors import safe_open

from .fp8 import dequant_fp8_block


# ---------------------------------------------------------------------------
# RoPE and norm helpers
# ---------------------------------------------------------------------------

def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Standard RMSNorm: x / rms(x) * weight."""
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by half for RoPE."""
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def _apply_rotary(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to q and k."""
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


def _build_rope(
    T: int,
    head_dim: int,
    theta: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build RoPE cos/sin tables.

    Returns:
        cos, sin — both [1, T, head_dim]
    """
    half = head_dim // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, half, dtype=torch.float32, device=device) / half)
    )
    t = torch.arange(T, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)        # [T, half]
    emb = torch.cat([freqs, freqs], dim=-1)  # [T, head_dim]
    return emb.cos()[None], emb.sin()[None]  # [1, T, head_dim]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class LayerRunner(Protocol):
    """Model-specific transformer layer executor.

    Implement one of these per model family (MiniMax, Qwen3, DeepSeek, ...).
    """

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids [B, T] → hidden states [B, T, hidden_size]."""
        ...

    def run_layer(
        self,
        hidden: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run one transformer layer.

        Args:
            hidden: [B, T, hidden_size] input hidden states.
            layer_idx: Which layer to run (weights loaded internally).

        Returns:
            (output_hidden, stats_dict)
            stats_dict keys: pre_attn, post_attn, pre_mlp, pre_down
            Each value is mean(|x|, dim=tokens) tensor.
        """
        ...


def collect_activation_stats(
    runner: LayerRunner,
    token_ids: torch.Tensor,
    n_layers: int,
) -> dict[int, dict[str, torch.Tensor]]:
    """Run calibration forward pass and collect per-channel activation stats.

    Args:
        runner: Model-specific LayerRunner implementation.
        token_ids: [B, T] calibration token IDs.
        n_layers: Number of transformer layers.

    Returns:
        {layer_idx: {"pre_attn": tensor, "post_attn": tensor,
                     "pre_mlp": tensor, "pre_down": tensor}}
    """
    stats: dict[int, dict[str, torch.Tensor]] = {}

    hidden = runner.embed(token_ids)  # [B, T, hidden_size]

    for layer_idx in range(n_layers):
        print(f"  Layer {layer_idx + 1}/{n_layers}...", end="\r", flush=True)
        hidden, layer_stats = runner.run_layer(hidden, layer_idx)
        stats[layer_idx] = layer_stats

    print(f"Collected stats for {len(stats)} layers")
    return stats


def load_calibration_stats(path: str | Path) -> dict[int, dict[str, torch.Tensor]]:
    """Load stats JSON and return {layer_idx: {act_type: tensor}}."""
    with open(path) as f:
        raw = json.load(f)
    return {
        int(k): {
            act_type: torch.tensor(v_list, dtype=torch.float32)
            for act_type, v_list in layer_dict.items()
        }
        for k, layer_dict in raw.items()
    }


# ---------------------------------------------------------------------------
# Layer runner stubs — implement per model family
# ---------------------------------------------------------------------------

class MiniMaxLayerRunner:
    """Streaming layer runner for MiniMax-M2.5 (229B, 256 experts, 62 layers).

    expert_buffer controls how many experts are loaded at once during dispatch.
    Peak expert memory ≈ expert_buffer × 54 MB (3 × [1536,3072] float32).
    Default 32 → ~1.7 GB. Set higher to trade more RAM for fewer shard reads.

    Key differences from Qwen3:
    - QK-norm is per-layer: weight shape [n_heads*head_dim], applied before reshape
    - Partial RoPE: only first rotary_dim=64 dims of head_dim=128 are rotated
    - Sigmoid routing (not softmax) + e_score_correction_bias
    - Expert weights named w1 (gate), w2 (down), w3 (up) under block_sparse_moe
    """

    def __init__(self, model_dir: str | Path, device: str = "cpu", expert_buffer: int = 32):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.expert_buffer = expert_buffer

        with open(self.model_dir / "config.json") as f:
            cfg = json.load(f)

        self.hidden_size: int = cfg["hidden_size"]
        self.n_heads: int = cfg["num_attention_heads"]
        self.n_kv_heads: int = cfg.get("num_key_value_heads", self.n_heads)
        self.head_dim: int = cfg.get("head_dim", self.hidden_size // self.n_heads)
        self.rotary_dim: int = cfg.get("rotary_dim", self.head_dim)
        self.rope_theta: float = float(cfg.get("rope_theta", 10000.0))
        self.rms_norm_eps: float = float(cfg.get("rms_norm_eps", 1e-6))
        self.n_experts: int = cfg.get("num_local_experts", cfg.get("num_experts", 0))
        self.n_experts_per_tok: int = cfg.get("num_experts_per_tok", 2)
        self.moe_intermediate_size: int = cfg.get("intermediate_size", 0)

        index_path = self.model_dir / "model.safetensors.index.json"
        with open(index_path) as f:
            self._weight_map: dict[str, str] = json.load(f)["weight_map"]

        sample_shard = next(iter(set(self._weight_map.values())))
        with safe_open(str(self.model_dir / sample_shard), framework="pt") as sf:
            self._is_fp8 = any(k.endswith(".weight_scale_inv") for k in sf.keys())

        if self._is_fp8:
            print("  Detected FP8 weights (block-quantized)")

        # Pre-build: layer_idx → {shard: [expert_idx, ...]}
        # Used for memory-efficient shard-by-shard expert streaming.
        self._expert_shards: dict[int, dict[str, list[int]]] = {}
        ep_base = "model.layers.{}.block_sparse_moe.experts.{}.w1.weight"
        for li in range(cfg.get("num_hidden_layers", 0)):
            shard_map: dict[str, list[int]] = {}
            for ei in range(self.n_experts):
                key = ep_base.format(li, ei)
                shard = self._weight_map.get(key)
                if shard:
                    shard_map.setdefault(shard, []).append(ei)
            self._expert_shards[li] = shard_map

    def _load_tensors(self, keys: list[str]) -> dict[str, torch.Tensor]:
        shard_to_keys: dict[str, list[str]] = {}
        for k in keys:
            shard = self._weight_map.get(k)
            if shard:
                shard_to_keys.setdefault(shard, []).append(k)
        result: dict[str, torch.Tensor] = {}
        for shard, shard_keys in shard_to_keys.items():
            with safe_open(str(self.model_dir / shard), framework="pt") as f:
                for k in shard_keys:
                    result[k] = f.get_tensor(k)
        return result

    def _load_layer(self, layer_idx: int) -> dict[str, torch.Tensor]:
        """Load non-expert weights for one layer (attention + layernorms + router)."""
        prefix = f"model.layers.{layer_idx}."
        layer_keys = [
            k for k in self._weight_map
            if k.startswith(prefix) and ".experts." not in k
        ]
        scale_inv_keys = {k for k in layer_keys if k.endswith(".weight_scale_inv")}
        weight_keys = [k for k in layer_keys if k not in scale_inv_keys]
        raw = self._load_tensors(layer_keys)
        scale_inv_lookup = {
            k.replace(".weight_scale_inv", ".weight"): k for k in scale_inv_keys
        }
        result: dict[str, torch.Tensor] = {}
        for k in weight_keys:
            t = raw[k]
            if t.dtype == torch.float8_e4m3fn:
                scale_key = scale_inv_lookup.get(k)
                if scale_key is not None and scale_key in raw:
                    t = dequant_fp8_block(t, raw[scale_key]).float()
                else:
                    t = t.float()
            else:
                t = t.float()
            result[k[len(prefix):]] = t.to(self.device)
        return result

    def _load_one_expert(
        self, sf, shard: str, layer_idx: int, expert_idx: int
    ) -> dict[str, torch.Tensor]:
        """Load w1/w2/w3 for a single expert from an already-open shard handle.

        Keeping the shard open across experts avoids repeated file opens while
        loading one expert at a time keeps peak memory to ~54 MB (3 × [1536,3072]).
        """
        ep = f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_idx}."
        ew: dict[str, torch.Tensor] = {}
        for wname in ("w1", "w2", "w3"):
            wkey = f"{ep}{wname}.weight"
            skey = f"{ep}{wname}.weight_scale_inv"
            t = sf.get_tensor(wkey)
            if t.dtype == torch.float8_e4m3fn and self._weight_map.get(skey) == shard:
                t = dequant_fp8_block(t, sf.get_tensor(skey)).float()
            else:
                t = t.float()
            ew[wname] = t.to(self.device)
        return ew

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        embed_key = "model.embed_tokens.weight"
        scale_key = "model.embed_tokens.weight_scale_inv"
        keys = [embed_key]
        if scale_key in self._weight_map:
            keys.append(scale_key)
        raw = self._load_tensors(keys)
        embed_w = raw[embed_key]
        if embed_w.dtype == torch.float8_e4m3fn and scale_key in raw:
            embed_w = dequant_fp8_block(embed_w, raw[scale_key]).float()
        else:
            embed_w = embed_w.float()
        hidden = F.embedding(token_ids.cpu(), embed_w.cpu())
        return hidden.to(self.device)

    def run_layer(
        self,
        hidden: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        w = self._load_layer(layer_idx)
        B, T, H = hidden.shape

        # ------------------------------------------------------------------
        # 1. Pre-attention RMSNorm + stat
        # ------------------------------------------------------------------
        x_attn = _rms_norm(hidden, w["input_layernorm.weight"], self.rms_norm_eps)
        stat_pre_attn = x_attn.reshape(-1, H).abs().mean(0).cpu()

        # ------------------------------------------------------------------
        # 2. QKV projections
        # ------------------------------------------------------------------
        q = F.linear(x_attn, w["self_attn.q_proj.weight"])  # [B, T, n_heads*head_dim]
        k = F.linear(x_attn, w["self_attn.k_proj.weight"])  # [B, T, n_kv_heads*head_dim]
        v = F.linear(x_attn, w["self_attn.v_proj.weight"])  # [B, T, n_kv_heads*head_dim]

        # ------------------------------------------------------------------
        # 3. QK-norm — per-layer: weight covers all heads, applied before reshape
        # ------------------------------------------------------------------
        if "self_attn.q_norm.weight" in w:
            q = _rms_norm(q, w["self_attn.q_norm.weight"], self.rms_norm_eps)
        if "self_attn.k_norm.weight" in w:
            k = _rms_norm(k, w["self_attn.k_norm.weight"], self.rms_norm_eps)

        # Reshape to [B, heads, T, head_dim]
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # ------------------------------------------------------------------
        # 4. Partial RoPE — only first rotary_dim dims, rest unchanged
        # ------------------------------------------------------------------
        cos, sin = _build_rope(T, self.rotary_dim, self.rope_theta, self.device)
        q_rot, k_rot = _apply_rotary(
            q[..., :self.rotary_dim], k[..., :self.rotary_dim], cos, sin
        )
        q = torch.cat([q_rot, q[..., self.rotary_dim:]], dim=-1)
        k = torch.cat([k_rot, k[..., self.rotary_dim:]], dim=-1)

        # ------------------------------------------------------------------
        # 5. GQA expand
        # ------------------------------------------------------------------
        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # ------------------------------------------------------------------
        # 6. Attention (bidirectional)
        # ------------------------------------------------------------------
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # ------------------------------------------------------------------
        # 7. Post-attention stat + o_proj + residual
        # ------------------------------------------------------------------
        post_attn = attn_out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        stat_post_attn = post_attn.reshape(-1, self.n_heads * self.head_dim).abs().mean(0).cpu()
        hidden = hidden + F.linear(post_attn, w["self_attn.o_proj.weight"])

        # ------------------------------------------------------------------
        # 8. Pre-MLP RMSNorm + stat
        # ------------------------------------------------------------------
        x_mlp = _rms_norm(hidden, w["post_attention_layernorm.weight"], self.rms_norm_eps)
        stat_pre_mlp = x_mlp.reshape(-1, H).abs().mean(0).cpu()

        # ------------------------------------------------------------------
        # 9. Sigmoid routing with correction bias
        # ------------------------------------------------------------------
        x_flat = x_mlp.reshape(-1, H)
        router_logits = F.linear(x_flat, w["block_sparse_moe.gate.weight"])
        if "block_sparse_moe.e_score_correction_bias" in w:
            router_logits = router_logits + w["block_sparse_moe.e_score_correction_bias"]
        routing_weights = torch.sigmoid(router_logits.float())
        topk_weights, topk_indices = torch.topk(
            routing_weights, self.n_experts_per_tok, dim=-1
        )
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # ------------------------------------------------------------------
        # 10. Expert dispatch (w1=gate, w3=up, w2=down)
        #     Stream expert weights shard-by-shard to keep peak memory low.
        #     ~2 experts/shard × 3 matrices × [1536,3072] ≈ 100MB peak vs 13.5GB
        # ------------------------------------------------------------------
        pre_down_parts: list[torch.Tensor] = []
        moe_out = torch.zeros_like(x_flat)

        for shard, expert_indices in self._expert_shards[layer_idx].items():
            with safe_open(str(self.model_dir / shard), framework="pt") as sf:
                # Process experts in chunks of expert_buffer to balance RAM vs I/O.
                # Peak memory ≈ expert_buffer × 54 MB (3 matrices per expert).
                for chunk_start in range(0, len(expert_indices), self.expert_buffer):
                    chunk = expert_indices[chunk_start : chunk_start + self.expert_buffer]

                    # Load buffer
                    loaded = {ei: self._load_one_expert(sf, shard, layer_idx, ei) for ei in chunk}

                    # Process buffer
                    for expert_idx, ew in loaded.items():
                        expert_mask = topk_indices == expert_idx
                        token_mask = expert_mask.any(dim=-1)
                        if token_mask.any():
                            token_indices = token_mask.nonzero(as_tuple=True)[0]
                            x_e = x_flat[token_indices]

                            gate_out = F.linear(x_e, ew["w1"])
                            up_out = F.linear(x_e, ew["w3"])
                            pre_down = F.silu(gate_out) * up_out

                            pre_down_parts.append(pre_down.detach().cpu())

                            down_out = F.linear(pre_down, ew["w2"])
                            expert_weight = (topk_weights * expert_mask.float()).sum(dim=-1)[token_indices]
                            moe_out.index_add_(0, token_indices, down_out * expert_weight.unsqueeze(-1))

                    del loaded

        if pre_down_parts:
            stat_pre_down = torch.cat(pre_down_parts, dim=0).abs().mean(0)
        else:
            stat_pre_down = torch.zeros(self.moe_intermediate_size)

        hidden = hidden + moe_out.reshape(B, T, H)

        del w
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return hidden, {
            "pre_attn": stat_pre_attn,
            "post_attn": stat_post_attn,
            "pre_mlp": stat_pre_mlp,
            "pre_down": stat_pre_down,
        }


# ---------------------------------------------------------------------------
# Qwen3 MoE LayerRunner
# ---------------------------------------------------------------------------

class Qwen3LayerRunner:
    """Streaming layer runner for Qwen3 MoE models.

    Loads one transformer layer at a time from safetensors shards,
    runs the forward pass, captures per-channel activation statistics,
    then frees the weights before moving to the next layer.

    Memory: peak ~2.5GB float32 per layer for 128-expert configs.
    Handles both FP8 (compressed-tensors) and BF16/FP16 input weights.
    """

    def __init__(self, model_dir: str | Path, device: str = "cpu"):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)

        with open(self.model_dir / "config.json") as f:
            cfg = json.load(f)

        self.hidden_size: int = cfg["hidden_size"]
        self.n_heads: int = cfg["num_attention_heads"]
        self.n_kv_heads: int = cfg.get("num_key_value_heads", self.n_heads)
        self.head_dim: int = cfg.get("head_dim", self.hidden_size // self.n_heads)
        self.rope_theta: float = float(cfg.get("rope_theta", 10000.0))
        self.rms_norm_eps: float = float(cfg.get("rms_norm_eps", 1e-6))
        self.n_experts: int = cfg.get("num_experts", cfg.get("num_local_experts", 0))
        self.n_experts_per_tok: int = cfg.get(
            "num_experts_per_tok", cfg.get("num_selected_experts", 2)
        )
        self.moe_intermediate_size: int = cfg.get(
            "moe_intermediate_size", cfg.get("intermediate_size", 0)
        )
        self.norm_topk_prob: bool = bool(cfg.get("norm_topk_prob", True))

        # Build weight_map: tensor_name → shard_filename
        index_path = self.model_dir / "model.safetensors.index.json"
        with open(index_path) as f:
            self._weight_map: dict[str, str] = json.load(f)["weight_map"]

        # Detect FP8 input
        sample_shard = next(iter(set(self._weight_map.values())))
        with safe_open(str(self.model_dir / sample_shard), framework="pt") as sf:
            self._is_fp8 = any(k.endswith(".weight_scale_inv") for k in sf.keys())

        if self._is_fp8:
            print(f"  Detected FP8 weights (block-quantized)")

    def _load_tensors(self, keys: list[str]) -> dict[str, torch.Tensor]:
        """Load a set of tensor keys from disk, grouped by shard."""
        shard_to_keys: dict[str, list[str]] = {}
        for k in keys:
            shard = self._weight_map.get(k)
            if shard:
                shard_to_keys.setdefault(shard, []).append(k)

        result: dict[str, torch.Tensor] = {}
        for shard, shard_keys in shard_to_keys.items():
            with safe_open(str(self.model_dir / shard), framework="pt") as f:
                for k in shard_keys:
                    result[k] = f.get_tensor(k)
        return result

    def _load_layer(self, layer_idx: int) -> dict[str, torch.Tensor]:
        """Load and dequantize all weights for one transformer layer.

        Returns a dict keyed by short name (layer prefix stripped),
        all tensors as float32 on self.device.
        FP8 weights are dequantized inline using block scales.
        """
        prefix = f"model.layers.{layer_idx}."
        layer_keys = [k for k in self._weight_map if k.startswith(prefix)]

        scale_inv_keys = {k for k in layer_keys if k.endswith(".weight_scale_inv")}
        weight_keys = [k for k in layer_keys if k not in scale_inv_keys]

        raw = self._load_tensors(layer_keys)

        # Build weight → scale_inv lookup
        scale_inv_lookup = {
            k.replace(".weight_scale_inv", ".weight"): k for k in scale_inv_keys
        }

        result: dict[str, torch.Tensor] = {}
        for k in weight_keys:
            t = raw[k]
            if t.dtype == torch.float8_e4m3fn:
                scale_key = scale_inv_lookup.get(k)
                if scale_key is not None and scale_key in raw:
                    t = dequant_fp8_block(t, raw[scale_key]).float()
                else:
                    t = t.float()
            else:
                t = t.float()
            short_key = k[len(prefix):]
            result[short_key] = t.to(self.device)

        return result

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids [B, T] → hidden [B, T, hidden_size] float32."""
        embed_key = "model.embed_tokens.weight"
        scale_key = "model.embed_tokens.weight_scale_inv"

        keys = [embed_key]
        if scale_key in self._weight_map:
            keys.append(scale_key)

        raw = self._load_tensors(keys)
        embed_w = raw[embed_key]

        if embed_w.dtype == torch.float8_e4m3fn and scale_key in raw:
            embed_w = dequant_fp8_block(embed_w, raw[scale_key]).float()
        else:
            embed_w = embed_w.float()

        # Embedding lookup on CPU, then move
        hidden = F.embedding(token_ids.cpu(), embed_w.cpu())
        return hidden.to(self.device)

    def run_layer(
        self,
        hidden: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run one transformer layer and return (output_hidden, activation_stats).

        Stats keys:
            pre_attn:  mean(|x|) before q/k/v_proj  — shape [hidden_size]
            post_attn: mean(|x|) before o_proj       — shape [n_heads * head_dim]
            pre_mlp:   mean(|x|) before gate/up_proj — shape [hidden_size]
            pre_down:  mean(|x|) before down_proj    — shape [moe_intermediate_size]
        """
        w = self._load_layer(layer_idx)
        B, T, H = hidden.shape

        # ------------------------------------------------------------------
        # 1. Pre-attention RMSNorm + stat
        # ------------------------------------------------------------------
        x_attn = _rms_norm(hidden, w["input_layernorm.weight"], self.rms_norm_eps)
        stat_pre_attn = x_attn.reshape(-1, H).abs().mean(0).cpu()

        # ------------------------------------------------------------------
        # 2. QKV projections
        # ------------------------------------------------------------------
        q = F.linear(x_attn, w["self_attn.q_proj.weight"],
                     w.get("self_attn.q_proj.bias"))   # [B, T, n_heads * head_dim]
        k = F.linear(x_attn, w["self_attn.k_proj.weight"],
                     w.get("self_attn.k_proj.bias"))   # [B, T, n_kv_heads * head_dim]
        v = F.linear(x_attn, w["self_attn.v_proj.weight"],
                     w.get("self_attn.v_proj.bias"))   # [B, T, n_kv_heads * head_dim]

        # Reshape to [B, heads, T, head_dim]
        q = q.reshape(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # ------------------------------------------------------------------
        # 3. QK-norm (Qwen3-specific)
        # ------------------------------------------------------------------
        if "self_attn.q_norm.weight" in w:
            q = _rms_norm(q, w["self_attn.q_norm.weight"], self.rms_norm_eps)
        if "self_attn.k_norm.weight" in w:
            k = _rms_norm(k, w["self_attn.k_norm.weight"], self.rms_norm_eps)

        # ------------------------------------------------------------------
        # 4. RoPE
        # ------------------------------------------------------------------
        cos, sin = _build_rope(T, self.head_dim, self.rope_theta, self.device)
        q, k = _apply_rotary(q, k, cos, sin)

        # ------------------------------------------------------------------
        # 5. GQA: expand k/v to match n_heads
        # ------------------------------------------------------------------
        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        # ------------------------------------------------------------------
        # 6. Scaled dot-product attention (bidirectional — no causal mask)
        # ------------------------------------------------------------------
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # ------------------------------------------------------------------
        # 7. Post-attention reshape + stat
        # ------------------------------------------------------------------
        post_attn = attn_out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        stat_post_attn = post_attn.reshape(-1, self.n_heads * self.head_dim).abs().mean(0).cpu()

        # ------------------------------------------------------------------
        # 8. o_proj + residual
        # ------------------------------------------------------------------
        hidden = hidden + F.linear(post_attn, w["self_attn.o_proj.weight"],
                                   w.get("self_attn.o_proj.bias"))

        # ------------------------------------------------------------------
        # 9. Pre-MLP RMSNorm + stat
        # ------------------------------------------------------------------
        x_mlp = _rms_norm(hidden, w["post_attention_layernorm.weight"], self.rms_norm_eps)
        stat_pre_mlp = x_mlp.reshape(-1, H).abs().mean(0).cpu()

        # ------------------------------------------------------------------
        # 10. MoE routing
        # ------------------------------------------------------------------
        x_flat = x_mlp.reshape(-1, H)  # [N, H]  where N = B*T
        N = x_flat.shape[0]

        router_logits = F.linear(x_flat, w["mlp.gate.weight"])  # [N, n_experts]
        routing_weights = torch.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_indices = torch.topk(
            routing_weights, self.n_experts_per_tok, dim=-1
        )  # both [N, k]
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # ------------------------------------------------------------------
        # 11. Expert dispatch: compute pre_down stats and weighted residual
        # ------------------------------------------------------------------
        pre_down_parts: list[torch.Tensor] = []
        moe_out = torch.zeros_like(x_flat)

        for expert_idx in range(self.n_experts):
            ep = f"mlp.experts.{expert_idx}."
            gate_w = w.get(f"{ep}gate_proj.weight")
            if gate_w is None:
                continue

            up_w = w[f"{ep}up_proj.weight"]
            down_w = w[f"{ep}down_proj.weight"]

            # Tokens that route to this expert
            expert_mask = topk_indices == expert_idx     # [N, k] bool
            token_mask = expert_mask.any(dim=-1)          # [N] bool
            if not token_mask.any():
                continue

            token_indices = token_mask.nonzero(as_tuple=True)[0]  # [n_tok]
            x_e = x_flat[token_indices]                            # [n_tok, H]

            gate_out = F.linear(x_e, gate_w)   # [n_tok, I]
            up_out = F.linear(x_e, up_w)       # [n_tok, I]
            pre_down = F.silu(gate_out) * up_out  # [n_tok, I]

            pre_down_parts.append(pre_down.detach().cpu())

            down_out = F.linear(pre_down, down_w)  # [n_tok, H]

            # Routing weight for this expert per selected token
            expert_weight = (topk_weights * expert_mask.float()).sum(dim=-1)
            expert_weight = expert_weight[token_indices]  # [n_tok]

            moe_out.index_add_(0, token_indices, down_out * expert_weight.unsqueeze(-1))

        if pre_down_parts:
            pre_down_all = torch.cat(pre_down_parts, dim=0)  # [total, I]
            stat_pre_down = pre_down_all.abs().mean(0)
        else:
            stat_pre_down = torch.zeros(self.moe_intermediate_size)

        # ------------------------------------------------------------------
        # 12. MoE residual
        # ------------------------------------------------------------------
        hidden = hidden + moe_out.reshape(B, T, H)

        del w
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return hidden, {
            "pre_attn": stat_pre_attn,
            "post_attn": stat_post_attn,
            "pre_mlp": stat_pre_mlp,
            "pre_down": stat_pre_down,
        }
