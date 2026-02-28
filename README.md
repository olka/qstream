# quant4

File-to-file MXFP4 quantization for large language models. Converts FP8 or FP16/BF16 safetensors models to MXFP4 without loading the full model into GPU memory, enabling quantization of models that don't fit on a single machine.

Targets vLLM's compressed-tensors MXFP4 format.

## Supported models

Tested on:
- **Qwen3-Coder-30B-A3B** (FP8 input, MoE with individual expert weights)
- **Qwen3.5-30B-A3B** (BF16 input, MoE with fused 3D expert tensors)
- **MiniMax-M2.5** (229B, 256 experts, FP8 input)

Any model with `weight_scale_inv` FP8 block scales or plain FP16/BF16 safetensors should work. Both 2D individual expert weights and 3D fused expert tensors (Qwen3.5-style `experts.gate_up_proj`) are handled.

## Install

```bash
pip install -e .
# or without installing:
cd quant4
python scripts/quantize.py --help
```

## Quick start

### Basic quantization

```bash
python scripts/quantize.py \
    --model_dir /path/to/model \
    --output_dir /path/to/output \
    --workers 8
```

### With CUDA acceleration

```bash
python scripts/quantize.py \
    --model_dir /path/to/model \
    --output_dir /path/to/output \
    --workers 8 \
    --device cuda
```

### With streaming activation calibration (better quality)

```bash
# Step 1: collect per-channel activation statistics
python scripts/calibrate.py \
    --model_dir /path/to/model \
    --corpus calibration.txt \
    --output_path stats.json \
    --model_family qwen3 \
    --n_tokens 512

# Step 2: quantize using real activation stats instead of γ proxy
python scripts/quantize.py \
    --model_dir /path/to/model \
    --output_dir /path/to/output \
    --calibration_stats stats.json \
    --workers 8

# Verify stats have 4 keys per layer
python -c "import json; d=json.load(open('stats.json')); print(len(d), list(d['0'].keys()))"
# → 48 ['pre_attn', 'post_attn', 'pre_mlp', 'pre_down']
```

The output directory is a drop-in replacement for the original — same directory structure, `config.json` updated with quantization config. Load with vLLM as normal.

## How it works

### MXFP4 format

Each weight tensor is quantized to MXFP4 E2M1: 4 bits per value, with 8 representable magnitudes `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`. Every 32 consecutive input channels share one e8m0 scale (`2^k` for integer k). Two values are packed per byte.

For FP8 input models, weights are first dequantized using their 128×128 block scales, then re-quantized to MXFP4.

### MSE-optimal scale selection

The naive approach picks the scale exponent by rounding `log2(block_max / 6)`. This minimizes overflow but not quantization error — for heavy-tailed blocks, rounding overestimates the needed scale, compressing precision for the bulk of values.

Instead, quant4 tries three candidate exponents `{floor−1, floor, floor+1}` per block and picks the one minimizing mean squared quantization error:

```
block_max = amax(|w_block|)          # always unweighted
floor     = floor(log2(block_max / 6))

for k in {floor-1, floor, floor+1}:
    dequant = round_mxfp4(w / 2^k) * 2^k
    mse[k]  = mean(γ² · (dequant - w)²)   # γ²=1 if no activation stats

best_k = argmin(mse)
```

All three candidates are evaluated in a single vectorized pass — no per-block Python loops.

**Why block_max uses unweighted amax:** using a γ-weighted percentile for block_max suppresses outlier input channels, pushing the candidate range too low and causing catastrophic scale underselection (0.66 dB worst-block SNR vs 8+ dB with amax). γ is useful for *choosing between* valid candidates, not for determining what the candidates are.

### Overflow correction

After MSE selects the best exponent, any remaining overflow is corrected. Two cases are distinguished:

- **Benign overflow** (`raw_exp == safe_exp − 1`): the selected exponent is one step below safe. At this scale, `block_max / scale` is at most 12×, and saturation at FP4_max=6 gives the same reconstruction as rounding at the next-larger scale. The MSE decision is correct; no correction is applied.
- **Catastrophic overflow** (`raw_exp < safe_exp − 1`): the selected exponent is two or more steps below safe. This happens when γ²-weighted MSE strongly favors smaller scales for typical channels while ignoring an outlier. The exponent is corrected to `safe_exp = ceil(log2(block_max / 6))`.

### Activation-aware scale selection

When activation statistics are available, the MSE candidate evaluation is weighted by per-channel activation magnitude γ:

```
mse_weighted[k] = mean(γ² · (dequant - w)²)
```

This shifts scale selection to minimize error on high-activation channels at the cost of near-zero channels, without affecting which candidate exponents are evaluated. Empirically this reduces median relative error and tightens the error distribution vs. unweighted MSE.

**Two sources for γ, in priority order:**

1. **Calibration stats** (`--calibration_stats stats.json`) — real per-channel `mean(|x|)` measured by running the model forward on calibration text. Separate stats per projection type: `pre_attn`, `post_attn`, `pre_mlp`, `pre_down`. More accurate, requires ~512 tokens of calibration data and a full streaming forward pass.

2. **γ proxy** (default, no calibration needed) — uses `input_layernorm.weight` tensors from the model itself as a stand-in for activation magnitudes. Free, but only covers channels where `in_features == hidden_size` (skips `down_proj`).

### Streaming calibration

The calibration runner loads one transformer layer at a time from disk, runs the forward pass in float32, captures activation statistics, then frees the weights before moving to the next layer.

Supported model families:
- **qwen3** — Qwen3/Qwen3.5 MoE models (softmax routing, standard RoPE)
- **minimax** — MiniMax-M2.5 (sigmoid routing, partial RoPE, 256 experts with shard-streaming)

### Zero-copy mode

For BF16 input models, quant4 symlinks unchanged shards (embedding, attention) and writes only the quantized expert/MLP shards. This reduces I/O significantly when most shards contain only excluded tensors. Enabled automatically when conditions are met; disable with `--no_zero_copy`.

## Options

### `scripts/quantize.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--workers` | 4 | Parallel shard workers |
| `--scale_percentile` | 99.5 | Percentile for unweighted block_max (100 = true amax) |
| `--exclude_layers` | see below | Substring patterns for tensors to skip |
| `--no_activation_aware` | off | Disable γ-weighted MSE, use plain unweighted MSE |
| `--calibration_stats` | none | Path to `stats.json` from `calibrate.py`; overrides γ proxy |
| `--device` | cpu | Device for quantization kernel (`cpu` or `cuda`) |
| `--fp8_block_size` | 128 | FP8 dequantization block size |
| `--no_zero_copy` | off | Force full read/write (no symlinks) |

Default exclude patterns: `*self_attn*`, `*.mlp.gate.`, `*lm_head*`, `*embed_tokens*`

### `scripts/calibrate.py`

| Flag | Default | Description |
|------|---------|-------------|
| `--model_dir` | required | Path to input model |
| `--corpus` | required | Plain text file for calibration |
| `--output_path` | `calibration_stats.json` | Output JSON path |
| `--n_tokens` | 512 | Number of tokens to calibrate on |
| `--model_family` | required | `qwen3` or `minimax` |
| `--device` | `cpu` | Compute device |
| `--expert_buffer` | 32 | Experts loaded at once, MiniMax only. Peak RAM ≈ N × 54 MB |

## Project structure

```
quant4/
├── quant4/
│   ├── core.py         MXFP4 constants, MSE-optimal quantize_mxfp4()
│   ├── fp8.py          FP8 block dequantization
│   ├── gamma.py        input_layernorm.weight loading, layer index extraction
│   ├── handlers.py     tensor handler abstraction (2D standard + 3D fused experts)
│   ├── shard.py        safetensors shard processing, activation_type_from_key()
│   └── calibrate.py    streaming calibration: Qwen3LayerRunner, MiniMaxLayerRunner
└── scripts/
    ├── quantize.py     main quantization CLI
    └── calibrate.py    calibration CLI
```

## vLLM compatibility

Output models use vLLM's compressed-tensors MXFP4 format:
- Weight tensors: `uint8` packed (two 4-bit codes per byte), key suffix `.weight_packed`
- Scale tensors: `uint8` e8m0 biased exponents, key suffix `.weight_scale`
- Config: `compressed-tensors` with `mxfp4-pack-quantized` format

Requires vLLM with MXFP4 support (Blackwell / SM100+ for CUTLASS kernels).
