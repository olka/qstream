# FINDINGS — MiniMax-M3 MXFP4 quantization & serving

A record of quantizing **MiniMax-M3** to MXFP4 with qstream and getting it to
serve correctly on the `vllm/vllm-openai:minimax-m3` fork (NVIDIA B300 / SM100,
which is also the Marlin path used by DGX Spark / SM121).

**TL;DR:** qstream's quantization was correct the entire time. The garbled output
was a single **qstream config-generation bug** — it targeted the *separate*
checkpoint projection names while vLLM resolves quant schemes against the *merged*
runtime modules (`qkv_proj`, `gate_up_proj`). Those linears fell back to
unquantized → weights never loaded → zero output. Fixed in `qstream/output.py`.

---

## 1. The model

- **MiniMax-M3-VL** (`model_type: minimax_m3_vl`), ~456B-param vision-language MoE.
- Local copy `/mnt/storage/M3`: 31 shards, 414 GB, shipped as **native MXFP8** —
  `float8_e4m3fn` weights + `weight_scale_inv` that is **uint8 E8M0**, block `[1,32]`
  (dequant = `w · 2^(scale-127)`). No `model.safetensors.index.json`.
- Text backbone: 60 layers, hidden 6144, GQA (4 KV heads), **partial RoPE**,
  per-head QK-norm, **gemma RMSNorm** (`use_gemma_norm`, weights centred near −1 →
  effective scale `1+w`), 128 routed experts (top-4) + **1 shared expert**, sigmoid
  routing with `e_score_correction_bias`, `routed_scaling_factor=2.0`.
- Layers 0–2 are **dense** (`mlp`, full attention); 3–59 are **MoE** + **block-sparse
  "lightning indexer" attention** (the MiniMax MSA library).
- Activation: **SwiGLU-OAI** (gpt-oss style, *uninterleaved* gate-then-up halves):
  `out = (clamp(up,±7)+1) · clamp(gate,max=7) · σ(1.702·gate)`
  (`swiglu_limit=7`, `swiglu_alpha=1.702`; `swiglu_beta` omitted from config →
  fork's config class defaults it to **1.0**).
- Vision tower / projector (~1.7 GB, BF16/F32) are separable; the fork's VL wrapper
  delegates text generation to `MiniMaxM3SparseForCausalLM`.

## 2. What qstream produced

Recipe (quality-preserving mixed): **routed experts → MXFP4**, everything else FP8
(attention, dense MLP, **shared experts**) → **native MXFP8** (lossless passthrough),
BF16/F32 (embeddings, lm_head, router gate, vision, norms) copied.
Invoke: `--include_layers "block_sparse_moe.experts"` (auto-detected `mxfp8` input).

New qstream capabilities added for M3 (all unit-tested):
- `qstream/fp8.py::dequant_mxfp8` (E8M0, `[1,32]`) + uint8 auto-dispatch in `dequant_fp8_block`.
- `qstream/shard.py`: `detect_input_format → "mxfp8"`; MXFP8 passthrough preserving the **uint8** scale (rename `_scale_inv`→`_scale`).
- `qstream/output.py`: `build_index_from_shards` (no index in source) + `build_quantization_config` (mixed MXFP4/MXFP8 config groups, `scale_dtype: torch.uint8`).

## 3. The symptom

The checkpoint loaded and served, but generated garbage — a CJK char then null
bytes / token repetition (`'头顶\x00\x00…'`, `'[View[View…'`). Identical on cutlass
and forced-Marlin → **kernel-independent**.

## 4. What was verified CORRECT (ruled out, numerically, in-container)

| Check | Method | Result |
|---|---|---|
| MXFP4 byte format | dequant our `q_proj` via vLLM's Marlin (`apply_fp4_marlin_linear`) vs qstream's dequant | **55.6 dB** |
| Quant fidelity | MXFP4(ours) vs MXFP8(source) through both load paths | **18.4 dB** (= expected 4-bit error) |
| MoE stacking + GEMM | assemble per-expert `w1/w3/w2`→3D `w13`, `prepare_moe_mxfp4_layer_for_marlin` + `fused_marlin_moe` vs reference | **48.5 dB** |
| SwiGLU-OAI kernel | `apply_moe_activation(SWIGLUOAI_UNINTERLEAVE, …, 7.0, 1.702, 1.0)` vs the exact formula | **0.0015** (vs 0.40 for plain SiLU·up) |
| gemma RMSNorm | per-layer instrumentation | finite (0.04 / 0.47) |
| MXFP8 e8m0 convention | matches vLLM (`descale = 2^(scale-127)`) | ✓ |

Instrumentation isolated the failure: dense layer-0 `after_post_norm=0.47` (finite)
→ **`after_gate_up_proj=0.0`** (exact zero) → activation correctly maps 0→0. So the
**MXFP4 linear itself returned zero from a finite input** — on the `MarlinMxFp4LinearKernel`
that gives 55 dB in isolation. A correct kernel + correct weight cannot give zero.

## 5. Root cause

Dumping `process_weights_after_loading` showed **only the non-merged output
projections** (`o_proj`, `down_proj`) reached the MXFP4 scheme — the **merged input
projections (`qkv_proj`, `gate_up_proj`) never did**.

vLLM fuses `q/k/v[/index_q/index_k] → qkv_proj` and `gate/up → gate_up_proj` at load
and resolves the quant scheme against the **merged** module name. qstream's config
targeted the **separate** names (`re:…self_attn\.q_proj$`, `…mlp\.gate_proj$`), which
the merged names don't match (and M3 registers no fused `packed_modules_mapping` that
would expand them). Scheme resolution proof:

```
q_proj/gate_proj/o_proj/down_proj  -> MXFP4 ✓
qkv_proj                            -> NONE -> UNQUANTIZED -> ZERO
gate_up_proj                        -> NONE -> UNQUANTIZED -> ZERO
```

→ merged linears load as plain BF16, our `weight_packed`/`weight_scale` keys never
bind, the weight stays zero → zero output → frozen residual stream → layer-3
sparse/MoE NaN → token-0 repetition. **NVFP4 checkpoints work because they target
`["Linear"]`/merged names; MXFP8 isn't packed so it's a different story.**

## 6. The fix

`qstream/output.py` now also emits **prefix-scoped merged-module targets** for both
config groups (`_merged_targets()`), mapping `q/k/v/index_q/index_k → qkv_proj` and
`gate/up → gate_up_proj`. Scoped to each module's own prefix so they never reach the
unquantized vision tower. Config-only — **no re-quantization needed** to fix an
existing checkpoint, just regenerate `config.json`.

Verified end-to-end after the fix:
```
"The capital of France is Paris."     "17 plus 25? A: 42"     "5 plus 7? A: 12"
Primary colors: Red, Blue, Yellow ... France is the most visited country in the world
after_gate_up_proj: mean=2.11 (was 0.0);  KV cache 31→40 GiB (weights now actually 4-bit)
```

## 7. Deployment recipe (this fork)

Config with merged targets **+** the MoE clamp patch (forwards swiglu_limit/alpha/beta
to the MoE quant config; fixes `AssertionError: SWIGLUOAI_UNINTERLEAVE requires
clamp_limit`). Patch lives in `M3-MXFP4/vllm_patch/`.

```bash
docker run --gpus all --privileged --ipc=host -p 8000:8000 \
  -e VLLM_MXFP4_USE_MARLIN=1 \
  -v /mnt/storage:/root/.cache/huggingface \
  -v /mnt/storage/M3-MXFP4/vllm_patch/compressed_tensors_moe_w4a4_mxfp4.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a4_mxfp4.py \
  vllm/vllm-openai:minimax-m3 /root/.cache/huggingface/M3-MXFP4 \
  --block-size 128 --tool-call-parser minimax_m3 --enable-auto-tool-choice \
  --reasoning-parser minimax_m3 --load-format fastsafetensors \
  --gpu-memory-utilization 0.97 --enforce-eager --max-model-len 8192 \
  --max-num-batched-tokens 2048 --linear-backend marlin
```

## 8. DGX Spark (SM121) notes

Correctness fix is kernel-independent, so it carries to Spark's Marlin path. Two
*separate* Spark blockers remain, unrelated to the quant:
- **MSA SM12x sparse-attention kernels** (MiniMax-AI/MSA PR #1 `fmha_sm12x`) — not in
  this image; needed for the block-sparse layers (3+) on SM121.
- **Memory**: ~224 GB of weights vs 2×128 GB Spark. TP=2 ≈ 112 GB/node leaves almost
  no KV/activation headroom — practically needs 3–4 Sparks (or a datacenter-Blackwell box).

## 9. Upstream-worthy notes

- vLLM PR #45381 marks MXFP4 on M3 as experimental ("multiple gaps incl. swigluoai
  param passing"); the merged-target gap is a *checkpoint-config* issue, while the
  clamp-param-passing gap (our patch) is genuinely a fork bug. MXFP8 (PR #1724) is the
  officially-supported/benchmarked path.

## 10. Environment gotchas

- venv is the project root itself: `/root/qstream/bin/python`; run `pytest tests/`
  (scoped, else it collects the venv's own site-package tests). `numpy` is a real
  runtime dep (safetensors save).
- In-container probes: write a `.py` file and `docker exec python3 file.py` —
  `python3 - <<HEREDOC` swallowed output. Always `torch.cuda.synchronize()` to surface
  async CUDA errors.
