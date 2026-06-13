<!--
  qstream MXFP4 model-card template.
  Replace every {{PLACEHOLDER}}. Delete <!-- OPTIONAL --> blocks that don't apply.
  Philosophy: lead with deterministic faithfulness metrics (PPL, SQNR), not noisy
  downstream task scores. State plainly what changed (experts) vs what is
  bit-identical to the source (everything else).
-->
---
license: {{LICENSE}}                      # e.g. apache-2.0  /  other
# license_name: {{LICENSE_NAME}}          # only if license: other
# license_link: LICENSE                    # only if license: other
library_name: transformers
pipeline_tag: {{PIPELINE_TAG}}            # e.g. text-generation / image-text-to-text
base_model: {{BASE_MODEL_HF}}             # e.g. stepfun-ai/Step-3.7-Flash
base_model_relation: quantized
tags:
  - moe
  - mxfp4
  - compressed-tensors
  - quantized
  - vllm
# - multimodal        # keep if the base is a VLM
---

# {{MODEL_NAME}} — MXFP4 (mixed precision)

A 4-bit **MXFP4** quantization of [{{MODEL_NAME}}]({{BASE_MODEL_URL}}), produced with
[**qstream**](https://github.com/olka/qstream). The routed MoE experts (≈95% of the
weights) are quantized to MXFP4; everything quality-sensitive stays {{REST_PRECISION}}.
<!-- REST_PRECISION: "BF16" (step3.7) or "native MXFP8" (M3). -->
<!-- OPTIONAL (only when prepending to the upstream card):
**The original model card follows in full [below](#original-model-card).** -->

| | |
|---|---|
| **Size** | **{{SIZE}}** (down from {{SOURCE_SIZE}} {{SOURCE_FORMAT}} source, ~{{PERCENT}}%) |
| **Format** | compressed-tensors `mixed-precision` ({{FORMAT_DETAIL}}) |
| **Base** | {{BASE_ONE_LINER}} |
<!-- BASE_ONE_LINER: params, MoE shape (N experts top-K + shared), routing,
     layers, context, anything architecturally notable (vision, MTP, …). -->

## What is quantized to what

| Component | Precision | Why |
|---|---|---|
| Routed experts (`{{EXPERT_PATH_GLOB}}`) | **MXFP4** (4-bit) | ~95% of the weights — the only place worth the size win |
| Attention, dense MLP, shared expert, router gate | **{{REST_PRECISION}}** | sensitive / runs on every token — kept lossless from the source |
<!-- OPTIONAL rows: -->
<!-- | MTP / next-token-prediction layers | **{{REST_PRECISION}}** | speculative-decoding draft path — unquantized | -->
| Embeddings, lm_head{{VISION_BIT}}, norms | **{{REST_PRECISION}}** | unchanged |
<!-- VISION_BIT: ", vision encoder, projector" if a VLM, else "". -->

## Quality & faithfulness

We report **deterministic, reproducible** faithfulness metrics rather than downstream
task scores. (Task evals served over vLLM with continuous batching{{MTP_NOTE}} are not
bitwise-deterministic — the same prompt at `temperature=0` can yield different reductions
depending on batch composition — so small-sample accuracies are noisy and we don't quote
them.)
<!-- MTP_NOTE: " + MTP" if speculative decoding is used, else "". -->

| Metric | Result | What it shows |
|---|---|---|
| **Perplexity** (clean English) | **{{PPL}}** | language modeling intact — a broken quant lands in the hundreds |
| **Routed-expert SQNR** | **≈ {{SQNR}} dB** | reconstruction error is just the unavoidable 4-bit rounding (MXFP4 vs the {{SOURCE_FORMAT}} source) |

Why this is enough to trust the checkpoint:

- **Only the routed experts changed.** ~95% of the weights are re-quantized to MXFP4;
  **everything else is bit-identical {{REST_PRECISION}}** to the source. So the model *is*
  the base model except for 4-bit rounding on the expert GEMMs.
- **The math path is verified.** The 2D-linear and 3D-MoE dequant/GEMM paths were checked
  numerically; the only residual is the ~{{SQNR}} dB expert rounding above.

PPL script: `evals/eval_ppl.py` in the qstream repo.

## Fidelity, footprint & provenance

<!-- OPTIONAL (VLM): -->
- **Vision is untouched:** the vision encoder + projector stay **{{REST_PRECISION}}**
  (bit-identical), so image capability equals the base model. Verified working end-to-end.
<!-- OPTIONAL (MTP): -->
- **MTP preserved:** the multi-token-prediction draft layers stay {{REST_PRECISION}}, so
  speculative decoding works (mean acceptance length ≈ {{MTP_ACCEPT}}).
- **Footprint:** ~{{WEIGHTS_GIB}} GiB of weights; fits a single ≥{{MIN_GPU}} GB GPU.
- **Provenance:** built with [qstream](https://github.com/olka/qstream) `@{{COMMIT}}` from
  the `{{SOURCE_RELEASE}}` release; mixed-precision recipe (experts→MXFP4, rest→{{REST_PRECISION}}).

## Serving with vLLM

Targets {{VLLM_IMAGE}}. The `config.json` here targets vLLM's *merged* runtime modules
(`qkv_proj`, `gate_up_proj`) so the fused linears load quantized.

```bash
docker run -d --name {{CONTAINER}} --gpus all --privileged --ipc=host -p 8000:8000 \
  -e VLLM_MXFP4_USE_MARLIN=1 \
  -v $(pwd):/model \
{{PATCH_MOUNTS}} \
  {{VLLM_IMAGE}} /model \
  --served-model-name {{SERVED_NAME}} \
{{SERVE_FLAGS}} \
  --gpu-memory-utilization 0.97 --enforce-eager \
  --linear-backend marlin --trust-remote-code
```

<!-- OPTIONAL (only if the checkpoint needs a runtime patch — see vllm_patch/):
### Runtime patch ([`vllm_patch/`](./vllm_patch))

{{PATCH_REASON_ONE_PARAGRAPH}} See [`vllm_patch/README.md`](./vllm_patch/README.md).
-->

## How it was made

```bash
qstream-quantize \
  --model_dir <{{SOURCE_RELEASE}}> \
  --output_dir ./{{OUTPUT_DIR}} \
  --include_layers "{{EXPERT_INCLUDE}}" \
  --device cuda --workers 8
```

`detect_input_format` auto-detects the source's {{SOURCE_FORMAT}}, dequantizes only the
routed experts and re-quantizes them to MXFP4, passes the {{REST_PRECISION}} remainder
through, and writes the mixed-precision `config.json`.

## License

Inherits the {{LICENSE_BLURB}} from the base model. This is a derivative (quantized) work
of {{MODEL_NAME}}.

<!-- OPTIONAL (when prepending to upstream card):
---

# Original model card

{{PASTE UPSTREAM README HERE}}
-->
