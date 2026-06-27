# qstream — TODO

Improvement backlog. Theme: qstream is strong at the quantization math but weak at
**output verification** — the output should be recomputed and validated as authoritative,
never trusted-by-inheritance from the source.

## 1. Fail loud on zero-match / coverage gaps

The quantizer silently produces a wrong-but-plausible output when the selection misses.

- **Zero-match include patterns.** An `--include_layers` pattern that matches **0 tensors**
  currently quantizes nothing and writes a model that is just a BF16 copy — no error, only a
  suspicious output size reveals it.
  - *Seen:* `*mlp.experts.*proj.weight` matched 0 tensors on Ornith-1.0-397B (fused 3D layout
    `mlp.experts.gate_up_proj` has no `.weight` suffix) → `QUANTIZE: 0`.
  - **Fix:** hard-error (or require an explicit `--allow-empty`) when any include pattern, or the
    include set as a whole, matches zero quantizable tensors.

- **Output config coverage.** Validate that the emitted `quantization_config` `targets`/`ignore`
  actually account for **every** `Linear` in the model. A Linear that is neither targeted nor
  ignored loads unquantized and can silently produce zero output (the "merged module not
  targeted" M3 saga).
  - **Fix:** after writing the config, diff its target/ignore coverage against the real module
    list and fail on any uncovered Linear.

- **Optional `--verify` pass.** Sample-SQNR sanity (warn on catastrophically low per-tensor SQNR,
  a sign of a scale/format bug) + config coverage + metadata recompute + a "does it load" smoke
  check — run before anything is uploaded.

## 2. Detect input format from `config.json`, not tensor-name sniffing

`detect_input_format` keys off `.weight_scale_inv` suffixes, so it only recognizes DeepSeek/Qwen
block-FP8 and MXFP8. It misclassifies other FP8 packings as `fp16` and then quantizes the raw
FP8 bytes **without applying the scale** → broken model.

- *Seen:* Ornith-1.0-397B-FP8 ships **compressed-tensors `float-quantized`, per-channel** FP8 —
  scales named `.weight_scale` (BF16, `[out, 1]`), not `.weight_scale_inv`. `detect_input_format`
  returns `fp16` → garbage quant. (Per-channel `fp8-ct` support is being added on branch
  `feat/fp8-compressed-tensors`.)
- **Fix:** read the source `config.json` `quantization_config` (`quant_method`, `format`,
  `strategy`, scale granularity) and dispatch on that, rather than sniffing tensor-name suffixes.
  The FP8 ecosystem already has ≥3 incompatible packings (block `weight_scale_inv`, per-channel
  `float-quantized` `weight_scale`, MXFP8 e8m0); suffix-sniffing will keep missing them.
