# evals

Lightweight quality checks that run against a **served vLLM OpenAI endpoint**
(`http://localhost:8000`), not against weights directly. Used to sanity-check
qstream quantizations (M3, Step-3.7, …).

Shared conventions:

- **`EVAL_MODEL`** env var selects the served model name (default varies per script).
  e.g. `EVAL_MODEL=step3p7-flash`.
- **`EVAL_MAX_TOKENS`** env var caps the completion length. Reasoning models emit
  long chains — too small a budget truncates the answer and *deflates* the score, so
  bump it (and the server's `--max-model-len`) for hard sets.
- **stdlib only** — no `datasets`/`requests`; data is pulled via `urllib` from the
  HF datasets-server JSON API (gated sets need an `Authorization: Bearer <token>`).
- Both reasoning fields are read: `reasoning` (step3p5/step3p7) and
  `reasoning_content` (minimax_m3).

| Script | Set | Notes |
|---|---|---|
| `eval_ppl.py` | fixed English passage | perplexity via `prompt_logprobs` — deterministic faithfulness check |
| `eval_gsm8k.py` | GSM8K (`/mnt/storage/gsm8k_test.jsonl`) | numeric answer match (`16.00 == 16`) |
| `eval_mmlu.py` | MMLU `all` test | subject-spread sampling (strides the subject-ordered split) |
| `eval_gpqa.py` | GPQA Diamond (`/mnt/storage/gpqa_diamond.jsonl`) | 4-choice MC, deterministic option shuffle |

```bash
EVAL_MODEL=step3p7-flash EVAL_MAX_TOKENS=4096 python3 evals/eval_gsm8k.py 1319
EVAL_MODEL=step3p7-flash python3 evals/eval_mmlu.py 300
```

> Note: vLLM with continuous batching + MTP is not bitwise-deterministic at
> `temperature=0` (batch-dependent reductions), so small-sample accuracies are
> noisy (±a few %). Prefer the deterministic `eval_ppl.py` for faithfulness claims.
