"""Quick perplexity of the served model via vLLM prompt_logprobs."""
import sys, json, math, urllib.request

MODEL = "/root/.cache/huggingface/M3-MXFP4"
TEXT = (
    "The quick brown fox jumps over the lazy dog. Machine learning models are "
    "trained on large datasets to predict the next token in a sequence. Quantization "
    "reduces the memory footprint of a neural network by representing weights with "
    "fewer bits, trading a small amount of accuracy for substantial savings in size "
    "and bandwidth. The capital of France is Paris, and the Eiffel Tower is one of "
    "the most recognizable landmarks in the world."
)

def post(body):
    req = urllib.request.Request("http://localhost:8000/v1/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))

d = post({"model": MODEL, "prompt": TEXT, "max_tokens": 1,
          "temperature": 0, "prompt_logprobs": 0, "echo": True})
pl = d["choices"][0].get("prompt_logprobs")
if not pl:
    print("no prompt_logprobs returned; keys:", list(d["choices"][0])); sys.exit(1)
lps = []
for entry in pl:
    if entry is None:    # first token has no logprob
        continue
    # entry: {token_id: {logprob, ...}} — take the chosen token's logprob
    lp = min((v["logprob"] for v in entry.values()))  # the realized token
    # actually the realized token is the one present; for prompt_logprobs the
    # dict holds the actual token id -> logprob
    lps.append(list(entry.values())[0]["logprob"])
mean_nll = -sum(lps) / len(lps)
print(f"tokens scored: {len(lps)}")
print(f"mean NLL: {mean_nll:.4f}")
print(f"PERPLEXITY: {math.exp(mean_nll):.3f}")
