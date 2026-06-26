"""MMLU accuracy of the served model, sampled across subjects (no datasets lib).

Pulls rows from the HF datasets-server JSON API. The `all` test split is ordered
by subject, so we stride across the full set to get a subject-spread sample rather
than the (STEM-heavy) front slice. 4-choice MC, letter extraction.

  EVAL_MODEL=step3p7-flash python3 evals/eval_mmlu.py 300
"""
import os, re, json, sys, urllib.request, concurrent.futures as cf

MODEL = os.environ.get("EVAL_MODEL", "step3p7-flash")
MAXTOK = int(os.environ.get("EVAL_MAX_TOKENS", "1536"))
LET = ["A", "B", "C", "D"]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
TOTAL = 14042  # cais/mmlu 'all' test split size

def fetch(n):
    """Strided windows across the full test set → spread over many subjects."""
    win = 10
    n_win = max(1, n // win)
    stride = TOTAL // n_win
    rows = []
    for w in range(n_win):
        url = (f"https://datasets-server.huggingface.co/rows?dataset=cais/mmlu"
               f"&config=all&split=test&offset={w * stride}&length={win}")
        d = json.load(urllib.request.urlopen(url, timeout=60))
        rows += [r["row"] for r in d["rows"]]
    return rows[:n]

def extract(t):
    for pat in [r"answer\s*(?:is|:)?\s*\(?([ABCD])\)?", r"\b([ABCD])\b(?!.*\b[ABCD]\b)"]:
        m = re.findall(pat, t, re.IGNORECASE)
        if m:
            return m[-1].upper()
    return None

def ask(row):
    opts = "\n".join(f"{LET[i]}. {c}" for i, c in enumerate(row["choices"]))
    body = {"model": MODEL, "messages": [{"role": "user", "content":
            f'{row["question"]}\n{opts}\nReply with the single letter of the correct option.'}],
            "max_tokens": MAXTOK, "temperature": 0}
    req = urllib.request.Request("http://localhost:8000/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    m = json.load(urllib.request.urlopen(req, timeout=300))["choices"][0]["message"]
    out = (m.get("content") or "") + " " + (m.get("reasoning") or m.get("reasoning_content") or "")
    return (extract(m.get("content") or "") or extract(out)), LET[row["answer"]]

rows = fetch(N)
print(f"fetched {len(rows)} MMLU questions across {len(rows) // 10} subject windows")
correct = 0
with cf.ThreadPoolExecutor(max_workers=8) as ex:
    for pred, gold in ex.map(ask, rows):
        correct += (pred == gold)
print(f"\nMMLU (subject-spread): {correct}/{len(rows)} = {100 * correct / len(rows):.1f}%")
