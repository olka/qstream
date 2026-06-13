"""GSM8K accuracy of the served model (chain-of-thought, extract final number)."""
import json, re, sys, urllib.request, concurrent.futures as cf

MODEL = "/root/.cache/huggingface/M3-MXFP4"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
rows = [json.loads(l) for l in open("/mnt/storage/gsm8k_test.jsonl")][:N]

def gold(ans):  # GSM8K answer after '####'
    return ans.split("####")[-1].strip().replace(",", "")

def last_number(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return nums[-1].rstrip(".") if nums else None

def ask(row):
    body = {"model": MODEL, "messages": [{"role": "user",
            "content": row["question"] + "\nSolve step by step, then give the final numeric answer on its own line."}],
            "max_tokens": 700, "temperature": 0}
    req = urllib.request.Request("http://localhost:8000/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=180))
    m = d["choices"][0]["message"]
    out = (m.get("content") or "") + " " + (m.get("reasoning_content") or "")
    return last_number(out), gold(row["answer"])

correct = 0
with cf.ThreadPoolExecutor(max_workers=16) as ex:
    for i, (pred, g) in enumerate(ex.map(ask, rows)):
        ok = pred is not None and pred == g
        correct += ok
        if i < 6:
            print(f"  [{ '✓' if ok else '✗'}] pred={pred} gold={g}")
print(f"\nGSM8K: {correct}/{len(rows)} = {100*correct/len(rows):.1f}%")
