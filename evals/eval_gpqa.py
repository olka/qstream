"""GPQA Diamond accuracy of the served model (4-choice MC, deterministic shuffle)."""
import os, re, json, sys, hashlib, urllib.request, concurrent.futures as cf
MODEL=os.environ.get("EVAL_MODEL","step3p7-flash"); LET=["A","B","C","D"]
MAXTOK=int(os.environ.get("EVAL_MAX_TOKENS","8192"))
rows=[json.loads(l) for l in open("/mnt/storage/gpqa_diamond.jsonl")]
N=int(sys.argv[1]) if len(sys.argv)>1 else len(rows)
rows=rows[:N]

def options(r):
    # deterministic order: sort the 4 answers by md5(question+answer)
    opts=[(r["correct"],True)]+[(x,False) for x in r["incorrect"]]
    opts.sort(key=lambda o: hashlib.md5((r["q"]+o[0]).encode()).hexdigest())
    gold=LET[[i for i,o in enumerate(opts) if o[1]][0]]
    return opts, gold

def extract(t):
    for pat in [r"answer\s*(?:is|:)?\s*\(?([ABCD])\)?", r"\b([ABCD])\b(?!.*\b[ABCD]\b)"]:
        m=re.findall(pat,t,re.IGNORECASE)
        if m: return m[-1].upper()
    return None

def ask(r):
    opts,gold=options(r)
    body={"model":MODEL,"messages":[{"role":"user","content":
        r["q"]+"\n"+"\n".join(f"{LET[i]}. {o[0]}" for i,o in enumerate(opts))+
        "\nThink step by step, then end with 'Answer: X' where X is the letter."}],
        "max_tokens":MAXTOK,"temperature":0}
    req=urllib.request.Request("http://localhost:8000/v1/chat/completions",
        data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    m=json.load(urllib.request.urlopen(req,timeout=600))["choices"][0]["message"]
    content=m.get("content") or ""
    reason=m.get("reasoning") or m.get("reasoning_content") or ""
    return (extract(content) or extract(reason)), gold

correct=0; scored=0
with cf.ThreadPoolExecutor(max_workers=12) as ex:
    for i,(pred,gold) in enumerate(ex.map(ask,rows)):
        scored+=1; correct+= (pred==gold)
        if i<6: print(f"  [{'✓' if pred==gold else '✗'}] pred={pred} gold={gold}")
print(f"\nGPQA-Diamond: {correct}/{scored} = {100*correct/scored:.1f}%")
