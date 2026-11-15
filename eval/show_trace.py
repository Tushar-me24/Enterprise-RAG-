"""
Print the step-by-step trace of every failed question from the latest eval run.

    python eval/show_trace.py          # failures only
    python eval/show_trace.py web-01   # one specific question id
"""
import glob
import json
import sys
from pathlib import Path

results_dir = Path(__file__).resolve().parent / "results"
latest = sorted(glob.glob(str(results_dir / "eval_*.json")))[-1]
rows = json.load(open(latest, encoding="utf-8"))
wanted = set(sys.argv[1:])

print(f"Run file: {Path(latest).name}\n")
for r in rows:
    if (wanted and r["id"] in wanted) or (not wanted and not r["passed"]):
        failed = [k for k, v in r["checks"].items() if not v and k != "latency_ok"]
        print(f"{r['id']}  | expected {r['expected_source']} | got {r['source_used']} | failed: {failed}")
        print(f"  Q: {r['question']}")
        for step in r["trace"]:
            print(f"    - {step}")
        print(f"  A: {(r['answer'] or '')[:300].replace(chr(10), ' ')}\n")
