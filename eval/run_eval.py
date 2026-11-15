"""
Evaluation harness for the Enterprise RAG agent.

Run from the project root:
    python eval/run_eval.py
    python eval/run_eval.py --category kb
    python eval/run_eval.py --delay 2 --max-latency 15

For each question in eval/questions.jsonl it runs the real agent (app.rag.workflow.ask)
and scores four quality checks plus latency:

  route         - did the agent use the expected path (private_kb / web_search / direct / insufficient_evidence)?
  accuracy      - does the answer contain every required fact (must_include)?
  no_forbidden  - does the answer avoid wrong facts (must_not_include)?
  grounding     - for KB and web answers: is there at least one citation and a Sources list?
                  (this checks citation presence, not that every claim is supported)
  latency       - reported separately; pass = finished within --max-latency seconds

A question PASSES when route, accuracy, no_forbidden and grounding all pass.
Results are saved to eval/results/ as JSON and CSV.
"""
import argparse
import csv
import json
import logging
import re
import sys
import time
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.rag.workflow import ask  # noqa: E402

DATASET = ROOT / "eval" / "questions.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"
QUALITY_CHECKS = ("route", "accuracy", "no_forbidden", "grounding")


def load_cases(path, category=None):
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            case = json.loads(line)
            if category and case["category"] != category:
                continue
            cases.append(case)
    return cases


def normalize(text):
    """Lowercase and strip invisible formatting differences (non-breaking spaces,
    narrow no-break spaces, fancy hyphens) so '10 GB' matches '10\u202fGB'."""
    text = unicodedata.normalize("NFKC", text or "").lower()
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    return re.sub(r"\s+", " ", text)


def contains(answer_norm, term):
    """term is a string, or a list of acceptable alternatives (any one is enough)."""
    options = term if isinstance(term, list) else [term]
    return any(normalize(o) in answer_norm for o in options)


def evaluate(case, out, latency, max_latency):
    answer = out.get("answer", "") or ""
    low = normalize(answer)
    expected = case["expected_source"]
    expected = expected if isinstance(expected, list) else [expected]
    source = out.get("source_used", "")
    citations = out.get("citations") or []

    needs_sources = any(e in ("private_kb", "web_search") for e in expected)

    return {
        "route": source in expected,
        "accuracy": all(contains(low, t) for t in case.get("must_include", [])),
        "no_forbidden": not any(normalize(t) in low for t in case.get("must_not_include", [])),
        "grounding": (len(citations) > 0 and "sources" in low) if needs_sources else True,
        "latency_ok": latency <= max_latency,
    }


def percentile(values, p):
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1))))
    return ordered[idx]


def pct(n, total):
    return f"{(100 * n / total):.0f}%" if total else "n/a"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--category", default=None, help="run only one category, e.g. kb")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds to wait between questions (API rate limits)")
    parser.add_argument("--max-latency", type=float, default=15.0, help="latency threshold in seconds")
    parser.add_argument("--verbose", action="store_true", help="show INFO logs (e.g. API retry / rate-limit messages)")
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    cases = load_cases(args.dataset, args.category)
    if not cases:
        print("No questions found.")
        return

    results = []
    for i, case in enumerate(cases, 1):
        error = None
        start = time.perf_counter()
        try:
            out = ask(case["question"])
        except Exception as e:  # keep going so one failure doesn't stop the run
            out, error = {}, f"{type(e).__name__}: {e}"
        latency = time.perf_counter() - start

        checks = evaluate(case, out, latency, args.max_latency)
        passed = error is None and all(checks[c] for c in QUALITY_CHECKS)

        results.append({
            "id": case["id"],
            "category": case["category"],
            "question": case["question"],
            "verified": case.get("verified", False),
            "expected_source": case["expected_source"],
            "source_used": out.get("source_used", ""),
            "passed": passed,
            "checks": checks,
            "latency_s": round(latency, 2),
            "answer": out.get("answer", ""),
            "citations": out.get("citations", []),
            "trace": out.get("trace", []),
            "error": error,
        })
        print(f"[{i:02d}/{len(cases)}] {'PASS' if passed else 'FAIL'}  {case['id']:<16} {latency:5.1f}s  source={out.get('source_used', '-')}")
        time.sleep(args.delay)

    # ---- summary ----
    total = len(results)
    n_pass = sum(r["passed"] for r in results)
    latencies = [r["latency_s"] for r in results]

    print("\n" + "=" * 60)
    print(f"OVERALL: {n_pass}/{total} passed ({pct(n_pass, total)})")

    print("\nBy check:")
    for c in QUALITY_CHECKS + ("latency_ok",):
        n = sum(r["checks"][c] for r in results)
        label = f"latency <= {args.max_latency:.0f}s" if c == "latency_ok" else c
        print(f"  {label:<18} {n}/{total} ({pct(n, total)})")

    print("\nBy category:")
    groups = defaultdict(list)
    for r in results:
        groups[r["category"]].append(r["passed"])
    for cat, flags in groups.items():
        print(f"  {cat:<14} {sum(flags)}/{len(flags)} ({pct(sum(flags), len(flags))})")

    print(f"\nLatency: p50 {percentile(latencies, 50):.1f}s | p95 {percentile(latencies, 95):.1f}s | max {max(latencies):.1f}s")

    unverified = sum(1 for r in results if not r["verified"])
    if unverified:
        print(f"\nNOTE: {unverified} question(s) have verified=false. Check their expected answers against your KB files before trusting the score.")

    failures = [r for r in results if not r["passed"]]
    if failures:
        print("\nFailures:")
        for r in failures:
            failed = [c for c in QUALITY_CHECKS if not r["checks"][c]]
            reason = r["error"] or ", ".join(failed)
            snippet = (r["answer"] or "").replace("\n", " ")[:160]
            print(f"  - {r['id']}: {reason} | got source={r['source_used'] or '-'} | {snippet}")

    # ---- save ----
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"eval_{stamp}.json"
    csv_path = RESULTS_DIR / f"eval_{stamp}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "category", "question", "passed", "route", "accuracy", "no_forbidden",
                         "grounding", "latency_ok", "latency_s", "expected_source", "source_used", "error"])
        for r in results:
            c = r["checks"]
            writer.writerow([r["id"], r["category"], r["question"], r["passed"], c["route"], c["accuracy"],
                             c["no_forbidden"], c["grounding"], c["latency_ok"], r["latency_s"],
                             r["expected_source"], r["source_used"], r["error"] or ""])

    print(f"\nSaved: {json_path}\n       {csv_path}")


if __name__ == "__main__":
    main()