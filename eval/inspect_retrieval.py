"""
Show exactly what the retriever returns for a question.

Run from the project root:
    python eval/inspect_retrieval.py
    python eval/inspect_retrieval.py "your question" --expect "text that must be in a chunk"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.rag.vectorstore import get_retriever  # noqa: E402

DEFAULTS = [
    ("What is the minimum password length?", "14 characters"),
    ("How many days in advance should I apply for planned annual leave?", "5 working days"),
]


def show(query, expect):
    docs = get_retriever().invoke(query)
    print("=" * 70)
    print(f"QUERY : {query}")
    print(f"CHUNKS: {len(docs)}")
    found_at = None
    for i, d in enumerate(docs, 1):
        text = d.page_content.replace("\n", " ")
        hit = bool(expect) and expect.lower() in " ".join(text.lower().split())
        if hit and found_at is None:
            found_at = i
        src = Path(str(d.metadata.get("source", "?")).replace("\\", "/")).name
        print(f"  {i}. {'<<< HAS ANSWER' if hit else '':<15} {src} ({len(d.page_content)} chars)")
        print(f"     {text[:170]}")
    if expect:
        verdict = f"found in chunk {found_at}" if found_at else "NOT retrieved at all"
        print(f"\nEXPECTED TEXT '{expect}': {verdict}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="?")
    p.add_argument("--expect", default="")
    a = p.parse_args()
    for q, e in ([(a.query, a.expect)] if a.query else DEFAULTS):
        show(q, e)
