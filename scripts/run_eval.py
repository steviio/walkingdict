"""Run the 3-layer retrieval pipeline over a gold JSONL and log per-query results.

Usage:
    python -m scripts.run_eval \
        --gold data/eval/gold.jsonl \
        --out  data/eval/eval_results/eval_results.jsonl \
        [--with-llm]   # also call the generator and record the response

Output rows (one per gold query):
    {
      "query", "bucket", "expected_word", "expected_source",
      "method",                     # exact | fuzzy | vector | llm_only
      "correction",                 # {original, corrected, was_corrected} | null
      "exact_words":   [str, ...],  # words returned by the exact/fuzzy layer
      "similar_words": [str, ...],  # words returned by the vector layer
      "all_ranked":    [str, ...],  # combined ranked list: exact first, then similar
      "low_confidence": bool,
      "rank_expected": int | null,  # 1-indexed position of expected_word in all_ranked
      "hit_at_1/3/5/10": bool,
      "latencies_ms": {"exact_fuzzy", "vector", "total"},
      "response": str | null        # populated iff --with-llm
    }
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.query_pipeline import QueryPipeline


def _words_from_chunks(chunks: list[dict]) -> list[str]:
    seen, out = set(), []
    for c in chunks:
        w = (c.get("word") or "").lower()
        if w and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def _rank_of(target: str, ranked: list[str]) -> int | None:
    target = target.lower()
    for i, w in enumerate(ranked, start=1):
        if w == target:
            return i
    return None


def run(gold_path: Path, out_path: Path, with_llm: bool) -> None:
    pipeline = QueryPipeline()
    pipeline.load()

    generator = None
    if with_llm:
        from rag.generator import Generator  # lazy import — keeps retrieval-only runs fast
        from rag.prompt_builder import PromptBuilder, UserProfile
        generator = (Generator(), PromptBuilder(), UserProfile(proficiency="intermediate", interests=[]))

    gold = [json.loads(l) for l in gold_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[eval] loaded {len(gold)} gold queries from {gold_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(gold, 1):
            t0 = time.perf_counter()
            result = pipeline.query(row["query"])
            t1 = time.perf_counter()

            exact_words = _words_from_chunks(result.chunks)
            similar_words = _words_from_chunks(result.similar_chunks)
            all_ranked = exact_words + [w for w in similar_words if w not in exact_words]

            rank = _rank_of(row["expected_word"], all_ranked)

            rec = {
                **row,
                "method": result.lookup_method,
                "correction": asdict(result.correction) if result.correction else None,
                "exact_words": exact_words,
                "similar_words": similar_words,
                "all_ranked": all_ranked,
                "low_confidence": result.low_confidence,
                "rank_expected": rank,
                "hit_at_1": rank is not None and rank <= 1,
                "hit_at_3": rank is not None and rank <= 3,
                "hit_at_5": rank is not None and rank <= 5,
                "hit_at_10": rank is not None and rank <= 10,
                "latencies_ms": {"total": round((t1 - t0) * 1000, 2)},
            }

            if generator is not None:
                gen, builder, profile = generator
                messages = builder.build(
                    query=row["query"],
                    chunks=result.chunks,
                    profile=profile,
                    low_confidence=result.low_confidence,
                    similar_chunks=result.similar_chunks,
                )
                tg0 = time.perf_counter()
                rec["response"] = gen.generate(messages)
                rec["latencies_ms"]["llm"] = round((time.perf_counter() - tg0) * 1000, 2)

            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 10 == 0 or i == len(gold):
                print(f"  [{i}/{len(gold)}] {row['query'][:40]!r:44}  method={result.lookup_method:<9} rank={rank}")

    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gold", default="data/eval/gold.jsonl", type=Path)
    p.add_argument("--out",  default="data/eval/eval_results/eval_results.jsonl", type=Path)
    p.add_argument("--with-llm", action="store_true",
                   help="Also call the LLM generator and record the response (slow).")
    args = p.parse_args()
    run(args.gold, args.out, args.with_llm)
