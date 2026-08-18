from __future__ import annotations

"""Production RAG Pipeline — M1 → M5 → M2 → M3 → answer → M4."""

import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RERANK_TOP_K
from src.m1_chunking import chunk_hierarchical, load_documents
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import evaluate_ragas, failure_analysis, load_test_set, save_report
from src.m5_enrichment import enrich_chunks


_BUILD_LATENCY_MS: dict[str, float] = {}
_QUERY_LATENCY_MS: dict[str, list[float]] = defaultdict(list)


def _record_build_latency(stage: str, started_at: float) -> None:
    _BUILD_LATENCY_MS[stage] = round((time.perf_counter() - started_at) * 1000, 3)


def _record_query_latency(stage: str, started_at: float) -> None:
    _QUERY_LATENCY_MS[stage].append((time.perf_counter() - started_at) * 1000)


def _build_retrieval_text(enriched_chunk) -> str:
    """Use contextual text + summary + HyQA for retrieval, never for final evidence."""
    parts: list[str] = []
    if enriched_chunk.summary:
        parts.append(f"Tóm tắt: {enriched_chunk.summary}")
    if enriched_chunk.hypothesis_questions:
        parts.append(
            "Câu hỏi có thể trả lời: "
            + " | ".join(enriched_chunk.hypothesis_questions)
        )
    parts.append(enriched_chunk.enriched_text)
    return "\n\n".join(part for part in parts if part.strip())


def build_pipeline():
    """Build the production retrieval stack and preserve parent context evidence."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: M1 hierarchical chunking. Children are retrieved; parents remain
    # available as generation context via metadata.
    t0 = time.perf_counter()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks: list[dict] = []
    parent_count = 0
    for doc in docs:
        parents, children = chunk_hierarchical(
            doc["text"], metadata=doc["metadata"]
        )
        parent_lookup = {
            parent.metadata["parent_id"]: parent.text for parent in parents
        }
        parent_count += len(parents)
        for child in children:
            metadata = {
                **child.metadata,
                "parent_id": child.parent_id,
                "parent_text": parent_lookup.get(child.parent_id, child.text),
            }
            all_chunks.append({"text": child.text, "metadata": metadata})
    _record_build_latency("chunking", t0)
    print(
        f"  ✓ {len(all_chunks)} child chunks / {parent_count} parents "
        f"from {len(docs)} documents ({_BUILD_LATENCY_MS['chunking']:.1f} ms)",
        flush=True,
    )

    # Step 2: M5 combined enrichment. The enriched representation is indexed,
    # while original parent_text/source metadata remains the final evidence.
    t0 = time.perf_counter()
    print(
        f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, combined mode)...",
        flush=True,
    )
    enriched = enrich_chunks(all_chunks, methods=["combined"])
    if enriched:
        all_chunks = [
            {
                "text": _build_retrieval_text(item),
                "metadata": item.auto_metadata,
            }
            for item in enriched
        ]
        print(f"  ✓ Enriched {len(enriched)} chunks", flush=True)
    else:
        print("  ⚠️  Enrichment returned no chunks — using raw children", flush=True)
    _record_build_latency("enrichment", t0)

    # Step 3: M2 hybrid BM25 + dense + RRF.
    t0 = time.perf_counter()
    print(
        f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense + RRF)...",
        flush=True,
    )
    search = HybridSearch()
    search.index(all_chunks)
    _record_build_latency("indexing", t0)
    print(f"  ✓ Indexed ({_BUILD_LATENCY_MS['indexing']:.1f} ms)", flush=True)

    # Step 4: M3 model remains lazy-loaded; explicitly load once here so the
    # first query does not hide model startup cost inside query latency.
    t0 = time.perf_counter()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    reranker._load_model()
    _record_build_latency("reranker_load", t0)
    print(
        f"  ✓ Reranker ready ({_BUILD_LATENCY_MS['reranker_load']:.1f} ms)",
        flush=True,
    )

    return search, reranker


def _evidence_context(metadata: dict, fallback_text: str) -> tuple[str, str]:
    """Return (dedupe key, evidence text) with source/version visible to the LLM."""
    source = str(metadata.get("source", "unknown"))
    parent_id = str(metadata.get("parent_id", ""))
    evidence = str(metadata.get("parent_text") or fallback_text)
    header = f"[Nguồn: {source}]"
    return parent_id or f"{source}:{hash(evidence)}", f"{header}\n{evidence}"


def _select_contexts(ranked_results, fallback_results, limit: int = 3) -> list[str]:
    """Deduplicate child hits that point to the same parent before generation."""
    contexts: list[str] = []
    seen: set[str] = set()

    def add(result) -> None:
        key, context = _evidence_context(result.metadata, result.text)
        if key not in seen and len(contexts) < limit:
            seen.add(key)
            contexts.append(context)

    for result in ranked_results:
        add(result)
    if len(contexts) < limit:
        for result in fallback_results:
            add(result)
    return contexts


def run_query(
    query: str,
    search: HybridSearch,
    reranker: CrossEncoderReranker,
) -> tuple[str, list[str]]:
    """Run one query through hybrid retrieval, reranking and grounded generation."""
    t0 = time.perf_counter()
    results = search.search(query)
    _record_query_latency("retrieval", t0)

    documents = [
        {"text": result.text, "score": result.score, "metadata": result.metadata}
        for result in results
    ]
    t0 = time.perf_counter()
    reranked = reranker.rerank(query, documents, top_k=RERANK_TOP_K)
    _record_query_latency("reranking", t0)
    contexts = _select_contexts(reranked, results, limit=RERANK_TOP_K)

    from config import OPENAI_API_KEY

    t0 = time.perf_counter()
    if OPENAI_API_KEY and contexts:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=OPENAI_API_KEY)
            context_str = "\n\n---\n\n".join(contexts)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Bạn trả lời CHỈ dựa trên context được cung cấp. "
                            "Nếu thiếu bằng chứng, nói 'Không tìm thấy.' "
                            "Khi có nhiều phiên bản của cùng một chính sách, ưu tiên phiên bản "
                            "mới/đang hiệu lực được nêu trong nguồn và giải thích ngắn nếu phiên bản cũ đã bị thay thế. "
                            "Không tự suy diễn ngoài context."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}",
                    },
                ],
                temperature=0,
            )
            answer = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            print(f"  ⚠️  LLM generation failed: {exc}", flush=True)
            answer = contexts[0]
    else:
        # Lab contract: missing key must never crash the pipeline.
        answer = contexts[0] if contexts else "Không tìm thấy thông tin."
    _record_query_latency("generation", t0)
    return answer, contexts


def _latency_report() -> dict:
    query_report = {}
    for stage, values in _QUERY_LATENCY_MS.items():
        if values:
            query_report[stage] = {
                "avg_ms": round(sum(values) / len(values), 3),
                "min_ms": round(min(values), 3),
                "max_ms": round(max(values), 3),
                "n": len(values),
            }
    return {"build_ms": dict(_BUILD_LATENCY_MS), "per_query_ms": query_report}


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Evaluate the full stack on the repository test set and persist evidence."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for index, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{index + 1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.perf_counter()
    print(
        f"\n[Eval] Running RAGAS (4 metrics × {len(test_set)} questions)...",
        flush=True,
    )
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    eval_ms = (time.perf_counter() - t0) * 1000
    _BUILD_LATENCY_MS["ragas_eval"] = round(eval_ms, 3)
    print(f"  ✓ RAGAS done ({eval_ms:.1f} ms)", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for metric in [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]:
        score = results.get(metric, 0)
        print(f"  {'✓' if score >= 0.75 else '✗'} {metric}: {score:.4f}")

    failures = failure_analysis(results.get("per_question", []), bottom_n=5)
    save_report(results, failures)
    with open("latency_breakdown.json", "w", encoding="utf-8") as handle:
        json.dump(_latency_report(), handle, ensure_ascii=False, indent=2)
    print("Latency report saved to latency_breakdown.json", flush=True)
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
