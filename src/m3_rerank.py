from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os
import re
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class _LexicalFallbackReranker:
    """Small deterministic fallback used only when the CrossEncoder cannot load."""

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))

    def predict(self, pairs: list[tuple[str, str]]):
        scores = []
        for query, document in pairs:
            q = self._tokens(query)
            d = self._tokens(document)
            if not q or not d:
                scores.append(0.0)
                continue
            overlap = len(q & d)
            scores.append(overlap / max(len(q), 1))
        return scores


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy-load and cache sentence_transformers.CrossEncoder."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            except Exception as exc:
                print(f"  ⚠️  CrossEncoder unavailable, using lexical fallback: {exc}")
                self._model = _LexicalFallbackReranker()
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = RERANK_TOP_K,
    ) -> list[RerankResult]:
        """Read query-document pairs deeply and keep the highest-scoring top-k."""
        if not documents or top_k <= 0:
            return []

        model = self._load_model()
        pairs = [(query, doc["text"]) for doc in documents]
        raw_scores = model.predict(pairs)
        try:
            scores = list(raw_scores)
        except TypeError:
            scores = [raw_scores]

        def to_float(value) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                if hasattr(value, "reshape"):
                    flattened = value.reshape(-1)
                    if len(flattened):
                        return float(flattened[0])
                raise

        scored = sorted(
            ((to_float(score), doc) for score, doc in zip(scores, documents)),
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            RerankResult(
                text=doc["text"],
                original_score=float(doc.get("score", 0.0)),
                rerank_score=score,
                metadata=dict(doc.get("metadata", {})),
                rank=rank,
            )
            for rank, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Optional lightweight reranker."""

    def __init__(self):
        self._model = None

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = RERANK_TOP_K,
    ) -> list[RerankResult]:
        if not documents or top_k <= 0:
            return []
        try:
            from flashrank import Ranker, RerankRequest

            if self._model is None:
                self._model = Ranker()
            passages = [
                {"id": index, "text": doc["text"], "meta": doc.get("metadata", {})}
                for index, doc in enumerate(documents)
            ]
            raw = self._model.rerank(
                RerankRequest(query=query, passages=passages)
            )[:top_k]
            results = []
            for rank, item in enumerate(raw):
                doc = documents[int(item.get("id", rank))]
                results.append(
                    RerankResult(
                        text=doc["text"],
                        original_score=float(doc.get("score", 0.0)),
                        rerank_score=float(item.get("score", 0.0)),
                        metadata=dict(doc.get("metadata", {})),
                        rank=rank,
                    )
                )
            return results
        except Exception as exc:
            print(f"  ⚠️  Flashrank unavailable: {exc}")
            return []


def benchmark_reranker(
    reranker, query: str, documents: list[dict], n_runs: int = 5
) -> dict:
    """Benchmark latency over n_runs."""
    if n_runs <= 0:
        return {"avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for result in reranker.rerank(query, docs):
        print(f"[{result.rank}] {result.rerank_score:.4f} | {result.text}")
