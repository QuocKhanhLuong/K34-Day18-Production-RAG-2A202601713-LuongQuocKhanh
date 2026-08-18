from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    BM25_TOP_K,
    COLLECTION_NAME,
    DENSE_TOP_K,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    HYBRID_TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text; normalize compound-word underscores for BM25."""
    try:
        from underthesea import word_tokenize

        segmented = word_tokenize(text, format="text")
        return segmented.replace("_", " ")
    except Exception as exc:
        # Safe fallback keeps the lexical retriever usable if the optional
        # tokenizer model is temporarily unavailable.
        print(f"  ⚠️  Vietnamese segmentation fallback: {exc}")
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens: list[list[str]] = []
        self.documents: list[dict] = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build a BM25Okapi index while retaining original text + metadata."""
        from rank_bm25 import BM25Okapi

        self.documents = list(chunks)
        self.corpus_tokens = [
            segment_vietnamese(chunk["text"]).lower().split()
            for chunk in self.documents
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search the BM25 index and return only positive-scoring candidates."""
        if self.bm25 is None or not self.documents or top_k <= 0:
            return []

        tokenized_query = segment_vietnamese(query).lower().split()
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores)), key=lambda i: float(scores[i]), reverse=True
        )[:top_k]

        results: list[SearchResult] = []
        for index in top_indices:
            score = float(scores[index])
            if score <= 0:
                continue
            doc = self.documents[index]
            results.append(
                SearchResult(
                    text=doc["text"],
                    score=score,
                    metadata=dict(doc.get("metadata", {})),
                    method="bm25",
                )
            )
        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient

        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Embed chunks with bge-m3 and upsert text + metadata into Qdrant."""
        from qdrant_client.models import Distance, PointStruct, VectorParams

        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        if not chunks:
            return

        texts = [chunk["text"] for chunk in chunks]
        vectors = self._get_encoder().encode(texts, show_progress_bar=True)
        points = []
        for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            payload = {**chunk.get("metadata", {}), "text": chunk["text"]}
            points.append(
                PointStruct(id=index, vector=vector.tolist(), payload=payload)
            )
        self.client.upsert(collection_name=collection, points=points, wait=True)

    def search(
        self,
        query: str,
        top_k: int = DENSE_TOP_K,
        collection: str = COLLECTION_NAME,
    ) -> list[SearchResult]:
        """Query Qdrant using the same bge-m3 embedding space used at index time."""
        if top_k <= 0:
            return []
        query_vector = self._get_encoder().encode(query).tolist()
        response = self.client.query_points(
            collection_name=collection, query=query_vector, limit=top_k
        )

        results: list[SearchResult] = []
        for point in response.points:
            payload = dict(point.payload or {})
            text = str(payload.pop("text", ""))
            if not text:
                continue
            results.append(
                SearchResult(
                    text=text,
                    score=float(point.score),
                    metadata=payload,
                    method="dense",
                )
            )
        return results


def _result_identity(result: SearchResult) -> tuple[str, str, str]:
    """Stable identity so the same source/chunk merges across retrievers."""
    source = str(result.metadata.get("source", ""))
    parent_id = str(result.metadata.get("parent_id", ""))
    return source, parent_id, result.text


def reciprocal_rank_fusion(
    results_list: list[list[SearchResult]],
    k: int = 60,
    top_k: int = HYBRID_TOP_K,
) -> list[SearchResult]:
    """Fuse rankings with RRF instead of mixing incomparable raw scores."""
    if top_k <= 0:
        return []
    if k < 0:
        raise ValueError("k must be non-negative")

    fused: dict[tuple[str, str, str], dict] = {}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            key = _result_identity(result)
            if key not in fused:
                fused[key] = {"score": 0.0, "result": result}
            fused[key]["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)
    return [
        SearchResult(
            text=item["result"].text,
            score=float(item["score"]),
            metadata=dict(item["result"].metadata),
            method="hybrid",
        )
        for item in ranked[:top_k]
    ]


class HybridSearch:
    """Combine Vietnamese BM25 and dense Qdrant retrieval with RRF."""

    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print("Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
