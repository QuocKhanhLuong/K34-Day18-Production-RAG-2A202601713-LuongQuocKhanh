from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import glob
import hashlib
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR,
    HIERARCHICAL_CHILD_SIZE,
    HIERARCHICAL_PARENT_SIZE,
    SEMANTIC_THRESHOLD,
)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh."""
    from pypdf import PdfReader

    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load markdown và PDF có text layer, luôn giữ metadata nguồn."""
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(
                f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, "
                "không có text layer (cần OCR)."
            )
    return docs


# ─── Baseline: Basic Chunking ────────────────────────────


def chunk_basic(
    text: str, chunk_size: int = 500, metadata: dict | None = None
) -> list[Chunk]:
    """Basic paragraph chunking dùng làm baseline."""
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > chunk_size and current:
            chunks.append(
                Chunk(
                    text=current.strip(),
                    metadata={**metadata, "chunk_index": len(chunks)},
                )
            )
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(
            Chunk(
                text=current.strip(),
                metadata={**metadata, "chunk_index": len(chunks)},
            )
        )
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────

_SEMANTIC_MODEL = None
_SEMANTIC_MODEL_FAILED = False


def _semantic_model():
    """Lazy-load và cache all-MiniLM-L6-v2 để không load lại mỗi lần chunk."""
    global _SEMANTIC_MODEL, _SEMANTIC_MODEL_FAILED
    if _SEMANTIC_MODEL is not None:
        return _SEMANTIC_MODEL
    if _SEMANTIC_MODEL_FAILED:
        return None
    try:
        from sentence_transformers import SentenceTransformer

        _SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        return _SEMANTIC_MODEL
    except Exception as exc:
        _SEMANTIC_MODEL_FAILED = True
        print(f"  ⚠️  Semantic model unavailable, using lexical fallback: {exc}")
        return None


def _lexical_similarity(a: str, b: str) -> float:
    tokens_a = set(re.findall(r"\w+", a.lower(), flags=re.UNICODE))
    tokens_b = set(re.findall(r"\w+", b.lower(), flags=re.UNICODE))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def chunk_semantic(
    text: str,
    threshold: float = SEMANTIC_THRESHOLD,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Nhóm các câu kề nhau theo cosine similarity của sentence embedding."""
    metadata = metadata or {}
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+|\n\n+", text)
        if s.strip()
    ]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [
            Chunk(
                text=sentences[0],
                metadata={**metadata, "strategy": "semantic", "chunk_index": 0},
            )
        ]

    model = _semantic_model()
    similarities: list[float] = []
    if model is not None:
        import numpy as np

        embeddings = model.encode(sentences, show_progress_bar=False)
        for i in range(1, len(sentences)):
            a = embeddings[i - 1]
            b = embeddings[i]
            denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
            similarities.append(float(np.dot(a, b) / denom))
    else:
        similarities = [
            _lexical_similarity(sentences[i - 1], sentences[i])
            for i in range(1, len(sentences))
        ]

    groups: list[list[str]] = [[sentences[0]]]
    for sentence, similarity in zip(sentences[1:], similarities):
        if similarity < threshold:
            groups.append([sentence])
        else:
            groups[-1].append(sentence)

    return [
        Chunk(
            text=" ".join(group).strip(),
            metadata={**metadata, "strategy": "semantic", "chunk_index": i},
        )
        for i, group in enumerate(groups)
        if group
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def _split_units_to_limit(units: list[str], limit: int) -> list[str]:
    """Pack text units under a char limit; split oversized units conservatively."""
    if limit <= 0:
        raise ValueError("chunk size must be positive")

    normalized: list[str] = []
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        if len(unit) <= limit:
            normalized.append(unit)
            continue

        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+|\n", unit)
            if s.strip()
        ]
        if len(sentences) == 1 and len(sentences[0]) > limit:
            words = sentences[0].split()
            current_words: list[str] = []
            for word in words:
                candidate = " ".join(current_words + [word])
                if current_words and len(candidate) > limit:
                    normalized.append(" ".join(current_words))
                    current_words = [word]
                else:
                    current_words.append(word)
            if current_words:
                normalized.append(" ".join(current_words))
        else:
            normalized.extend(_split_units_to_limit(sentences, limit))

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for unit in normalized:
        separator_len = 2 if current else 0
        if current and current_len + separator_len + len(unit) > limit:
            chunks.append("\n\n".join(current).strip())
            current = [unit]
            current_len = len(unit)
        else:
            current.append(unit)
            current_len += separator_len + len(unit)
    if current:
        chunks.append("\n\n".join(current).strip())
    return chunks


def chunk_hierarchical(
    text: str,
    parent_size: int = HIERARCHICAL_PARENT_SIZE,
    child_size: int = HIERARCHICAL_CHILD_SIZE,
    metadata: dict | None = None,
) -> tuple[list[Chunk], list[Chunk]]:
    """Create parent chunks for context and smaller linked children for retrieval."""
    metadata = metadata or {}
    if not text.strip():
        return [], []
    if child_size >= parent_size:
        raise ValueError("child_size must be smaller than parent_size")

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    parent_texts = _split_units_to_limit(paragraphs, parent_size)

    source = str(metadata.get("source", "document"))
    source_key = hashlib.sha1(source.encode("utf-8")).hexdigest()[:8]
    parents: list[Chunk] = []
    children: list[Chunk] = []

    for parent_index, parent_text in enumerate(parent_texts):
        parent_id = f"parent_{source_key}_{parent_index:04d}"
        parent_meta = {
            **metadata,
            "strategy": "hierarchical",
            "chunk_type": "parent",
            "chunk_index": parent_index,
            "parent_id": parent_id,
        }
        parents.append(Chunk(text=parent_text, metadata=parent_meta))

        child_units = [p.strip() for p in parent_text.split("\n\n") if p.strip()]
        child_texts = _split_units_to_limit(child_units, child_size)
        for child_index, child_text in enumerate(child_texts):
            child_meta = {
                **metadata,
                "strategy": "hierarchical",
                "chunk_type": "child",
                "chunk_index": child_index,
                "parent_id": parent_id,
            }
            children.append(
                Chunk(text=child_text, metadata=child_meta, parent_id=parent_id)
            )

    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """Split Markdown on level 1-3 headings while preserving each section intact."""
    metadata = metadata or {}
    if not text.strip():
        return []

    heading_re = re.compile(r"^#{1,3}\s+.+$")
    chunks: list[Chunk] = []
    current_heading = ""
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        body = "\n".join(current_lines).strip()
        if not body and not current_heading:
            current_lines = []
            return
        section = (
            re.sub(r"^#{1,3}\s+", "", current_heading).strip()
            if current_heading
            else str(metadata.get("source", "document"))
        )
        chunk_text = "\n\n".join(part for part in [current_heading, body] if part).strip()
        chunks.append(
            Chunk(
                text=chunk_text,
                metadata={
                    **metadata,
                    "strategy": "structure",
                    "section": section,
                    "chunk_index": len(chunks),
                },
            )
        )
        current_lines = []

    for line in text.splitlines():
        if heading_re.match(line.strip()):
            flush()
            current_heading = line.strip()
        else:
            current_lines.append(line)
    flush()

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """Run all strategies and compare chunk-size statistics."""

    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, stats in results.items():
        print(
            f"{name:<15} {stats['count']:>7} {stats['avg_len']:>5} "
            f"{stats['min_len']:>5} {stats['max_len']:>5}"
        )
    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
