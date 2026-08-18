from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import json
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu nhưng vẫn giữ nguyên text gốc."""

    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"/"combined"


def _openai_client():
    if not OPENAI_API_KEY:
        return None
    from openai import OpenAI

    return OpenAI(api_key=OPENAI_API_KEY)


def _clean_json_content(content: str) -> dict:
    """Parse JSON returned directly or wrapped in a markdown code fence."""
    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    parsed = json.loads(cleaned)
    return parsed if isinstance(parsed, dict) else {}


def _fallback_summary(text: str) -> str:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text.strip())
        if sentence.strip()
    ]
    if not sentences:
        return text.strip()
    return " ".join(sentences[:2]).strip()


def _fallback_questions(text: str, n_questions: int = 3) -> list[str]:
    candidates = [
        sentence.strip().strip("#>*- ")
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
        if len(sentence.strip().strip("#>*- ")) > 10
    ]
    questions: list[str] = []
    for sentence in candidates:
        sentence = sentence.rstrip(".!? ")
        if not sentence:
            continue
        # Keep enough source vocabulary to bridge lexical gaps while making the
        # fallback unmistakably query-like.
        snippet = " ".join(sentence.split()[:14])
        questions.append(f"Thông tin nào được nêu về: {snippet}?")
        if len(questions) >= n_questions:
            break
    return questions


def _fallback_context(source: str) -> str:
    return f"Trích từ tài liệu {source}." if source else "Đoạn trích từ tài liệu chính sách nội bộ."


def _fallback_metadata(text: str) -> dict:
    lowered = text.lower()
    if any(token in lowered for token in ["mật khẩu", "vpn", "malware", "mfa", "bảo mật"]):
        category = "it"
    elif any(token in lowered for token in ["lương", "chi phí", "tạm ứng", "vnđ", "báo giá"]):
        category = "finance"
    elif any(token in lowered for token in ["nghỉ", "nhân viên", "thâm niên", "thử việc", "mentor"]):
        category = "hr"
    else:
        category = "policy"

    heading = next(
        (
            re.sub(r"^#{1,6}\s*", "", line).strip()
            for line in text.splitlines()
            if re.match(r"^#{1,6}\s+", line.strip())
        ),
        "general",
    )
    entities = re.findall(r"\b\d[\d.,]*(?:\s*(?:VNĐ|ngày|năm|tháng|%))?", text)
    return {
        "topic": heading,
        "entities": entities[:8],
        "category": category,
        "language": "vi",
    }


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """Create a short Vietnamese summary, with a local extractive fallback."""
    client = _openai_client()
    if client is not None:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt.",
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            if content.strip():
                return content.strip()
        except Exception as exc:
            print(f"  ⚠️  OpenAI summarize failed, using local fallback: {exc}")
    return _fallback_summary(text)


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """Generate queries a chunk could answer to bridge vocabulary mismatch."""
    if n_questions <= 0:
        return []
    client = _openai_client()
    if client is not None:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. "
                            "Trả về mỗi câu hỏi trên một dòng."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=220,
                temperature=0,
            )
            content = response.choices[0].message.content or ""
            questions = [
                q.strip().lstrip("0123456789.-) ")
                for q in content.splitlines()
                if q.strip()
            ]
            if questions:
                return questions[:n_questions]
        except Exception as exc:
            print(f"  ⚠️  OpenAI HyQA failed, using local fallback: {exc}")
    return _fallback_questions(text, n_questions=n_questions)


# ─── Technique 3: Contextual Prepend ─────────────────────


def contextual_prepend(text: str, document_title: str = "") -> str:
    """Prepend one sentence that places the chunk inside its source document."""
    client = _openai_client()
    context = ""
    if client is not None:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Viết đúng 1 câu ngắn mô tả đoạn văn nằm ở đâu trong tài liệu "
                            "và nói về chủ đề gì."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}",
                    },
                ],
                max_tokens=90,
                temperature=0,
            )
            context = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            print(f"  ⚠️  OpenAI contextual failed, using local fallback: {exc}")
    if not context:
        context = _fallback_context(document_title)
    return f"{context}\n\n{text}"


# ─── Technique 4: Auto Metadata Extraction ───────────────


def extract_metadata(text: str) -> dict:
    """Extract lightweight topic/entity/category metadata with a safe fallback."""
    client = _openai_client()
    if client is not None:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            'Trích xuất metadata và chỉ trả JSON: '
                            '{"topic":"...","entities":["..."],'
                            '"category":"policy|hr|it|finance","language":"vi|en"}.'
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=170,
                temperature=0,
                response_format={"type": "json_object"},
            )
            parsed = _clean_json_content(response.choices[0].message.content or "{}")
            if parsed:
                return parsed
        except Exception as exc:
            print(f"  ⚠️  OpenAI metadata failed, using local fallback: {exc}")
    return _fallback_metadata(text)


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Get summary + questions + context + metadata in one production API call."""
    client = _openai_client()
    if client is not None:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Phân tích đoạn văn và chỉ trả về JSON có schema: "
                            '{"summary":"tóm tắt 2-3 câu",'
                            '"questions":["câu hỏi 1","câu hỏi 2","câu hỏi 3"],'
                            '"context":"1 câu mô tả vị trí/chủ đề của đoạn trong tài liệu",'
                            '"metadata":{"topic":"...","entities":["..."],'
                            '"category":"policy|hr|it|finance","language":"vi|en"}}.'
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}",
                    },
                ],
                max_tokens=420,
                temperature=0,
                response_format={"type": "json_object"},
            )
            parsed = _clean_json_content(response.choices[0].message.content or "{}")
            if parsed:
                return parsed
        except Exception as exc:
            # Do not fan out to four more API calls after a combined-call failure.
            print(f"  ⚠️  Combined enrichment failed, using local fallback: {exc}")

    return {
        "summary": _fallback_summary(text),
        "questions": _fallback_questions(text, n_questions=3),
        "context": _fallback_context(source),
        "metadata": _fallback_metadata(text),
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """Enrich chunks without ever dropping original text or source metadata."""
    if methods is None:
        methods = ["combined"]
    allowed = {"summary", "hyqa", "contextual", "metadata", "combined"}
    unknown = set(methods) - allowed
    if unknown:
        raise ValueError(f"Unknown enrichment methods: {sorted(unknown)}")

    use_combined = "combined" in methods
    enriched: list[EnrichedChunk] = []

    for index, chunk in enumerate(chunks):
        text = str(chunk["text"])
        original_metadata = dict(chunk.get("metadata", {}))
        source = str(original_metadata.get("source", ""))

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = str(result.get("summary", "") or "")
            questions = [str(q) for q in result.get("questions", []) if str(q).strip()]
            context_line = str(result.get("context", "") or "").strip()
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
            if not isinstance(auto_meta, dict):
                auto_meta = {}
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = (
                generate_hypothesis_questions(text) if "hyqa" in methods else []
            )
            enriched_text = (
                contextual_prepend(text, source) if "contextual" in methods else text
            )
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        # Original metadata wins on collision so enrichment can never overwrite
        # source/version evidence carried from M1.
        merged_metadata = {**auto_meta, **original_metadata}
        enriched.append(
            EnrichedChunk(
                original_text=text,
                enriched_text=enriched_text,
                summary=summary,
                hypothesis_questions=questions,
                auto_metadata=merged_metadata,
                method="+".join(methods),
            )
        )

        if (index + 1) % 10 == 0 or (index + 1) == len(chunks):
            print(f"  Enriched {index + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


if __name__ == "__main__":
    sample = (
        "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. "
        "Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."
    )
    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")
    print(f"Summary: {summarize_chunk(sample)}\n")
    print(f"HyQA questions: {generate_hypothesis_questions(sample)}\n")
    print(f"Contextual: {contextual_prepend(sample, 'Sổ tay nhân viên VinUni 2024')}\n")
    print(f"Auto metadata: {extract_metadata(sample)}")
