"""Generate Lab 18 analysis/reflection from the real reports produced by main.py.

Run after:
    python main.py

Then:
    python scripts/finalize_submission.py
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
ANALYSIS = ROOT / "analysis"
METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
)


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing {path}. Run `python main.py` first.")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _score(report: dict, metric: str) -> float:
    try:
        return float(report.get("aggregate", {}).get(metric, 0.0))
    except (TypeError, ValueError):
        return 0.0


def _metric_table(naive: dict, prod: dict) -> str:
    labels = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "context_precision": "Context Precision",
        "context_recall": "Context Recall",
    }
    rows = [
        "| Metric | Naive Baseline | Production | Δ |",
        "|---|---:|---:|---:|",
    ]
    for metric in METRICS:
        n = _score(naive, metric)
        p = _score(prod, metric)
        rows.append(f"| {labels[metric]} | {n:.4f} | {p:.4f} | {p - n:+.4f} |")
    return "\n".join(rows)


def _short(text: str, limit: int = 700) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_failure_analysis(naive: dict, prod: dict) -> str:
    failures = prod.get("failures", [])[:5]
    blocks = [
        "# Failure Analysis — Lab 18: Production RAG",
        "",
        "**Cá nhân:** Lương Quốc Khánh",
        "",
        "## RAGAS Scores",
        "",
        _metric_table(naive, prod),
        "",
        "## Bottom-5 Failures",
        "",
    ]

    if not failures:
        blocks.extend(
            [
                "> RAGAS không trả per-question results nên chưa có bottom-5. "
                "Kiểm tra OPENAI_API_KEY/dependency rồi chạy lại `python main.py`.",
                "",
            ]
        )
    else:
        for index, failure in enumerate(failures, start=1):
            contexts = failure.get("contexts", [])
            context_excerpt = " || ".join(_short(c, 350) for c in contexts[:3])
            blocks.extend(
                [
                    f"### #{index}",
                    f"- **Question:** {_short(failure.get('question', ''))}",
                    f"- **Expected:** {_short(failure.get('ground_truth', ''))}",
                    f"- **Got:** {_short(failure.get('answer', ''))}",
                    f"- **Context evidence:** {_short(context_excerpt)}",
                    f"- **Worst metric:** {failure.get('worst_metric', '')} = {float(failure.get('score', 0.0)):.4f}",
                    f"- **Error Tree:** {failure.get('error_tree', '')}",
                    f"- **Root cause:** {failure.get('diagnosis', '')}",
                    f"- **Suggested fix:** {failure.get('suggested_fix', '')}",
                    "",
                ]
            )

    if failures:
        case = failures[0]
        blocks.extend(
            [
                "## Case Study",
                "",
                f"**Question:** {_short(case.get('question', ''))}",
                "",
                "**Error Tree walkthrough:**",
                f"1. Output so với ground truth: xem `Got` và `Expected` ở failure #1.",
                f"2. Context đúng/đủ: kiểm tra `Context evidence`; worst metric là `{case.get('worst_metric', '')}`.",
                f"3. Chẩn đoán: {case.get('diagnosis', '')}",
                f"4. Fix có thể kiểm tra lại: {case.get('suggested_fix', '')}",
                "",
                "**Nếu có thêm 1 giờ:** chạy ablation theo đúng worst metric: "
                "M1 boundary/parent-child → M2 BM25+dense/RRF → M3 reranker → prompt, "
                "mỗi lần chỉ đổi một biến rồi chạy lại cùng test set.",
                "",
            ]
        )
    return "\n".join(blocks)


def build_group_report(naive: dict, prod: dict, latency: dict) -> str:
    deltas = {metric: _score(prod, metric) - _score(naive, metric) for metric in METRICS}
    biggest = max(deltas, key=deltas.get)
    weakest = min(METRICS, key=lambda metric: _score(prod, metric))
    query_latency = latency.get("per_query_ms", {})

    blocks = [
        "# Group Report — Lab 18: Production RAG",
        "",
        "**Hình thức:** Cá nhân  ",
        "**Thành viên:** Lương Quốc Khánh  ",
        "",
        "## Thành viên & Phân công",
        "",
        "| Tên | Module | Hoàn thành |",
        "|---|---|---|",
        "| Lương Quốc Khánh | M1–M5 + pipeline + analysis | ✅ |",
        "",
        "## Kết quả RAGAS",
        "",
        _metric_table(naive, prod),
        "",
        "## Key Findings",
        "",
        f"1. **Biggest improvement:** `{biggest}` với Δ = {deltas[biggest]:+.4f}.",
        f"2. **Biggest challenge:** `{weakest}` là metric production thấp nhất ({_score(prod, weakest):.4f}); dùng Error Tree trong failure analysis để chọn đúng layer cần sửa.",
        "3. **Version conflict:** corpus có policy cũ/mới, vì vậy pipeline giữ `source` + `parent_id`, retrieve child nhưng trả parent evidence; prompt yêu cầu ưu tiên phiên bản mới/đang hiệu lực nếu context nêu rõ.",
        "4. **Cost control:** M5 dùng combined enrichment một call/chunk; khi API lỗi/không có key thì fallback local và vẫn giữ nguyên `original_text`/source metadata.",
        "",
        "## Latency Breakdown",
        "",
    ]

    if query_latency:
        blocks.extend(
            [
                "| Stage | Avg (ms) | Min (ms) | Max (ms) | n |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for stage, stats in query_latency.items():
            blocks.append(
                f"| {stage} | {float(stats.get('avg_ms', 0)):.3f} | "
                f"{float(stats.get('min_ms', 0)):.3f} | {float(stats.get('max_ms', 0)):.3f} | "
                f"{int(stats.get('n', 0))} |"
            )
    else:
        blocks.append("Chưa có latency data; chạy lại `python main.py`.")

    blocks.extend(
        [
            "",
            "## Presentation Notes (5 phút)",
            "",
            "1. Baseline: paragraph chunking + dense-only; production: hierarchical + enrichment + BM25/dense RRF + rerank.",
            f"2. Biggest win theo số liệu: `{biggest}` ({deltas[biggest]:+.4f}).",
            "3. Case study: dùng failure #1 trong `analysis/failure_analysis.md` và đi theo Error Tree.",
            f"4. Next optimization: bắt đầu từ `{weakest}` vì đây là metric yếu nhất hiện tại.",
            "",
        ]
    )
    return "\n".join(blocks)


def build_reflection(naive: dict, prod: dict, latency: dict) -> str:
    weakest = min(METRICS, key=lambda metric: _score(prod, metric))
    build_ms = latency.get("build_ms", {})
    return f"""# Individual Reflection — Lab 18

**Tên:** Lương Quốc Khánh  
**Module phụ trách:** M1–M5 (bài cá nhân)

## 1. Mapping bài giảng → code

| Lecture concept | Module | Hàm/class cụ thể | Observation |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()` | Dùng `all-MiniLM-L6-v2`, cosine giữa câu kề nhau và threshold từ config. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Retrieve child nhỏ nhưng giữ `parent_id`/`parent_text` để trả context lớn hơn. |
| BM25 + Dense fusion | M2 | `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | BM25 xử lý keyword/con số; bge-m3 xử lý semantic; RRF gộp rank thay vì cộng raw score. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Giảm candidate về top-{3}; giữ original score, metadata và rank để debug. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Metric production thấp nhất hiện tại: `{weakest}` = {_score(prod, weakest):.4f}. |
| Controlled enrichment | M5 | `_enrich_single_call()`, `enrich_chunks()` | Combined mode 1 call/chunk; fallback không key; source metadata không bị enrichment ghi đè. |

## 2. Kết quả quan sát

{_metric_table(naive, prod)}

- Build chunking: {float(build_ms.get('chunking', 0)):.3f} ms.
- Enrichment: {float(build_ms.get('enrichment', 0)):.3f} ms.
- Indexing: {float(build_ms.get('indexing', 0)):.3f} ms.
- Reranker load: {float(build_ms.get('reranker_load', 0)):.3f} ms.

Điểm quan trọng nhất của lab không phải chỉ là tăng một score tổng, mà là giữ đủ evidence để biết lỗi nằm ở M1, M2, M3 hay generation. Corpus có version cũ/mới làm việc giữ `source` và parent context trở thành một phần của correctness chứ không chỉ là logging.

## 3. Khó khăn & cách giải quyết

- **Version conflict:** nếu chỉ lấy đoạn giống query, policy v2023/v1 có thể cạnh tranh trực tiếp với policy mới. Cách xử lý là không hard-code đáp án; giữ source/version evidence qua toàn pipeline, retrieve child → expand parent và buộc generation ưu tiên phiên bản mới/đang hiệu lực khi context chứng minh điều đó.
- **External dependency:** Qdrant, model Hugging Face và OpenAI/RAGAS đều có thể không sẵn sàng. M1/M3/M4/M5 được viết theo lazy-load/try-except/fallback để lỗi ngoài hệ thống không làm pipeline crash; tuy nhiên report RAGAS hợp lệ vẫn cần API/dependency chạy thật.
- **Score không cùng thang:** BM25 score và cosine score không cộng trực tiếp. M2 dùng RRF theo rank để tránh một retriever lấn át chỉ vì scale khác.

## 4. Action Plan — AI Secretary / MamaGift

### Hiện tại
- Bài toán phù hợp trực tiếp với document QA: ingest tài liệu hành chính/PDF tiếng Việt rồi truy xuất evidence để trả lời.
- Failure nguy hiểm nhất là lấy sai văn bản/phiên bản hoặc trả lời vượt quá evidence.

### Plan áp dụng
1. [ ] **Chunking:** structure-aware cho văn bản hành chính + hierarchical child/parent để giữ điều/khoản và context lớn.
2. [ ] **Search:** hybrid BM25 + dense; BM25 ưu tiên số hiệu, ngày, mã văn bản; dense bắt paraphrase.
3. [ ] **Reranking:** cross-encoder trên top candidate trước khi đưa context vào LLM.
4. [ ] **Evaluation:** RAGAS cho retrieval/generation, cộng custom exact-field checks cho số tiền, deadline, số hiệu văn bản.
5. [ ] **Enrichment:** contextual prepend với tên tài liệu/số hiệu/ngày hiệu lực; không để metadata sinh tự động ghi đè source gốc.

### Timeline
- Tuần 1: benchmark chunking + hybrid retrieval trên bộ câu hỏi thật.
- Tuần 2: reranking + version/effective-date policy + failure taxonomy.
- Tuần 3: RAGAS/custom eval regression set và latency/cost budget.

## 5. Nếu làm lại

Tôi sẽ đo retrieval trước generation: recall@k theo source đúng, sau đó mới rerank và chạy RAGAS. Cách này tách được việc "không tìm thấy evidence" khỏi việc "đã có evidence nhưng model trả lời sai", giúp debug nhanh hơn thay vì chỉ nhìn một điểm cuối.
"""


def main() -> None:
    naive = _load_json(REPORTS / "naive_baseline_report.json")
    prod = _load_json(REPORTS / "ragas_report.json")
    latency_path = REPORTS / "latency_breakdown.json"
    latency = _load_json(latency_path) if latency_path.exists() else {}

    ANALYSIS.mkdir(exist_ok=True)
    reflections = ANALYSIS / "reflections"
    reflections.mkdir(exist_ok=True)

    (ANALYSIS / "failure_analysis.md").write_text(
        build_failure_analysis(naive, prod), encoding="utf-8"
    )
    (ANALYSIS / "group_report.md").write_text(
        build_group_report(naive, prod, latency), encoding="utf-8"
    )
    (reflections / "reflection_LuongQuocKhanh.md").write_text(
        build_reflection(naive, prod, latency), encoding="utf-8"
    )
    print("Generated:")
    print("  analysis/failure_analysis.md")
    print("  analysis/group_report.md")
    print("  analysis/reflections/reflection_LuongQuocKhanh.md")


if __name__ == "__main__":
    main()
