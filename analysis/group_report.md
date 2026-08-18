# Group Report — Lab 18: Production RAG

**Hình thức:** Cá nhân  
**Thành viên:** Lương Quốc Khánh  

## Thành viên & Phân công

| Tên | Module | Hoàn thành |
|---|---|---|
| Lương Quốc Khánh | M1–M5 + pipeline + analysis | ✅ |

## Kết quả RAGAS

| Metric | Naive Baseline | Production | Δ |
|---|---:|---:|---:|
| Faithfulness | 0.7917 | 0.9083 | +0.1167 |
| Answer Relevancy | 0.7199 | 0.8254 | +0.1055 |
| Context Precision | 0.9250 | 0.9167 | -0.0083 |
| Context Recall | 0.9250 | 0.9500 | +0.0250 |

## Key Findings

1. **Biggest improvement:** `faithfulness` với Δ = +0.1167.
2. **Biggest challenge:** `answer_relevancy` là metric production thấp nhất (0.8254); dùng Error Tree trong failure analysis để chọn đúng layer cần sửa.
3. **Version conflict:** corpus có policy cũ/mới, vì vậy pipeline giữ `source` + `parent_id`, retrieve child nhưng trả parent evidence; prompt yêu cầu ưu tiên phiên bản mới/đang hiệu lực nếu context nêu rõ.
4. **Cost control:** M5 dùng combined enrichment một call/chunk; khi API lỗi/không có key thì fallback local và vẫn giữ nguyên `original_text`/source metadata.

## Latency Breakdown

| Stage | Avg (ms) | Min (ms) | Max (ms) | n |
|---|---:|---:|---:|---:|
| retrieval | 526.424 | 139.279 | 1457.434 | 20 |
| reranking | 4775.374 | 4042.133 | 6321.902 | 20 |
| generation | 9666.898 | 1006.679 | 44486.764 | 20 |

## Presentation Notes (5 phút)

1. Baseline: paragraph chunking + dense-only; production: hierarchical + enrichment + BM25/dense RRF + rerank.
2. Biggest win theo số liệu: `faithfulness` (+0.1167).
3. Case study: dùng failure #1 trong `analysis/failure_analysis.md` và đi theo Error Tree.
4. Next optimization: bắt đầu từ `answer_relevancy` vì đây là metric yếu nhất hiện tại.
