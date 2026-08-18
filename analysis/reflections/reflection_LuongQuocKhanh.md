# Individual Reflection — Lab 18

**Tên:** Lương Quốc Khánh  
**Module phụ trách:** M1–M5 (bài cá nhân)

## 1. Mapping bài giảng → code

| Lecture concept | Module | Hàm/class cụ thể | Observation |
|---|---|---|---|
| Semantic chunking | M1 | `chunk_semantic()` | Dùng `all-MiniLM-L6-v2`, cosine giữa câu kề nhau và threshold từ config. |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Retrieve child nhỏ nhưng giữ `parent_id`/`parent_text` để trả context lớn hơn. |
| BM25 + Dense fusion | M2 | `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | BM25 xử lý keyword/con số; bge-m3 xử lý semantic; RRF gộp rank thay vì cộng raw score. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | Giảm candidate về top-3; giữ original score, metadata và rank để debug. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Metric production thấp nhất hiện tại: `answer_relevancy` = 0.8254. |
| Controlled enrichment | M5 | `_enrich_single_call()`, `enrich_chunks()` | Combined mode 1 call/chunk; fallback không key; source metadata không bị enrichment ghi đè. |

## 2. Kết quả quan sát

| Metric | Naive Baseline | Production | Δ |
|---|---:|---:|---:|
| Faithfulness | 0.7917 | 0.9083 | +0.1167 |
| Answer Relevancy | 0.7199 | 0.8254 | +0.1055 |
| Context Precision | 0.9250 | 0.9167 | -0.0083 |
| Context Recall | 0.9250 | 0.9500 | +0.0250 |

- Build chunking: 51.762 ms.
- Enrichment: 767919.474 ms.
- Indexing: 65304.548 ms.
- Reranker load: 50894.076 ms.

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
