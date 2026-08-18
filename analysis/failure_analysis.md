# Failure Analysis — Lab 18: Production RAG

**Cá nhân:** Lương Quốc Khánh

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|---|---:|---:|---:|
| Faithfulness | 0.7917 | 0.9083 | +0.1167 |
| Answer Relevancy | 0.7199 | 0.8254 | +0.1055 |
| Context Precision | 0.9250 | 0.9167 | -0.0083 |
| Context Recall | 0.9250 | 0.9500 | +0.0250 |

## Bottom-5 Failures

### #1
- **Question:** Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?
- **Expected:** KHÔNG. Nhân viên thử việc chưa được hưởng gói bảo hiểm sức khỏe PVI. Chỉ được tham gia bảo hiểm xã hội bắt buộc.
- **Got:** Nhân viên thử việc chưa được hưởng gói bảo hiểm sức khỏe PVI.
- **Context evidence:** [Nguồn: thu_viec.md] # Chính sách thử việc > Phiên bản: 1.2 | Ngày hiệu lực: 01/01/2024 | Phòng ban: Nhân sự ## Thời gian thử việc Thời gian thử việc tiêu chuẩn là **60 ngày** kể từ ngày bắt đầu làm việc. Đối với vị trí quản lý cấp cao (Manager trở lên), thời gian thử việc có thể kéo dài đến 90 ngày. ## Lương thử việc Nhân viên thử việc được nhận … || [Nguồn: bao_hiem_suc_khoe.md] # Chính sách bảo hiểm sức khỏe > Phiên bản: 1.5 | Ngày hiệu lực: 01/04/2024 | Phòng ban: Nhân sự ## Bảo hiểm cho nhân viên Công ty cung cấp gói bảo hiểm sức khỏe toàn diện qua **PVI Insurance** cho tất cả nhân viên chính thức. Hạn mức bảo hiểm: **200.000.000 VNĐ/năm** bao gồm nội trú, ngoại trú và nha khoa. ## Gói…
- **Worst metric:** answer_relevancy = 0.0000
- **Error Tree:** Answer đúng ground truth? → Context có evidence? → Nếu thiếu: M1/M2/metadata-version; nếu đủ: prompt/generation.
- **Root cause:** Answer chưa trả lời đúng trọng tâm câu hỏi dù có thể đã có context.
- **Suggested fix:** Rút gọn prompt, nhắc model trả lời trực tiếp đúng intent và thêm regression test cho query này.

### #2
- **Question:** Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?
- **Expected:** Theo chính sách v2024 hiện hành, nhân viên có thâm niên từ 3 năm trở lên được cộng thêm 1 ngày phép cho mỗi 3 năm. Chính sách cũ v2023 yêu cầu 5 năm.
- **Got:** Nhân viên có thâm niên từ **3 năm trở lên** được cộng thêm **1 ngày phép** cho mỗi 3 năm làm việc liên tục theo chính sách nghỉ phép năm phiên bản 2024.
- **Context evidence:** [Nguồn: nghi_phep_nam_v2023.md] # Chính sách nghỉ phép năm (Phiên bản 2023) > Phiên bản: 1.0 | Ngày hiệu lực: 01/01/2023 | Phòng ban: Nhân sự ## Số ngày phép năm Mỗi nhân viên chính thức được hưởng **12 ngày phép năm** có lương. Số ngày phép được tính theo năm dương lịch từ 01/01 đến 31/12. Nhân viên vào công ty giữa năm sẽ được tính pro-rata theo… || [Nguồn: nghi_phep_nam_v2024.md] # Chính sách nghỉ phép năm (Phiên bản 2024) > Phiên bản: 2.0 | Ngày hiệu lực: 01/01/2024 | Phòng ban: Nhân sự ## Số ngày phép năm Mỗi nhân viên chính thức được hưởng **15 ngày phép năm** có lương, tăng từ 12 ngày so với chính sách năm 2023. Chính sách này thay thế hoàn toàn phiên bản 1.0 ban hành ngày 01/01/2023…
- **Worst metric:** context_precision = 0.5000
- **Error Tree:** Answer đúng ground truth? → Context có evidence? → Nếu thiếu: M1/M2/metadata-version; nếu đủ: prompt/generation.
- **Root cause:** Retriever đưa quá nhiều chunk không liên quan vào top context.
- **Suggested fix:** Kiểm tra BM25/dense ranks, tăng chất lượng RRF/reranking hoặc thêm metadata/version filter.

### #3
- **Question:** Mật khẩu phải có tối thiểu bao nhiêu ký tự?
- **Expected:** Theo chính sách hiện hành (v2.0), mật khẩu phải có tối thiểu 12 ký tự. Chính sách cũ (v1.0) yêu cầu 8 ký tự nhưng đã bị thay thế.
- **Got:** Mật khẩu phải có tối thiểu **12 ký tự** theo Chính sách mật khẩu v2.0.
- **Context evidence:** [Nguồn: mat_khau_v1.md] # Chính sách mật khẩu (Phiên bản cũ) > Phiên bản: 1.0 | Ngày hiệu lực: 01/01/2022 | Phòng ban: CNTT | Trạng thái: ĐÃ THAY THẾ bởi v2.0 ## Yêu cầu mật khẩu Mật khẩu phải có tối thiểu **8 ký tự**, bao gồm ít nhất 1 chữ hoa, 1 chữ thường và 1 số. Không được sử dụng tên đăng nhập hoặc các thông tin cá nhân dễ đoán làm mật khẩu.… || [Nguồn: mat_khau_v2.md] # Chính sách mật khẩu (Phiên bản hiện hành) > Phiên bản: 2.0 | Ngày hiệu lực: 01/07/2024 | Phòng ban: CNTT ## Yêu cầu mật khẩu Mật khẩu phải có tối thiểu **12 ký tự**, bao gồm ít nhất 1 chữ hoa, 1 chữ thường, 1 số và 1 ký tự đặc biệt (!@#$%^&*). Khuyến khích sử dụng passphrase dài hơn 16 ký tự. ## Xác thực đa yếu tố (MF…
- **Worst metric:** context_precision = 0.5000
- **Error Tree:** Answer đúng ground truth? → Context có evidence? → Nếu thiếu: M1/M2/metadata-version; nếu đủ: prompt/generation.
- **Root cause:** Retriever đưa quá nhiều chunk không liên quan vào top context.
- **Suggested fix:** Kiểm tra BM25/dense ranks, tăng chất lượng RRF/reranking hoặc thêm metadata/version filter.

### #4
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior cao nhất là 20.000.000 VNĐ/tháng. Lương thử việc = 85% x 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got:** Lương thử việc của nhân viên Junior (P1-P2) là 85% của mức lương gross cao nhất là 20.000.000 VNĐ. Tính toán như sau: 20.000.000 VNĐ * 85% = 17.000.000 VNĐ. Vậy, lương thử việc cao nhất của nhân viên Junior là 17.000.000 VNĐ/tháng.
- **Context evidence:** [Nguồn: bang_luong_2024.md] # Bảng lương năm 2024 > Phiên bản: 3.0 | Ngày hiệu lực: 01/01/2024 | Phòng ban: Nhân sự & Tài chính ## Khung lương theo cấp bậc | Cấp bậc | Lương gross (VNĐ/tháng) | |----------|--------------------------| | Junior (P1-P2) | 12.000.000 - 20.000.000 | | Senior (P3-P4) | 20.000.000 - 35.000.000 | | Lead (P5) | 35.000.000 … || [Nguồn: thu_viec.md] # Chính sách thử việc > Phiên bản: 1.2 | Ngày hiệu lực: 01/01/2024 | Phòng ban: Nhân sự ## Thời gian thử việc Thời gian thử việc tiêu chuẩn là **60 ngày** kể từ ngày bắt đầu làm việc. Đối với vị trí quản lý cấp cao (Manager trở lên), thời gian thử việc có thể kéo dài đến 90 ngày. ## Lương thử việc Nhân viên thử việc được n…
- **Worst metric:** faithfulness = 0.5000
- **Error Tree:** Answer đúng ground truth? → Context có evidence? → Nếu thiếu: M1/M2/metadata-version; nếu đủ: prompt/generation.
- **Root cause:** Generation không bám đủ vào evidence trong context.
- **Suggested fix:** Siết prompt 'chỉ dựa trên context', giảm temperature và kiểm tra citation/evidence trước khi trả lời.

### #5
- **Question:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Expected:** Có, theo chính sách mật khẩu v2.0 hiện hành, tất cả nhân viên bắt buộc kích hoạt MFA cho email, VPN và hệ thống nội bộ. Chính sách cũ v1.0 không yêu cầu MFA.
- **Got:** Có, tất cả nhân viên **bắt buộc** phải kích hoạt xác thực đa yếu tố (MFA) cho tài khoản email, VPN và các hệ thống nội bộ.
- **Context evidence:** [Nguồn: mat_khau_v2.md] # Chính sách mật khẩu (Phiên bản hiện hành) > Phiên bản: 2.0 | Ngày hiệu lực: 01/07/2024 | Phòng ban: CNTT ## Yêu cầu mật khẩu Mật khẩu phải có tối thiểu **12 ký tự**, bao gồm ít nhất 1 chữ hoa, 1 chữ thường, 1 số và 1 ký tự đặc biệt (!@#$%^&*). Khuyến khích sử dụng passphrase dài hơn 16 ký tự. ## Xác thực đa yếu tố (MFA) T… || [Nguồn: bao_mat_su_co.md] # Quy trình xử lý sự cố bảo mật > Phiên bản: 1.1 | Ngày hiệu lực: 01/05/2024 | Phòng ban: CNTT & An ninh thông tin ## Báo cáo sự cố Khi phát hiện hoặc nghi ngờ sự cố bảo mật, nhân viên **phải báo cáo trong vòng 1 giờ** qua email **helpdesk@cty.vn** hoặc hotline CNTT nội bộ (ext. 9999). Tuyệt đối **không tự ý xử lý mal…
- **Worst metric:** context_recall = 0.5000
- **Error Tree:** Answer đúng ground truth? → Context có evidence? → Nếu thiếu: M1/M2/metadata-version; nếu đủ: prompt/generation.
- **Root cause:** Context thiếu evidence cần thiết để khôi phục đầy đủ ground truth.
- **Suggested fix:** Kiểm tra M1 boundary/parent-child, M2 retrieval và source/version metadata; bổ sung chunk hoặc retrieval candidate bị thiếu.

## Case Study

**Question:** Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?

**Error Tree walkthrough:**
1. Output so với ground truth: xem `Got` và `Expected` ở failure #1.
2. Context đúng/đủ: kiểm tra `Context evidence`; worst metric là `answer_relevancy`.
3. Chẩn đoán: Answer chưa trả lời đúng trọng tâm câu hỏi dù có thể đã có context.
4. Fix có thể kiểm tra lại: Rút gọn prompt, nhắc model trả lời trực tiếp đúng intent và thêm regression test cho query này.

**Nếu có thêm 1 giờ:** chạy ablation theo đúng worst metric: M1 boundary/parent-child → M2 BM25+dense/RRF → M3 reranker → prompt, mỗi lần chỉ đổi một biến rồi chạy lại cùng test set.
