# NimbusAI — GPU Cost Optimization: Bài viết ngắn

**Tác giả:** Vũ Hải Nam · Lab 25 — Track 2 · 2026-08-27

---

## 1. Baseline vs. Optimized

| | Baseline | Optimized | Tiết kiệm |
|---|---|---|---|
| **Chi phí tháng (inference + purchasing)** | $27,133 | $14,626 | **$12,507 (46%)** |
| **Inference $/1M-token** | $6.488 | $1.126 | **82.6%** |

`$/GPU-giờ` sẽ không lộ ra vấn đề gì cả — công ty vẫn trả đúng giá niêm yết cho từng GPU-giờ. Nhưng `$/1M-token` giảm gần 6 lần (từ $6.49 xuống $1.13) cho thấy phần lớn hóa đơn GPU trước đây không mua thêm được token nào, chỉ mua sự lãng phí. Đây chính là lý do NimbusAI phải đo bằng đơn vị output (token) chứ không phải đơn vị input (giờ thuê).

## 2. Phân tích từng đòn bẩy

| Đòn bẩy | Tiết kiệm/tháng | Đóng góp |
|---|---|---|
| Purchasing (spot/reserved) | $10,040 | **80.3%** |
| Inference (cascade/cache/batch) | $1,212 | 9.7% |
| Right-size util-lies | $655 | 5.2% |
| Kill idle GPUs | $600 | 4.8% |

**Purchasing đóng góp lớn nhất (80.3% tổng savings)**, không phải vì nó "thông minh" hơn các đòn bẩy khác, mà vì nó tác động lên **toàn bộ chi phí GPU-giờ nền tảng** ($25,667/tháng), trong khi inference-levers chỉ tác động lên phần chi phí token tương đối nhỏ ($1,466/tháng baseline daily × 30). Đây là bài học quan trọng cho ưu tiên: **sửa đúng cấu trúc mua hàng (spot cho job gián đoạn, reserved cho job duty cao) mang lại ROI cao nhất vì nó là đòn bẩy có "bề mặt" lớn nhất** — áp dụng một lần cho toàn bộ workload, không cần tối ưu logic ứng dụng.

Ngược lại, cascade + cache + batch tuy chỉ đóng góp 9.7% về số tuyệt đối ở M2, nhưng **tự thân nó giảm 82.6% chi phí trên đúng phần nó kiểm soát** ($/1M-token). Nó bị "lu mờ" trong bảng tổng hợp M5 chỉ vì baseline inference vốn đã nhỏ hơn baseline purchasing nhiều lần — không phải vì đòn bẩy này yếu. Nếu NimbusAI scale traffic lên 10x mà giữ nguyên fleet GPU, thứ tự ưu tiên sẽ đảo ngược ngay.

**Đề xuất thứ tự triển khai** (theo ROI thực hiện / độ khó):
1. **Purchasing trước** — chỉ cần đổi hợp đồng mua (spot/reserved), không đụng code, thu về $10,040/tháng gần như ngay lập tức.
2. **Cascade + cache + batch** — cần sửa routing logic, nhưng return on engineering effort rất cao (82.6% trên phần nó kiểm soát) và sẽ nhân lên khi traffic tăng.
3. **Right-size util-lies + kill idle** — dọn dẹp vận hành, ROI thấp hơn về số tuyệt đối ở quy mô hiện tại nhưng gần như miễn phí để thực hiện (chỉ là thay đổi cấu hình instance).

## 3. GPU-Util Lie

**GPU bị lie:** `gpu-h100-4` (Util 98.2%, MFU chỉ 0.194) và `gpu-a10g-1` (Util 96.9%, MFU 0.268).

`nvidia-smi` đo "GPU-Util" bằng cách hỏi: *trong 1 giây, có ít nhất một kernel nào đang chạy trên GPU không?* Đó là một đồng hồ đo **thời gian bận**, không phải đồng hồ đo **công việc hữu ích**. Một GPU có thể "bận" 98% thời gian nhưng phần lớn thời gian đó dùng để **chờ dữ liệu từ bộ nhớ HBM** (memory stall) hoặc **chờ CPU phát lệnh kernel tiếp theo** (kernel launch overhead) — cả hai đều khiến đơn vị tính toán (Tensor Core) đứng im trong khi đồng hồ util vẫn tích "đang hoạt động".

`gpu-h100-4` có MFU 0.194 nghĩa là NimbusAI chỉ nhận được **~19.4% FLOPs** so với những gì họ trả tiền cho cả giờ H100. Về mặt tài chính, đây tương đương với việc trả $2.50/giờ cho H100 nhưng chỉ nhận hiệu năng tính toán tương đương ~$0.49/giờ thực dùng — **80.6% tiền GPU-giờ đó biến mất vào memory stall, không tạo ra token nào**. Đây chính là lý do M1 Extension 2 đề xuất downgrade 6 GPU H100 memory-bound này xuống A100 ($511/tháng mỗi GPU): nếu workload vốn dĩ bị nghẽn ở băng thông bộ nhớ chứ không phải FLOPs, thuê một GPU có FLOPs cao hơn (H100) không giải quyết được gì — tiền đó lẽ ra nên chi cho băng thông, không phải FLOPs.

## 4. Phần mở rộng đã thực hiện (5/5)

### Extension 1 — Cải thiện `recommend_tier()`
Thêm interruption rate riêng theo GPU type (H100/B200 ~2-3%, A10G/L4 ~9-10%) và so sánh reserved 1yr vs 3yr theo `job_days` thực tế thay vì mặc định luôn chọn 3yr.
**Kết quả:** savings giữ nguyên 39.1% ở cả v1 và v2 — **không phải vì policy mới vô dụng**, mà vì dataset hiện tại không có GPU nào vượt ngưỡng rủi ro interrupt (12%) và mọi job đủ điều kiện reserved đều chạy đủ dài (≥14 ngày, duty ≥75%) để 3yr vẫn là lựa chọn đúng. **Insight:** chính sách cũ (v1) "tình cờ đúng" trên dữ liệu này, nhưng không có căn cứ định lượng — v2 sẽ tự động rẽ nhánh khác (về spot hoặc reserved_1yr) ngay khi công ty thêm GPU rủi ro cao hơn hoặc job ngắn hạn/bursty, còn v1 sẽ tiếp tục sai một cách âm thầm.

### Extension 2 — Right-sizing theo MBU
Tính `$/GB-VRAM` cho 7 GPU type, đề xuất downgrade cho GPU có MBU < 60% (target khỏe mạnh) nếu có GPU rẻ hơn đủ băng thông đo được (+15% headroom an toàn).
**Kết quả:** 9/11 GPU đủ điều kiện downgrade, tổng **$3,924/tháng tiết kiệm** (giả định 24/7). Riêng `gpu-a100-0` (MBU 0.276) **không** được đề xuất — băng thông đo được (0.55 TB/s × 1.15) vượt quá cả A10G (0.6) lẫn L4 (0.3), nên hạ cấp sẽ chỉ đẩy nghẽn cổ chai xuống thấp hơn. Đây là câu trả lời trực tiếp cho câu hỏi "tại sao không chỉ chọn GPU rẻ nhất theo `$/GPU-hr`": vì làm vậy có thể phá vỡ SLA latency mà không tiết kiệm được gì thực chất.

### Extension 3 — `cache_is_worth_it()`
Tính break-even số lần đọc lại cần thiết để cache có lợi: `write_cost / (price_in × (1 - read_discount))`.
**Kết quả:** break-even chỉ ~1.4 lần đọc (cả 2 tier small/large), trong khi dữ liệu thực tế trung bình **300 lần đọc lại/prefix** (8 cặp team-project chia nhau 2,400 request) — vượt xa 215 lần so với ngưỡng cần thiết. Cache "WORTH IT" gần như tuyệt đối trong bối cảnh NimbusAI, vì traffic của họ rất tập trung (ít prefix, nhiều request lặp lại) — mô hình lý tưởng cho prompt caching.

### Extension 4 — Ngân sách Reasoning
Tách riêng $ và Wh cho `is_reasoning=1` vs `0`, mô phỏng routing rule cap traffic reasoning.
**Kết quả:** reasoning chỉ chiếm **8.4% số request** nhưng ngốn **94.0% tổng năng lượng** — vì mỗi request reasoning vừa lớn hơn (nhiều token hơn), vừa nhân năng lượng lên 80× so với truy vấn thường. Cap ở 10% (mức mặc định) không đổi gì vì traffic hiện tại đã dưới ngưỡng; cap chặt hơn xuống 5% sẽ demote 81 request lớn nhất, tiết kiệm **7,612 Wh/ngày (~25.6% tổng Wh)** nhưng chỉ **$0.24/ngày** — vì giá `$` phụ thuộc `route_tier`, không phụ thuộc cờ `is_reasoning`. **Insight quan trọng:** tối ưu chi phí (`$`) và tối ưu năng lượng (`Wh`/carbon) là **hai trục độc lập** — một hành động có thể gần như không ảnh hưởng hóa đơn nhưng cắt giảm carbon footprint rất lớn.

### Extension 5 — Carbon-aware Scheduling
So sánh 5 vùng triển khai (`gCO2/kWh`, `$/kWh`) cho 5 job `interruptible=1`.
**Kết quả:** chuyển toàn bộ job gián đoạn từ `us-east-1` sang `europe-north1` (sạch nhất, 30 gCO2/kWh) tiết kiệm **626.15 kgCO2e/tháng** (giảm 92.1% carbon mỗi job) **và** giảm thêm **$53.67** chi phí điện — không phải đánh đổi trong trường hợp này. Vùng "balanced" (chuẩn hóa cả 2 trục) lại là `us-east-wa`, không phải `europe-north1` — cho thấy "sạch nhất" và "cân bằng nhất" là hai khái niệm khác nhau. Trade-off thực sự không nằm ở $/carbon mà ở **latency**: chỉ job `interruptible=1` (training/batch, không có người dùng chờ phản hồi) mới hợp lý để dịch chuyển vùng xa; dịch vụ inference 24/7 phục vụ người dùng trực tiếp không được xét vì sẽ cộng thêm độ trễ round-trip xuyên lục địa.

## 5. Khuyến nghị cho NimbusAI (3 hành động đầu tiên)

1. **Chuyển ngay các job đủ điều kiện sang spot/reserved theo M3** — ROI cao nhất ($10,040/tháng), không cần thay đổi code, chỉ là quyết định mua hàng. Ưu tiên tuyệt đối vì đây là 80% tổng savings tiềm năng.
2. **Bật cascade + prompt caching + batch API cho toàn bộ traffic inference** — cache đã được xác nhận có lợi về kinh tế (Extension 3, vượt break-even 215 lần), và đây là đòn bẩy sẽ mở rộng theo scale traffic, không giới hạn bởi fleet GPU hiện tại như purchasing.
3. **Điều tra và downgrade các GPU bị "Util lie"** (`gpu-h100-4`, `gpu-a10g-1`, và 7 GPU memory-bound khác từ Extension 2) — đây là dấu hiệu workload đang bị nghẽn bộ nhớ chứ không phải thiếu FLOPs; tiếp tục thuê GPU cao cấp hơn cho các job này là đốt tiền vô ích, trong khi downgrade mang lại thêm $3,924/tháng gần như miễn phí về mặt kỹ thuật.

*(Carbon-aware scheduling và reasoning budget cap là các hành động "quick win" bổ sung — chi phí triển khai thấp, tác động tốt lên carbon footprint, nên đưa vào roadmap ngay sau 3 hành động chính ở trên.)*
