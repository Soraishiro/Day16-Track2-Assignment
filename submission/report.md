# Báo cáo Lab 16 — Cloud AI Infrastructure (nhánh AWS)

1. Tôi dùng **AWS**, region **us-east-1**, compute node là **t3.small** (2 vCPU / 2 GB), chạy Ubuntu 22.04, source commit `55539f6`. Hạ tầng dựng bằng Terraform theo starter: VPC 10.0.0.0/16, 2 subnet public + 2 subnet private, NAT Gateway, ALB, Bastion `t3.micro` ở subnet public.

2. Dataset **Credit Card Fraud Detection** có **284.807 dòng × 31 cột**, 0 giá trị thiếu, phân bố lớp cực lệch: 492 gian lận / 284.315 bình thường (tỉ lệ 1:578, tức 0,17%). Tôi chia **60/20/20** (train/validation/test) với `stratify` theo nhãn và **seed 16**.

3. Load dữ liệu mất **2,22 s** (đọc file CSV 144 MB); training mất **3,43 s**; **best iteration = 68** — LightGBM dựng tối đa 300 cây nhưng dừng sớm vì AUC trên tập validation không cải thiện nữa.

4. Kết quả trên tập test: **AUC-ROC 0,9768**, **Accuracy 0,9995**, **F1 0,8541**, **Precision 0,9080**, **Recall 0,8061**. Confusion matrix: TN=56.856, FP=8, FN=19, TP=79 — tức bắt được 79/98 vụ gian lận, bỏ sót 19 vụ, báo động giả 8 vụ. AUC tính trên xác suất, các chỉ số còn lại tính trên nhãn dự đoán tại ngưỡng **0,34** (chọn trên validation theo tiêu chí F1, không chọn trên test).

5. Latency 1 dòng **1,23 ms**, throughput batch 1.000 dòng **324.129 dòng/giây** — đo bằng median sau 5–20 lần warm-up, qua `model.predict_proba` trên DataFrame. Tôi đo thêm bằng `booster_.predict` trên ndarray thuần thì con số là **0,056 ms** và **562.484 dòng/giây**: chênh lệch ~22 lần chính là chi phí pandas + kiểm tra tên cột của sklearn, không phải bản thân LightGBM. Hai số này là hai phép đo khác nhau, không thể trộn. Ảnh: `screenshots/training_results.png`.

6. Lúc benchmark đang chạy, `top` cho thấy tiến trình `python3` dùng **196,9% CPU** (trên 2 vCPU) với RSS 834 MB; `free -h` ghi tổng 1,9 GB RAM, 229 MB đã dùng, 1,4 GB available, không có swap; `ip -s link` trên `ens5` ghi nhận **280.120.850 byte** nhận về và **2.564.941 byte** gửi đi — dung lượng nhận về gần như toàn bộ là tải dataset. Ảnh: `screenshots/resources.png`.

7. Billing tại **02/10/2026 17:23 (GMT+7)** ghi nhận **"No data to display" / "Estimated grand total: USD 0.00"** — tức **chưa cập nhật**, không phải lab miễn phí. AWS cập nhật hóa đơn chậm ít nhất 24 giờ, nên tại thời điểm làm bài dữ liệu chưa có là bình thường. Ước tính riêng của tôi cho khoảng 1 giờ 25 phút tài nguyên tồn tại: NAT Gateway ~0,07 USD (tính cả ~0,28 GB data processing), ALB ~0,04 USD, public IPv4 x2 ~0,02 USD, EBS gp3 ~38 GB ~0,01 USD — **tổng khoảng 0,15 USD**. Hai EC2 (t3.micro + t3.small) nằm trong Free Tier nên không tính phí. Đây là ước tính tính tay, không phải hóa đơn. Ảnh: `screenshots/billing.png`.

8. Tôi đã tải `benchmark.py` + `benchmark_result.json` về máy laptop **trước khi** xoá (JSON nộp là lần chạy lúc `2026-10-02T10:29:14Z`, khớp với ảnh terminal), rồi mới chạy `terraform destroy` lúc **2026-10-02 17:42 (GMT+7)** — kết quả `Destroy complete! Resources: 27 destroyed.` và `terraform state list` trả về **rỗng (0 item)**. Đối chiếu lại bằng AWS CLI sau đó: cả 2 EC2 ở trạng thái `terminated`, NAT Gateway `deleted`, không còn Elastic IP / ALB / target group / key pair / IAM role / EBS volume nào; chỉ còn VPC `vpc-0131d05e0b6cc1283` vốn đã tồn tại từ trước và không thuộc lab. Access key của `ai-lab-user` tôi **giữ lại** vì tài khoản này còn dùng cho các bài sau — nếu không dùng nữa thì vào IAM → Security credentials → xoá key.

---

## Ghi chú: các điểm tôi làm khác starter, và lý do

- **`t3.medium` → `t3.small`.** Gói Free Tier của tài khoản chỉ cho phép 8 loại EC2 và `t3.medium` không nằm trong đó, API từ chối ngay với `InvalidParameterCombination`. `t3.small` là lựa chọn gần nhất: cùng 2 vCPU, RAM 2 GB thay vì 4 GB. Tôi đặt override trong `terraform.tfvars` (đã gitignore) để repo vẫn giữ nguyên giá trị mặc định của starter.
- **Thêm `hour_of_day` thì không giúp gì.** Dữ liệu cho thấy gian lận tập trung lúc 2h sáng (2,05% so với 0,17% trung bình), nhưng khi thêm feature này thì AUC validation giảm 0,9734 → 0,9612 (dạng số) hoặc **trùng khít** với không thêm (dạng categorical, tức model bỏ qua hẳn). Nguyên nhân: mỗi giờ chỉ có trung bình 12 vụ gian lận trong tập train, thấp hơn `min_data_in_leaf=20`, nên không đủ dữ liệu học được ở mức từng dòng. Cuối cùng tôi giữ 30 feature gốc.
- **Không cân bằng lớp.** Thử `scale_pos_weight=577.9`, `is_unbalance=True` và balanced bagging — cả ba đều làm Recall tăng (0,879) nhưng Precision rơi xuống dưới 0,16, do trần Precision của bài toán base rate 0,17% vốn đã rất hạn chế. Mặc định cho kết quả tốt hơn hẳn.
- **`user_data_cpu.sh` giữ nguyên bản gốc.** Tôi từng thêm cờ `--break-system-packages` vì tưởng Ubuntu 22.04 chặn PEP 668, nhưng VM đi kèm pip 22.0.2 không có cờ đó và cũng không có file `EXTERNALLY-MANAGED`, nên bản gốc chạy đúng. Bản GCP cần cờ này vì Debian 12 đi kèm pip 23+.

## Điểm cần lưu ý khi đọc số liệu

Tập test chỉ chứa **98 vụ gian lận**, nên bắt thêm hay bỏ sót 1 vụ đã làm Recall nhảy 1,02 điểm phần trăm. Các con số Recall/Precision ở đây vì thế nhạy với nhiễu, và đó là lý do tôi báo cáo kèm confusion matrix chứ không chỉ dựa vào một con số tổng hợp.

Về ảnh `training_results.png`: tôi chạy script nhiều lần để lấy ảnh `top` đúng lúc training, nên ảnh cho thấy **một lần chạy khác** với file `benchmark_result.json` nộp kèm. Trong hai lần chạy đó toàn bộ chỉ số mô hình **trùng khớp tuyệt đối** (AUC 0,976848, F1 0,854054, Precision 0,908046, Recall 0,806122, best iteration 68) vì tôi đặt `random_state` và `deterministic=true`; chỉ các phép đo thời gian thay đổi vài phần trăm giữa các lần chạy, đó là điển hình của phép đo wall-clock.
