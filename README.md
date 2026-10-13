# AnnoPilot — Model-assisted QC

AnnoPilot phát hiện nghi vấn **sót vật thể** và **sai class**, xếp hạng để reviewer tìm được nhiều lỗi thật với ít công review hơn. CVAT giữ annotation chính; AnnoPilot phân tích bằng chứng và theo dõi QC.

Workflow: **Pre-label → annotator sửa trên CVAT → AnnoPilot QC → reviewer xác nhận/sửa → đồng bộ → nghiệm thu**.

## Hướng triển khai đã chốt

- Baseline: một lần inference YOLO26n/YOLO26s pretrained.
- Pilot: kiểm tra chéo nhiều detector, giữ prediction gốc, bất đồng class và candidate chưa được chọn sau fusion. WBF đề xuất pre-label.
- Đồng thuận không phải ground truth; candidate chưa được chấp thuận không tự trở thành nhãn cuối cùng.
- Queue xử lý từng frame; inference nhiều model cũng commit prediction/QC sau mỗi frame.
- Đo bằng Golden Benchmark độc lập: recall Missing Object/Wrong Class, precision/F1, lỗi thật/giờ review, thời gian inference/QC và số vòng bị trả lại.
- TTA/perturbation, ba detector bắt buộc và Quality Gate khóa export CVAT thuộc bước mở rộng.

## Chạy local trên Windows

Python 3.10+, Node.js 22+. Cài và khởi động API/UI:

```powershell
.\Start-AnnoPilot.ps1 -Install
```

Các lần sau: `.\Start-AnnoPilot.ps1`. Frontend: http://localhost:3000. Swagger: http://localhost:8000/docs.

API nhận prediction JSON mà không cần cài model. Để inference local:

```powershell
.\backend\.venv\Scripts\python.exe -m pip install -r backend/requirements-models.txt
```

YOLO26n là mặc định; chọn YOLO26s trong dropdown YOLO trên Model QC. Lựa chọn áp dụng cho Queue, kiểm tra frame, detector và M1 của ensemble; chỉ chọn một biến thể YOLO mỗi lần. Prediction YOLOv8 cũ được giữ với model ID gốc, không đổi tên thành YOLO26.

Checkpoint chính thức nằm tại `backend/models/yolo26n.pt` và `backend/models/yolo26s.pt`, không tự fallback về YOLOv8/fine-tuned checkpoint. File weights không được commit; máy mới cần tải chúng từ release Ultralytics `v8.4.0`. Triad gồm YOLO26 đã chọn, Faster R-CNN MobileNetV3 và RT-DETR-l; Faster R-CNN có thể tải weights lần đầu.

Đã chạy thật cả hai YOLO26 trên CPU của máy hiện tại: n trung bình **85,4 ms/ảnh**, s **165,08 ms/ảnh**, ở imgsz 640, 15 ảnh demo, hai lượt sau warm-up. Đây là kiểm tra tài nguyên/tốc độ, chưa đo chất lượng detection hay toàn pipeline ensemble. [Báo cáo phần cứng](doc/yolo26-hardware.md).

```powershell
.\backend\.venv\Scripts\python.exe backend/scripts/benchmark_yolo26.py --device cpu
```

## QC và ensemble

Spatial matching không ép cùng class. Mỗi model có tối đa một phiếu/candidate; mọi detection được giữ trong evidence. Class chọn theo tổng `model_weight × confidence`; tọa độ fusion dùng detection bỏ phiếu cho class được chọn.

- Tier 1: ít nhất hai model cùng class, min confidence ≥ .85, average pairwise IoU ≥ .60.
- Tier 2: matched nhưng bất đồng class.
- Tier 3: một model phát hiện hoặc đồng thuận chưa đạt ngưỡng Tier 1.

QC đọc prediction gốc. Bất đồng vẫn tạo Wrong Class candidate dù annotation trùng class mà fusion chọn. Candidate không được chọn vẫn có thể tạo Missing Object candidate.

Ngưỡng mặc định: match IoU .50; missing max IoU .20; missing confidence .40; wrong-class confidence .45; min area 100 px². Đây là heuristic cần kiểm chứng, không phải xác suất lỗi đã calibration.

## Đánh giá và nghiệm thu

Mở **Model QC → Đánh giá Baseline → Pilot & vòng nghiệm thu**:

1. Nạp golden JSON do người kiểm chứng độc lập cung cấp, bao gồm vùng không có warning.
2. Chọn baseline và ít nhất hai nguồn pilot, bao gồm baseline.
3. So sánh trước khi sửa annotation. Hai cấu hình dùng cùng snapshot và golden; evaluator không sửa operational issues.
4. Nhập thời gian review thực tế nếu đã đo. Không có mẫu/thời gian hiển thị `—`.
5. Ghi nhận nộp → trả lại (bắt buộc lý do) → nộp lại → chấp nhận. Số lần audit không phải số vòng trả lại.

[Hướng dẫn JSON, API và metric](doc/huong-dan-su-dung.md) · [Hướng triển khai đã chốt](doc/project-direction.md).

## Kiểm tra

```powershell
Set-Location backend
.\.venv\Scripts\python.exe -m pytest -q
Set-Location ../frontend
npm.cmd run build
```

Test dùng SQLite/uploads/exports tạm. Migration thêm bảng/cột và giữ dữ liệu hiện có khi backend khởi động lại.

## Giới hạn

Chưa có benchmark thật chứng minh ensemble tốt hơn baseline. Tests phần mềm không thay thế đánh giá model. Matching dùng greedy IoU, có thể nhầm trong cảnh đông đúc. COCO pretrained không có rider riêng, chỉ nhận stop sign trong nhóm traffic sign; mapping không tạo thêm năng lực cho model.

Queue state ở RAM, chưa resume bền vững sau restart. Sync CVAT do người dùng kích hoạt; chưa có push tự động từng frame, CVAT issue overlay, Quality Gate khóa export hay TTA. Append tránh box trùng chính xác trong shapes hiện có, chưa có idempotency/concurrency phân tán. Task/Job, frame mapping và taxonomy cần khớp cấu hình.

Tỷ lệ frame không có warning không chứng minh ảnh sạch hoặc công review giảm. Nghiệm thu là quyết định do người dùng ghi nhận.
