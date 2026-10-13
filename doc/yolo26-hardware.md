# YOLO26 — kiểm tra trên máy hiện tại

Đo ngày 13/10/2026, checkpoint chính thức Ultralytics release `v8.4.0`.

Máy: Intel Core i5-12450H (8 core/12 thread), 15,71 GiB RAM, RTX 3050 Laptop 4096 MiB VRAM. Runtime: Python 3.14, PyTorch `2.14.1+cpu`, Ultralytics `8.4.174`. PyTorch hiện không hỗ trợ CUDA, nên kết quả bên dưới là **CPU**; chưa đo GPU.

15 ảnh `demo_data/images`, ảnh đầu 768×432; imgsz 640, confidence .35, 8 CPU thread, xử lý từng ảnh, mỗi ảnh hai lần sau 3 warm-up calls. Mỗi model chạy trong process riêng. Thời gian warm-up/first call không nằm trong mean/p95.

| Model | Mean / ảnh | P95 | Throughput | Peak process RSS | First inference |
|---|---:|---:|---:|---:|---:|
| YOLO26n | 85,40 ms | 99,30 ms | 11,71 ảnh/s | 395,6 MiB | 2,47 s |
| YOLO26s | 165,08 ms | 184,87 ms | 6,06 ảnh/s | 481,7 MiB | 1,41 s |

Peak process RSS gồm Python/model/inference của process thử, không phải tổng RAM toàn ứng dụng. Throughput là inference tuần tự sau warm-up, không bao gồm QC, database, WBF, CVAT hay các model khác. Hai lần cold inference không đủ để so tốc độ khởi động ổn định. Không có golden ground truth trong thử nghiệm này, nên chưa kết luận YOLO26s bắt lỗi tốt hơn n.

**Kết quả:** cả n và s chạy được trong môi trường CPU hiện tại. Chọn n mặc định để ưu tiên tốc độ; s có thể chọn trên Model QC để thử nghiệm recall/precision. GPU vật lý có sẵn nhưng cần runtime PyTorch CUDA để đo khả năng GPU; lượt này không thay bản PyTorch đang dùng.

Báo cáo thô, checkpoint SHA256 và cấu hình: [yolo26-hardware.json](../backend/benchmarks/yolo26-hardware.json).

Chạy lại, không mở hay sửa database AnnoPilot:

```powershell
.\backend\.venv\Scripts\python.exe backend/scripts/benchmark_yolo26.py --device cpu --limit 15 --repeat 2 --threads 8
```

Sau khi có môi trường CUDA, dùng `--device cuda:0` để đo riêng. Nếu CUDA không khả dụng, script báo lỗi thay vì ghi kết quả CPU dưới tên GPU.

Checkpoint local có thể tải bằng:

```powershell
curl.exe --fail --location --output backend/models/yolo26n.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt
curl.exe --fail --location --output backend/models/yolo26s.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt
```

Pipeline dùng `.pt` và output chuẩn của Ultralytics. Không đổi tên prediction/checkpoint YOLOv8 thành YOLO26; các nguồn cũ hoặc biến thể YOLO inactive không được chọn tự động vào ensemble hiện hành.
