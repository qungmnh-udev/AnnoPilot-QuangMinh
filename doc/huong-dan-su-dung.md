# Hướng dẫn AnnoPilot — Model-assisted QC

## 1. Workflow

**Pre-label → annotator sửa trên CVAT → AnnoPilot QC → reviewer xác nhận/sửa → đồng bộ → nghiệm thu.** CVAT giữ annotation chính; AnnoPilot lưu mirror và bằng chứng. Chấp nhận gợi ý trên AnnoPilot cập nhật mirror local; cần push và kiểm tra lại trên CVAT trước nghiệm thu.

Baseline: một lần inference YOLO26n/YOLO26s. Pilot: multi-model cross-verification. WBF phục vụ pre-label; disagreement và candidate không được fusion chọn phục vụ QC. TTA và Quality Gate khóa export chưa triển khai.

## 2. Cài đặt

```powershell
.\Start-AnnoPilot.ps1 -Install
```

Frontend: `http://localhost:3000`. Swagger: `http://localhost:8000/docs`. Health trả `status: ok`.

Để inference local:

```powershell
.\backend\.venv\Scripts\python.exe -m pip install -r backend/requirements-models.txt
```

Base API nhận prediction JSON mà không cần torch/torchvision/ultralytics. Faster R-CNN có thể tải weights lần đầu. Inference lỗi không fallback sang prediction tổng hợp.

YOLO26n mặc định, YOLO26s là lựa chọn thay thế trong dropdown YOLO ở đầu trang Model QC. Lựa chọn được lưu trong browser và gửi vào API `model_name` cho Queue/frame/detector; M1 của ensemble dùng cùng lựa chọn. Chuyển biến thể sẽ inference lại nếu nguồn cũ đã inactive; prediction và model ID cũ được giữ làm lịch sử.

Checkpoint `.pt` chính thức nằm tại `backend/models/yolo26n.pt` và `backend/models/yolo26s.pt`. Máy khác cần tải từ `https://github.com/ultralytics/assets/releases/download/v8.4.0/`. Không dùng decoder OpenCV ONNX cũ cho YOLO26 NMS-free. Runtime tự chọn CUDA khi PyTorch hỗ trợ, nếu không dùng CPU; có thể ép CPU bằng `ANNO_INFERENCE_DEVICE=cpu`.

[Kết quả đo trên máy hiện tại và lệnh chạy benchmark](yolo26-hardware.md).

## 3. Nạp dữ liệu và QC

1. Upload annotation/media hoặc ZIP, chọn format phù hợp. Model QC hiện tập trung bounding box 2D.
2. Chạy queue baseline hoặc nạp prediction thật. Kết quả cập nhật theo frame.
3. Nạp thêm model hoặc chạy cấu hình YOLO26n/YOLO26s + Faster R-CNN + RT-DETR trong Multi-Model WBF. Pilot không bắt buộc đủ ba model.
4. Review theo QC Score. Mở **Bằng chứng từng model** để xem class/confidence/box và model không có detection tương ứng.
5. Sửa và Save trên CVAT; đồng bộ về AnnoPilot để xác minh.

Tier 1 là đồng thuận cao cần xác minh; Tier 2 là bất đồng class; Tier 3 là detection một model hoặc đồng thuận dưới ngưỡng. Frame chưa inference/import hoặc lỗi inference được báo riêng. Không có warning không có nghĩa không có lỗi.

Prediction JSON cho `POST /api/datasets/{id}/qc/predictions`:

```json
{
  "source_model": "yolo26n",
  "predictions": [
    {
      "file_name": "frame_000001.jpg",
      "checkpoint": "yolo26n.pt",
      "inference_seconds": 0.12,
      "predictions": [
        {"label": "car", "confidence": 0.91, "geometry": {"x1": 100, "y1": 120, "x2": 250, "y2": 220}}
      ]
    },
    {"file_name": "frame_000002.jpg", "checkpoint": "yolo26n.pt", "predictions": []}
  ]
}
```

Số liệu trên chỉ minh họa schema. Tên frame phải khớp dataset. Nạp lại thay prediction cùng nguồn trên các frame được cung cấp. Frame không có detection vẫn cần `predictions: []` để ghi coverage. Checkpoint/thời gian phải phản ánh dữ liệu thực.

Tên nguồn khác nhau chưa chứng minh model độc lập. Inference local ghi checkpoint/hash và phiên bản thư viện; evaluator từ chối nguồn có cùng hash checkpoint đã biết. Với import không có hash, nhóm phải kiểm chứng nguồn model.

## 4. Golden Benchmark và baseline → pilot

Bắt đầu với khoảng 30–50 frame được người kiểm chứng xem toàn ảnh, gồm vùng không có warning/prediction. Không lấy consensus làm ground truth; không lấy ảnh train làm validation độc lập.

Nạp JSON trong **Đánh giá Baseline → Pilot & vòng nghiệm thu**:

```json
{
  "verified_by": "Người kiểm chứng độc lập",
  "notes": "Đã xem toàn ảnh, gồm vùng không có warning",
  "includes_unwarned_regions": true,
  "samples": [
    {
      "file_name": "frame_000001.jpg",
      "annotations": [
        {"label": "truck", "geometry": {"x1": 100, "y1": 120, "x2": 250, "y2": 220}},
        {"label": "pedestrian", "geometry": {"x1": 400, "y1": 150, "x2": 430, "y2": 240}}
      ]
    },
    {"file_name": "frame_000002.jpg", "annotations": []}
  ]
}
```

Đây là ví dụ format, không phải nhãn đúng của ảnh thực. Golden `annotations: []` chỉ dùng khi reviewer xác nhận không có vật thể trong taxonomy. Attestation do người cung cấp chịu trách nhiệm; phần mềm không tự chứng minh ground truth độc lập.

Chọn baseline và ít nhất hai nguồn pilot, gồm baseline. Tất cả model phải có coverage trên toàn bộ frame benchmark; kết quả rỗng cần record thành công. So sánh trước khi áp dụng sửa nhãn.

Evaluator đối soát annotation hiện hành với golden bằng greedy IoU ≥ .50, không ép cùng class: golden unmatched là Missing Object; matched nhưng khác class là Wrong Class. Warning matched một-một với lỗi thật cùng loại và vị trí, không tính trùng.

- Recall = TP / (TP + FN), riêng hai loại lỗi.
- Precision = TP / (TP + FP); F1 cũng được lưu.
- Nhận ra vùng sai class khác với đề xuất đúng class: `class_suggestion_accuracy` đo riêng class gợi ý trên các TP.
- Shared omission của mọi model vẫn tính FN nhờ golden.
- Không có lỗi thật: recall `null`/`—`; không có warning: precision `null`/`—`.
- Lỗi thật/giờ review = TP × 60 / phút review thực tế đã nhập; chưa đo thì để trống.
- QC time đo riêng; inference time là tổng theo model/frame nếu có đủ thông tin. Có thể gồm model load/warmup, cần chuẩn hóa chế độ đo khi báo cáo.

Mỗi cặp run lưu config, snapshot annotation/prediction/provenance và kết quả. Evaluator không sửa annotation/operational issue. Hai run với đầu vào khác nhau không được dùng như một phép so sánh công bằng.

## 5. Review, CVAT và nghiệm thu

- ACCEPTED: reviewer chấp nhận gợi ý, cập nhật mirror; retry cùng quyết định không thêm box lần nữa.
- REJECTED/FALSE_POSITIVE: ghi lý do, giữ lịch sử evidence.
- Xác minh đã sửa trên CVAT: dùng live sync; local audit không chứng minh đã sửa trên CVAT.
- Undo: mở lại issue trước khi đổi quyết định; lưu lịch sử.

Cấu hình URL, Task/Job và token của người dùng; không có token demo mặc định. Live/API lỗi không fallback về file cũ để xác minh. Sync rectangle tracks tính nội suy theo frame của dataset; cần kiểm tra frame mapping với Task/Job thực tế.

Push `merged` chỉ bổ sung Missing Object ACCEPTED/RESOLVED nếu chưa có annotation tương ứng. Không thêm PENDING/REJECTED/FALSE_POSITIVE. Push `predictions` ưu tiên fused pre-label, không gửi đồng thời mọi box từng model; annotator vẫn phải review.

Append tránh box trùng chính xác trong `shapes` hiện có; chưa bảo đảm idempotency giữa worker concurrent hoặc box đã được người sửa. Replace ghi đè shapes, giữ tracks/tags lấy từ CVAT; đồng bộ inbound trước khi dùng. Class không mapping được phải sửa, không gán vào label bất kỳ.

Ghi **Nộp → Trả lại → Nộp lại → Chấp nhận**. Trả lại bắt buộc lý do. Số vòng trả lại đếm RETURNED event, không đếm issue/audit. Mỗi dataset giữ một cấu hình nghiệm thu; so baseline/pilot trên hai bản dataset tương đương với cùng tiêu chí, reviewer và quy mô. Đây là quyết định con người ghi nhận, chưa có Quality Gate tự động.

## 6. API mới

| Endpoint | Mục đích |
|---|---|
| POST `/api/datasets/{id}/qc/benchmarks` | Nạp phiên bản golden |
| GET `/api/datasets/{id}/qc/evaluations` | Golden và lịch sử metric |
| POST `/api/datasets/{id}/qc/evaluations/compare` | Lưu cặp run baseline/pilot |
| GET `/api/datasets/{id}/qc/evaluations/{run_id}` | Config, snapshot và evidence |
| GET/POST `/api/datasets/{id}/qc/acceptance` | Lifecycle nghiệm thu |
| GET `/api/samples/{id}/qc/review-history` | Lịch sử review |

Compare payload: `benchmark_id`, `baseline_model`, `pilot_models`, optional `config`, `evaluation_iou`, `review_minutes`. Swagger có schema đầy đủ.

## 7. Kiểm tra và giới hạn

```powershell
Set-Location backend
.\.venv\Scripts\python.exe -m pytest -q
Set-Location ../frontend
npm.cmd run build
```

Synthetic benchmark chỉ kiểm tra phần mềm, không chứng minh recall thật. Queue chưa resume qua restart. TTA, tự động push/issue vào CVAT từng frame và Quality Gate khóa export là bước sau. Chưa có benchmark thực tế chứng minh pilot cải thiện baseline.
