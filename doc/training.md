# Huấn luyện model ngay trong Model QC

## Cách dùng

1. Nạp dataset có ảnh và box; sửa, Save trên CVAT rồi đồng bộ nhãn về AnnoPilot.
2. Mở **Model QC → Huấn luyện model của bạn**. Dataset đang chọn là dữ liệu dùng để học.
3. Chọn **YOLO26n** cho lần đầu, giữ **Thông thường · 30 vòng**. Cũng hỗ trợ YOLO26s, RT-DETR-l và Faster R-CNN MobileNetV3.
4. Xác nhận đã kiểm tra nhãn rồi **Bắt đầu huấn luyện**. Không cần nhập đường dẫn checkpoint.
5. Theo dõi số vòng, số ảnh học/kiểm tra và thiết bị thực tế. Có thể đóng mục này rồi mở lại. Một thời điểm chỉ chạy một job training.
6. Hoàn tất: checkpoint được tự đăng ký cho **model và dataset đó**. Chạy lại **Queue QC/kiểm tra frame** với biến thể YOLO vừa học, hoặc **Inference & WBF** để dùng các model đã học. Prediction cũ được đánh dấu cần chạy lại, không được đổi tên thành kết quả mới.
7. **Dừng huấn luyện** yêu cầu dừng ở bước an toàn. Job lỗi/dừng giữ checkpoint cũ. Trong **Model đang dùng và các phiên bản trước**, có thể dùng lại checkpoint trước hoặc model gốc.

Ảnh chưa có nhãn được bỏ qua. Chỉ bật tùy chọn dùng ảnh rỗng nếu đã xác minh rằng ảnh không có vật thể thuộc taxonomy. Cần ít nhất 4 ảnh khác nhau để kiểm tra pipeline; bộ nhỏ chưa đủ chứng minh chất lượng.

## Dữ liệu và đo chất lượng

- Hỗ trợ BBox 2D của 10 class: pedestrian, rider, car, truck, bus, train, motorcycle, bicycle, traffic light, traffic sign. Class lạ, box ngoài ảnh và bản nhãn mâu thuẫn của cùng nội dung ảnh bị từ chối.
- Tạo snapshot nhãn và bản sao ảnh cho mỗi lượt. Ảnh đổi nội dung trong lúc chuẩn bị làm job dừng.
- Mặc định 80% train / 20% validation, seed 42. Ảnh trùng byte được gộp theo SHA256; không chép ảnh train sang validation. Ảnh đã đổi encoding/crop và frame liền nhau vẫn cần người dùng tách theo cảnh/video để tránh rò rỉ.
- Loại ảnh Golden đã khai báo trước khi chia, gồm cả bản trùng byte ở dataset khác. Chuẩn bị Golden trước khi train. Evaluator từ chối dùng ảnh train/validation đã ghi trong provenance làm Golden của chính checkpoint đó. Chưa phát hiện mọi ảnh đã chỉnh sửa hoặc nguồn import không có provenance.
- YOLO26/RT-DETR hiển thị metric validator thư viện. Faster R-CNN chọn checkpoint theo validation loss; không gọi loss là mAP. Đo precision/recall trên Golden độc lập sau training.
- Hoàn tất training tự kích hoạt checkpoint, chưa có quality gate chứng minh tốt hơn bản trước. Có thể quay về bản gốc. Đồng thuận model không phải ground truth.

## CPU và NVIDIA GPU

Tự chọn GPU khi PyTorch nhận CUDA; nếu không dùng CPU. Chọn GPU rõ ràng mà CUDA không khả dụng sẽ báo lỗi. Training và inference của cả ba họ đều hỗ trợ thiết bị CPU/CUDA.

Máy được kiểm tra có RTX 3050 Laptop 4 GB, nhưng môi trường đang dùng `torch 2.14.1+cpu`. Agent không tải được wheel CUDA vì kết nối `download.pytorch.org` bị chặn WinError 10013; chưa kiểm chứng training thật trên GPU.

Dừng backend và job training rồi chạy tại máy có mạng, từ gốc repo:

```powershell
.\Install-GPU.ps1
```

Script cài torch/torchvision từ index CUDA 13.0 rồi kiểm tra CUDA và phép tính tensor GPU. Có thể chọn `-Cuda cu128` khi phù hợp driver/wheel Python; đối chiếu [bộ chọn cài đặt PyTorch](https://pytorch.org/get-started/locally/). Khởi động lại backend sau khi cài; UI sẽ hiển thị GPU khi sẵn sàng. Với 4 GB, bắt đầu YOLO26n, batch 1, ảnh 320/640; nếu thiếu VRAM, giảm kích thước hoặc dùng CPU.

## Vận hành

- Job/progress lưu trong SQLite; worker là process riêng, API restart không tự dừng worker. Mất heartbeat hơn 120 giây thì đánh dấu INTERRUPTED; có thể bắt đầu lượt mới, chưa tự resume optimizer sau khi máy/worker chết.
- Slot unique trong DB ngăn nhiều API worker train đồng thời. Snapshot/weights/log ở `training-jobs/<UUID>` hoặc biến `TRAINING_DIR`. Backup thư mục này cùng DB và uploads.
- Registry lưu SHA256, model, dataset, job và hash ảnh đã dùng. Checkpoint được thử reload/inference trước khi kích hoạt; không ghi đè weights gốc.
- YOLOv8 và decoder ONNX cũ đã bị loại khỏi đường chạy; import, ensemble và evaluation từ chối nguồn YOLOv8. Dữ liệu lịch sử không bị đổi thành YOLO26.
- API: `GET /api/training/hardware`; `GET /api/datasets/{id}/training/preview`; `GET/POST .../training`; `POST .../training/{job_id}/cancel`; `GET .../training/{job_id}/log`; `POST .../training/checkpoint`.
- POST training yêu cầu `labels_reviewed: true`; tùy chọn epochs, device (auto/cpu/gpu), image_size, batch_size, seed, validation_fraction và include_verified_empty.

CLI dùng cùng backend:

```powershell
.\backend\.venv\Scripts\python.exe training/train.py --dataset-id 1 --model yolo26n --device auto --epochs 30 --labels-reviewed
```

Smoke training thật với ảnh tổng hợp/database riêng, không phải benchmark chất lượng:

```powershell
.\backend\.venv\Scripts\python.exe backend/scripts/smoke_training.py --models yolo26n rtdetr_l fasterrcnn_mobilenet_v3
```

Tham khảo triển khai: [callbacks Ultralytics](https://docs.ultralytics.com/usage/callbacks), [TorchVision finetuning](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html).
