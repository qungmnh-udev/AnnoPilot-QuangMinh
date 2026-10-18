# Chạy Multi-Model WBF thực tế và so sánh inference có nhiễu

## Mục tiêu

Chạy detector thật trên cùng bộ ảnh để so sánh ảnh hưởng của số model, số lượt inference và nhiễu ảnh. Không sinh prediction tổng hợp, không lấy số box tăng làm bằng chứng chất lượng tăng.

Đã chạy đủ sáu cấu hình trên **15 ảnh / 540 calls đo**: [báo cáo thực tế ngày 15/10](wbf-experiment-results.md).

## Sáu cấu hình đối chiếu

| Cấu hình | Model | Ảnh đầu vào mỗi frame | Số lần gọi detector mặc định |
|---|---|---|---:|
| single_model_once | YOLO26 đang chọn | Gốc một lần | 1 |
| single_model_repeated_clean | Cùng YOLO26 | Gốc lặp 4 lần | 4 |
| single_model_repeated_perturbed | Cùng YOLO26 | Gốc + Gaussian + độ sáng + blur | 4 |
| multi_model_once | YOLO26 + Faster R-CNN MobileNetV3 + RT-DETR-l | Gốc một lần/model | 3 |
| multi_model_repeated_clean | Ba model trên | Gốc lặp 4 lần/model | 12 |
| multi_model_repeated_perturbed | Ba model trên | Gốc + Gaussian + độ sáng + blur trên mỗi model | 12 |

Mặc định **36 calls/frame**, chưa kể một lần warm-up mỗi model trước các nhánh đo. Các nhánh chạy inference riêng, không lấy lại prediction từ nhánh khác để giả lập tốc độ. Nhánh một lượt cũng dùng PNG lossless từ ảnh đã giải mã để mọi nhánh đối chiếu cùng pixel gốc.

Lặp ảnh gốc giúp kiểm tra tính ổn định của detector. So nhánh lặp sạch với nhánh lặp có nhiễu của cùng cấu hình model để đo tác động của perturbation. So một model với nhiều model trong cùng chế độ ảnh để đo tác động của ensemble.

## Cách chạy trên giao diện

1. Khởi động lại backend để nạp API mới; mở frontend dev hiện tại hoặc build phiên bản mới.
2. Upload/chọn dataset có ảnh và annotation khớp nhau.
3. Vào **Model QC**, chọn YOLO26n hoặc YOLO26s.
4. Mở **Multi-Model WBF**. Phần đầu modal có **Thử nghiệm WBF thực tế: một lượt và nhiều lượt có nhiễu**.
5. Chọn số ảnh (1–30), số lượt (2–8), seed và cường độ nhiễu. Dùng 4 lượt để có đủ Gaussian, độ sáng, blur.
6. Nếu có Golden đã kiểm chứng, nạp trước ở panel đánh giá, sau đó chọn Golden trong panel thử nghiệm. Nếu không có, app chỉ đo tốc độ và mức nhất quán.
7. Bấm **Chạy so sánh inference thật**. Theo dõi số frame đã xong và cấu hình đang chạy; bảng kết quả cập nhật sau mỗi frame hoàn tất đủ sáu nhánh.
8. Chọn lịch sử thử nghiệm và tải JSON để xem prediction từng lượt, phép biến đổi, thời gian và kết quả WBF.

Thử nghiệm chạy nền và vẫn tiếp tục khi đóng modal. Mỗi process API chỉ nhận một thử nghiệm tại một thời điểm. Chưa có nút hủy/resume; restart backend làm run đang chạy thành INTERRUPTED khi truy vấn lại. Các báo cáo đã hoàn thành vẫn lưu trên đĩa. Không xóa dataset hoặc media trong khi chạy thử nghiệm.

## Cấu hình nhiễu và fusion

- Gaussian: sigma mặc định **8** theo thang pixel 0–255, seed mặc định **42**.
- Độ sáng: lượt độ sáng đầu tăng **15%**; lượt độ sáng tiếp theo nếu có giảm 15%.
- Blur: Gaussian blur radius mặc định **0,7**.
- Lượt 0 luôn là ảnh gốc. Lượt tiếp theo luân phiên Gaussian → brightness → blur; trên 4 lượt tiếp tục chu kỳ với seed theo frame/lượt.
- Chỉ đổi quang học, giữ nguyên kích thước và hệ tọa độ box; không cần đảo phép biến đổi hình học.
- Confidence detector/fusion mặc định **0,35**; IoU fusion **0,55**; các model có weight bằng nhau trong phép so sánh này. Panel WBF thông thường vẫn có cấu hình weights riêng.
- Lượt lặp của từng model được fusion trước; sau đó mỗi model vật lý chỉ có **một phiếu** trong WBF giữa model. Bốn lượt YOLO không được tính như bốn detector độc lập hoặc tự tạo Tier 1 nhiều model.
- QC của thử nghiệm chạy trên kết quả đã gộp theo từng model; prediction từng lượt và biến đổi vẫn giữ trong report để kiểm tra.

Các kết quả lưu **riêng** tại `.runtime/wbf-experiments/<dataset_id>/<run_id>.json`. Không áp dụng nhãn gợi ý, không thay prediction/issue đang review, không push CVAT. Muốn dùng pre-label vận hành, chạy luồng Multi-Model WBF thông thường rồi review/push riêng.

## Đọc kết quả

| Chỉ số | Cách hiểu |
|---|---|
| Calls / detections | Lượng xử lý / số box đầu ra; nhiều box hơn có thể gồm cả vật thể mới và false positive. |
| Inference / fusion / QC seconds | Thời gian từng phần; warm-up tách riêng, tổng wall time còn có I/O và tạo ảnh nhiễu. |
| Detection precision/recall/F1 | Chỉ tính khi có Golden; match một-một cùng class với IoU ≥ 0,50. Đây là metric tại ngưỡng cố định, không phải COCO mAP. |
| QC recall Missing/Wrong Class | Chỉ có khi có Golden và annotation đầu vào; đánh giá khả năng phát hiện lỗi của bản annotation. |
| F1 nhất quán với baseline | So box đầu ra với YOLO một lượt gốc. Baseline không phải ground truth; giá trị thấp có thể là thêm đối tượng đúng hoặc detection sai. |

Không có Golden thì quality metric là `null`/`—`. Nếu CLI chưa cung cấp annotation đầu vào, số issue/QC metric cũng để trống thay vì giả định mọi prediction là vật thể bị sót. Tên nguồn, checkpoint/hash, phiên bản thư viện và runtime được ghi vào báo cáo thật.

## Chạy bằng CLI, không đụng database vận hành

Từ thư mục gốc AnnoPilot:

```powershell
.\backend\.venv\Scripts\python.exe backend/scripts/benchmark_wbf_perturbations.py --limit 15 --repeats 4 --output .runtime/wbf-experiments/demo-real-15.json
```

Đo chất lượng khi đã có Golden JSON đúng schema benchmark:

```powershell
.\backend\.venv\Scripts\python.exe backend/scripts/benchmark_wbf_perturbations.py --images duong-dan-anh --golden golden.json --annotations annotation-snapshot.json --limit 15 --repeats 4 --output .runtime/wbf-experiments/golden-comparison.json
```

`annotation-snapshot.json` chứa `samples[{file_name, annotations[{label, geometry:{x1,y1,x2,y2}}]}]`, là nhãn cần QC trước sửa, không phải Golden. Nếu chỉ đo detection thì có thể bỏ `--annotations`. CLI giới hạn ảnh như cấu hình API, đặt torch 8 thread; kết quả thời gian không mặc nhiên tương đương process API có số thread khác.

## API

- `POST /api/datasets/{id}/ensemble/experiments`: bắt đầu, trả 202 và run ID; request có model, số ảnh/lượt, seed, cường độ nhiễu và benchmark_id tùy chọn.
- `GET /api/datasets/{id}/ensemble/experiments`: 20 run gần nhất và summary.
- `GET /api/datasets/{id}/ensemble/experiments/{run_id}`: trạng thái/progress/summary.
- `GET /api/datasets/{id}/ensemble/experiments/{run_id}/report`: JSON đầy đủ prediction từng lượt.

## Giới hạn

Đây là thử nghiệm perturbation/TTA quang học, chưa tích hợp vào Queue QC mặc định hay CVAT streaming. Các lượt không phải phép đo thống kê trên nhiều dataset độc lập. Muốn kết luận nhiễu cải thiện chất lượng cần Golden độc lập, đủ cỡ mẫu, giữ cùng cấu hình và kiểm tra cả false positive lẫn false negative. Chưa có Quality Gate khóa export.
