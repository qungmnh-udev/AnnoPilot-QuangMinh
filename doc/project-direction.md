# Hướng triển khai AnnoPilot đã chốt

Chốt theo trao đổi ngày 13/10/2026, đối chiếu bản nhận xét mentor cập nhật 11/10/2026 trong workspace repo_T003.

Mục tiêu: Model-assisted QC phát hiện sót vật thể và sai class, ưu tiên nghi vấn để reviewer tìm nhiều lỗi thật với ít công review hơn.

CVAT giữ annotation chính. Workflow: pretrained model tạo pre-label → annotator sửa → AnnoPilot QC → reviewer xác nhận/sửa → đồng bộ → nghiệm thu.

Baseline: YOLO26n mặc định hoặc YOLO26s đã chọn, một lần inference. Pilot: multi-model cross-verification với cùng biến thể YOLO baseline; WBF phục vụ pre-label, QC đọc prediction gốc, disagreement và candidate chưa được chọn. Không bắt buộc ba model; cần kiểm chứng checkpoint độc lập. YOLOv8 cũ chỉ giữ để đối chứng lịch sử.

Golden Benchmark độc lập đo recall Missing Object/Wrong Class, precision/F1, chất lượng class gợi ý, lỗi thật/giờ review, inference/QC time và số vòng trả lại. Không lấy consensus làm ground truth, không lấy số warning/Tier 1 làm bằng chứng tiết kiệm công review.

Ưu tiên: baseline/benchmark → disagreement QC → đo lường/vòng nghiệm thu → streaming/sync → thực nghiệm pilot. TTA và Quality Gate khóa export thuộc bước mở rộng.

Code hiện có candidate evidence, evaluator với snapshot, lifecycle nghiệm thu và UI tương ứng. Chưa hoàn thành benchmark thật; queue qua restart, push/issue CVAT tự động từng frame và TTA chưa triển khai.
