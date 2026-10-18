# Kết quả chạy thật WBF và perturbation · 15/10/2026

**Trạng thái: COMPLETED · 15 ảnh · 540 calls đo + 3 calls warm-up.**

Model: YOLO26n, Faster R-CNN MobileNetV3, RT-DETR-l. Thiết bị CPU, torch 8 thread; confidence 0,35, fusion IoU 0,55, weight mỗi model bằng 1. Gaussian sigma 8, brightness +15%, Gaussian blur radius 0,7; seed 42. Các nhánh dùng cùng bộ ảnh, mỗi nhánh inference riêng.

| Cấu hình | Calls | Box đầu ra | Inference tổng (s) | Fusion tổng (s) | F1 nhất quán với YOLO một lượt |
|---|---:|---:|---:|---:|---:|
| YOLO26n, 1 lượt gốc | 15 | 277 | 2.157 | 0.001 | 100.00% |
| YOLO26n, 4 lượt ảnh gốc | 60 | 277 | 7.953 | 0.002 | 100.00% |
| YOLO26n, gốc + 3 loại nhiễu | 60 | 290 | 9.496 | 0.056 | 97.35% |
| 3 model WBF, 1 lượt gốc/model | 45 | 1022 | 41.051 | 0.261 | 42.03% |
| 3 model WBF, 4 lượt gốc/model | 180 | 1022 | 170.302 | 0.230 | 42.03% |
| 3 model WBF, gốc + 3 loại nhiễu/model | 180 | 1148 | 162.727 | 0.954 | 38.46% |

Warm-up riêng: **8.673 s**. Tổng wall time gồm tạo PNG nhiễu, inference, fusion, ghi report và overhead: **430.402 s**.

## Diễn giải

- Các lượt lặp ảnh gốc giữ nguyên prediction khi detector trả kết quả giống nhau; bước gộp giữ nguyên các box chồng lấn khác nhau trong trường hợp này.
- Prediction lặp ảnh gốc khớp chính xác với nhánh một lượt trên cả 15 ảnh, cho cả một model và nhiều model.
- Nhánh có nhiễu thay đổi số box và mức nhất quán. Box tăng chưa chứng minh recall tăng; có thể gồm false positive.
- **Chưa có Golden độc lập trong lượt đo này**, do đó precision/recall/F1 chất lượng detection và QC đều để null. F1 trong bảng chỉ so với prediction baseline, không phải ground truth.
- CLI không cung cấp annotation cần QC nên không báo số lỗi QC giả định trên các ảnh này.
- Thời gian là số đo tham khảo trong phiên làm việc trên CPU; một số kiểm chứng khác có chạy song song ở đầu phiên. Chưa phải benchmark hiệu năng trong môi trường kiểm soát tải nền.

## Bằng chứng và tái lập

- [Summary JSON, config, checkpoint/hash và danh sách ảnh](../backend/benchmarks/wbf-perturbations-summary.json).
- [Raw prediction từng lượt trong workspace](../.runtime/wbf-experiments/demo-real-15-final.json) (artifact local, được gitignore).
- [Hướng dẫn chạy UI, CLI và đọc metric](wbf-perturbation-experiment.md).

## Kiểm chứng phần mềm

- Backend: **89 tests passed**.
- TypeScript + Vite production build: passed, xuất sang `.runtime/build-wbf-20261015` vì thư mục dist cũ bị chặn quyền cleanup.
- API WBF thông thường chạy thật trên một ảnh: 116 prediction từ ba model, 62 box fusion với cấu hình triad mặc định.
- API thử nghiệm chạy thật sáu cấu hình trên một ảnh; POST, polling trạng thái và tải report thành công.
- Smoke tests dùng database tách biệt; không sửa annotation/prediction/issue trong database vận hành. UI đã build; chưa có kiểm thử trình duyệt tự động.
