# Sổ Tay Hướng Dẫn Sử Dụng AnnoPilot — Model-Assisted QC

> **Dự án:** AnnoPilot — Nền tảng Đánh giá & Kiểm soát Chất lượng Gán nhãn Dữ liệu  
> **Chuyên đề N2-04D:** *Model-assisted QC: bắt vật thể bị sót và sai class trên BDD100K*  
> **Nhóm thực hiện:** Nhóm 2 – Giữ chất lượng dữ liệu xe tự hành (Autonomous Driving)  
> **Tập dữ liệu chuẩn:** BDD100K (Berkeley DeepDrive 100K)  
> **Phiên bản:** v1.0.0 (Cập nhật: 2026)

---

## MỤC LỤC
1. [Giới thiệu & Bài toán Nghiệp vụ](#1-giới-thiệu--bài-toán-nghiệp-vụ)
2. [Kiến trúc Kỹ thuật & Cấu trúc Thư mục](#2-kiến-trúc-kỹ-thuật--cấu-trúc-thư-mục)
3. [Yêu cầu Hệ thống & Môi trường](#3-yêu-cầu-hệ-thống--môi-trường)
4. [Cài đặt & Khởi động Nhanh](#4-cài-đặt--khởi-động-nhanh)
5. [Kịch bản Kiểm thử Tự động qua Terminal (CLI Test)](#5-kịch-bản-kiểm-thử-tự-động-qua-terminal-cli-test)
6. [Hướng dẫn Chi tiết Thao tác trên Giao diện Web (Web UI)](#6-hướng-dẫn-chi-tiết-thao-tác-trên-giao-diện-web-web-ui)
7. [Đặc tả Hệ thống API Endpoints](#7-đặc-tả-hệ-thống-api-endpoints)
8. [Xử lý Sự cố Thường gặp (Troubleshooting)](#8-xử-lý-sự-cố-thường-gặp-troubleshooting)

---

## 1. Giới thiệu & Bài toán Nghiệp vụ

### 1.1. Bối cảnh
Trong huấn luyện mô hình thị giác máy tính cho xe tự hành (Autonomous Driving), chất lượng của tập dữ liệu gán nhãn (Ground Truth) là yếu tố sống còn. Tuy nhiên, việc rà soát thủ công (Manual QC) toàn bộ hàng chục ngàn bức ảnh từ tập BDD100K gặp 3 rào cản nghiêm trọng:
1. **Lãng phí nguồn lực:** Rất nhiều ảnh hoàn hảo nhưng reviewer vẫn phải bấm mở xem từng ảnh, gây hao tổn 70–80% thời gian vô ích.
2. **Nguy cơ bỏ sót lỗi tử huyệt:** Con người mỏi mắt dễ bỏ sót người đi bộ từ xa (`pedestrian`), người lái xe máy/xe đạp (`rider`) hoặc xe con khuất góc nhìn (`car`).
3. **Nhầm lẫn các lớp đối tượng tương đồng:** Thường xuyên nhầm giữa `bus` và `truck`, `bicycle` và `motorcycle`, `pedestrian` và `rider`.

### 1.2. Giải pháp AnnoPilot Model-Assisted QC (Human-in-the-loop)
AnnoPilot ứng dụng một mô hình nhận diện vật thể tiền huấn luyện (**Pretrained Object Detector - YOLOv8**) đóng vai trò làm **Reviewer thứ hai (Second Reviewer)** đối soát độc lập với nhãn người gán:
* **Bắt lỗi sót vật thể (`MISSING_OBJECT`):** Model phát hiện vật thể với độ tin cậy cao ($Confidence \ge 0.45$) nhưng người gán nhãn không hề vẽ box ($IoU < 0.30$).
* **Bắt lỗi sai phân loại (`WRONG_CLASS`):** Cả người và model đều vẽ box trùng tọa độ ($IoU \ge 0.50$), nhưng nhãn người gán khác nhãn model dự đoán, rơi vào vùng tiền nghiệm nhầm lẫn lái xe (`Driving Confusion Priors`).
* **Hàng đợi ưu tiên (Prioritized Review Queue):** Chấm điểm rủi ro (`qc_score`) từ 0.0 đến 1.0; tự động đẩy các bức ảnh nguy cơ lỗi cao nhất (`HIGH`) lên đầu danh sách để chuyên gia rà soát trước.
* **Cắt giảm khối lượng công việc (Workload Reduction):** Tự động phân loại các ảnh có độ đồng thuận tuyệt đối là `CLEAN` và bỏ qua rà soát thủ công.
* **Tích hợp sâu CVAT qua Deep-link (`Open in CVAT`):** Kết nối trực tiếp với Annotation Editor của CVAT theo định dạng URL `.../tasks/{task_id}/jobs/{job_id}?frame={frame_number}`, giúp annotator nhảy thẳng vào đúng frame để sửa.
* **Sửa lỗi khép kín (Closed-Loop Resolution):** Hỗ trợ cả 2 chế độ:
  - *Chế độ CVAT chuẩn:* Bấm `[ ↗ Open in CVAT ]` để sửa nhãn nguồn, sau đó bấm `[ ✓ Đã sửa trên CVAT (Resolved) ]`.
  - *Chế độ Ngoại tuyến (Fast Demo):* Bấm `[ ⚡ 1-Click Sửa tự động ]` để tự động thêm box / đổi class ngay lập tức (tiện cho demo mà không cần khởi động cụm Docker CVAT).

### 1.3. Sơ đồ Kiến trúc CVAT × QC Dashboard (Chuẩn tài liệu thiết kế v0.1)

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   TẦNG 1: CVAT (SYSTEM OF RECORD)                       │
│    - Lưu trữ dữ liệu gốc: Video, Image Frames theo Frame Number (0..N) │
│    - Annotation Editor: Chỉnh sửa bounding box, class, polygon        │
│    - Export / REST API: Xuất dữ liệu COCO, BDD100K JSON                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Export Data (BDD100K / COCO)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               TẦNG 2: MODEL QC BACKEND & PIPELINE ENGINE               │
│    - Pretrained Detector Service: Chạy mô hình YOLOv8 ONNX             │
│    - Discrepancy Engine: So khớp IoU, phát hiện MISSING & WRONG_CLASS   │
│    - Prioritization: Chấm điểm QC Score & xếp hạng độ nghi vấn         │
│    - Deep-link Generator: Sinh URL /tasks/{task_id}?frame={frame_num}   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ JSON API & Deep-link URL
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│              TẦNG 3: QC DASHBOARD FRONTEND (REVIEWER WORKSPACE)        │
│    - KPI Metrics: Tỷ lệ giảm tải (Workload Reduction), Số ca Flagged   │
│    - Prioritized Queue: Hàng đợi ảnh lỗi xếp từ rủi ro cao đến thấp   │
│    - Visual Diff: So sánh trực quan Lớp người gán vs Lớp model        │
│    - Nút bấm chủ đạo: [ ↗ Open in CVAT ] (Mở thẳng đúng frame trên CVAT)│
│    - Triage khép kín: [ Đã sửa trên CVAT ] / [ 1-Click Sửa ] / [ Báo sai]│
└────────────────────────────────────────────────────────────────────────┘
```

> **Nguyên tắc cốt lõi phân định trách nhiệm:**
> - **CVAT = Nơi sửa dữ liệu (System of Record):** Lưu trữ toàn bộ dữ liệu nguồn, đảm bảo tính nhất quán của tập nhãn.
> - **QC Dashboard = Nơi phát hiện, xếp hạng và review (Decision & Prioritization Layer):** Không gán nhãn lại từ đầu mà tập trung phân tích, xếp hạng và điều hướng reviewer xử lý các ca lỗi nhanh nhất.

## 2. Kiến trúc Kỹ thuật & Cấu trúc Thư mục

### 2.1. Ngăn xếp Công nghệ (Tech Stack)
* **Backend:** Python 3.10+, FastAPI, SQLAlchemy ORM, SQLite, OpenCV DNN (chạy mô hình YOLOv8 ONNX không cần GPU hay PyTorch nặng nề), Pydantic.
* **Frontend:** React 19, TypeScript, Vite 6, Tailwind/CSS3 phong cách Glassmorphism & Cyber-inspection, Lucide React Icons.
* **Mô hình AI:** `yolov8n.onnx` hỗ trợ các lớp đối tượng giao thông BDD100K (`car`, `truck`, `bus`, `pedestrian`, `rider`, `bicycle`, `motorcycle`, `traffic light`, `traffic sign`).

### 2.2. Bản đồ Thư mục Dự án

```text
AnnoPilot/
├── doc/
│   └── huong-dan-su-dung.md      # Tài liệu này (Hướng dẫn chi tiết)
├── backend/
│   ├── app/
│   │   ├── main.py               # Khởi tạo FastAPI app và mount routes
│   │   ├── config.py             # Cấu hình đường dẫn, upload dir, database URL
│   │   ├── database.py           # Kết nối SQLite & session maker
│   │   ├── models/               # Khai báo cấu trúc bảng CSDL (Sample, Annotation, QCIssue, Prediction)
│   │   ├── parsers/              # Bộ phân tích định dạng: BDD100K, CVAT, COCO, YOLO, KITTI
│   │   ├── services/
│   │   │   └── qc_engine.py      # Core AI Engine: IoU matching, ranking, detection, resolution
│   │   └── routers/
│   │       ├── api.py            # Toàn bộ API endpoints RESTful (/qc/*, /datasets/*, /samples/*)
│   │       └── auth.py           # API xác thực người dùng
│   ├── models/
│   │   └── yolov8n.onnx          # Trọng số Pretrained Detector phục vụ QC
│   ├── demo_quick_test.py        # Script chạy thử toàn bộ luồng kiểm thử qua terminal
│   ├── tests/
│   │   ├── test_qc_engine.py     # Unit test thuật toán so khớp IoU và phát hiện lỗi
│   │   └── test_qc_api.py        # Integration test kiểm thử các API endpoints QC
│   └── requirements.txt          # Danh sách thư viện Python
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ModelQCPage.tsx   # Giao diện chính phân hệ Model-Assisted QC
│   │   │   ├── HomePage.tsx      # Giao diện Trang chủ tổng quan
│   │   │   └── DatasetView.tsx   # Quản lý xem chi tiết nhãn dataset
│   │   ├── hooks/
│   │   │   └── usePageNavigation.ts # Quản lý định tuyến và menu điều hướng
│   │   ├── App.tsx               # Khung bố cục chính và thanh Sidebar
│   │   ├── types.ts              # Interface định nghĩa kiểu dữ liệu TypeScript
│   │   └── style.css             # Định dạng giao diện Dark Mode cao cấp
│   ├── package.json
│   └── vite.config.ts
├── demo_data/
│   ├── images/                   # 15 ảnh đường phố BDD100K mẫu
│   └── bdd100k_val_sample.json   # Nhãn chuẩn định dạng BDD100K
├── Start-AnnoPilot.ps1           # Script PowerShell khởi động cả 2 server cùng lúc
└── annopilot.db                  # File CSDL SQLite (tự động tạo)
```

---

## 3. Yêu cầu Hệ thống & Môi trường

* **Hệ điều hành:** Windows 10/11, macOS, hoặc Linux Ubuntu 20.04+.
* **Python:** 3.10 trở lên.
* **Node.js:** Phiên bản 20 LTS hoặc 22 LTS (kèm `npm`).
* **RAM:** Tối thiểu 4 GB (Khuyên dùng 8 GB+).
* **Ổ cứng trống:** 2 GB.
* **Quyền PowerShell trên Windows:** Nếu gặp lỗi hạn chế script, mở PowerShell và chạy:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
  ```

---

## 4. Cài đặt & Khởi động Nhanh

### Cách 1: Khởi động tự động bằng Script 1 lệnh (Khuyên dùng trên Windows)
Tại thư mục gốc dự án:
```powershell
.\Start-AnnoPilot.ps1
```
Script sẽ tự động kiểm tra virtualenv, cài gói cần thiết và mở 2 cửa sổ server.

---

### Cách 2: Khởi động thủ công từng phần

#### Bước 1: Khởi động Backend (Port 8000)
Mở cửa sổ Terminal thứ nhất:
```powershell
cd d:\AI\repo\AnnoPilot-qminh\backend

# 1. Kích hoạt môi trường ảo Python
.\.venv\Scripts\Activate.ps1

# 2. Cài đặt dependencies (chỉ cần chạy lần đầu tiên)
pip install -r requirements.txt

# 3. Khởi chạy máy chủ FastAPI
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```
* **Kiểm tra trạng thái Backend:** Mở trình duyệt vào `http://localhost:8000/api/health` (Trả về `{"status":"healthy"}`).
* **Tài liệu Swagger UI:** `http://localhost:8000/docs`.

#### Bước 2: Khởi động Frontend (Port 3000)
Mở cửa sổ Terminal thứ hai:
```powershell
cd d:\AI\repo\AnnoPilot-qminh\frontend

# 1. Cài đặt các gói phụ thuộc (nếu chưa cài)
npm ci
# (Hoặc nếu PowerShell chặn script: npm.cmd ci)

# 2. Khởi chạy Vite Dev Server
npm run dev
# (Hoặc: npm.cmd run dev)
```
* **Mở ứng dụng trên trình duyệt:** **`http://localhost:3000`**

---

## 5. Kịch bản Kiểm thử Tự động qua Terminal (CLI Test)

Dự án cung cấp sẵn script `demo_quick_test.py` giúp bạn kiểm tra toàn bộ luồng xử lý của bài toán đề tài N2-04D ngay trên terminal trong vòng 2 giây mà không cần bấm tay:

```powershell
cd d:\AI\repo\AnnoPilot-qminh\backend
.\.venv\Scripts\python.exe demo_quick_test.py
```

### Các bước script tự động thực thi và in kết quả:
1. **Khởi tạo Dataset & Sample:** Tạo dataset mẫu BDD100K với 3 ảnh đường phố tiêu biểu.
2. **Nạp Ground Truth:** Nạp nhãn người gán (trong đó cố ý tạo 1 lỗi sót `car` và 1 lỗi gán nhầm `truck` thành `bus`).
3. **Nạp Pretrained Predictions:** Nạp kết quả phát hiện của mô hình YOLOv8.
4. **Đối soát & Chấm điểm (Audit Run):**
   * Tính chỉ số **Workload Reduction: 66.67%** (2/3 ảnh sạch không cần review).
   * Phát hiện chính xác 1 lỗi `MISSING_OBJECT` và 1 lỗi `WRONG_CLASS`.
5. **Xếp hạng hàng đợi:** Đẩy bức ảnh bị lỗi lên vị trí ưu tiên **#1** với điểm `qc_score = 0.920` (Mức độ: `HIGH`).
6. **Thực thi 1-Click Resolution:**
   * Tự động tạo thêm Annotation mới cho vật thể bị sót.
   * Tự động sửa nhãn `bus` thành `truck` trong CSDL.
   * Đánh dấu vấn đề QC là `ACCEPTED`.
7. **Chạy kiểm thử Unit Tests tự động:**
   ```powershell
   .\.venv\Scripts\python.exe -m pytest -q
   ```
   *Kết quả:* **48 passed** (Toàn bộ 48 bài kiểm thử đơn vị và tích hợp đều đạt 100%).

---

## 6. Hướng dẫn Chi tiết Thao tác trên Giao diện Web (Web UI)

### Bước 1: Nạp Dataset (CVAT demo.zip hoặc BDD100K) vào hệ thống
1. Truy cập `http://localhost:3000`.
2. Trên màn hình Dashboard, bấm nút **"Upload Dataset"** (góc trên bên phải).
3. Điền các trường thông tin:
   * **Dataset Name:** Ví dụ `CVAT Demo Tracking Day 03` (hoặc tên tuỳ ý).
   * **Annotation Format:** Chọn `Auto Detect` hoặc `CVAT`.
   * **Annotation Files / ZIP:** Chọn trực tiếp file `demo.zip` (từ thư mục `Downloads`).
     > **Lưu ý đặc biệt:** File `demo.zip` đã đóng gói sẵn cả file `annotations.xml` (dạng CVAT video track) và toàn bộ 360 ảnh `images/frame_000000.PNG` đến `frame_000359.PNG`. Backend AnnoPilot sẽ tự động giải nén, phát hiện ảnh và khớp 100% 360 frames cùng 2,135 bounding boxes mà **không cần nạp riêng file media**.
   * **Tự động liên kết CVAT:** Hệ thống tự động đọc `<task><id>2</id>` và `<segment><id>2</id>` trong metadata của `demo.zip`, tự động điền `CVAT Task ID = 2`, `Job ID = 2` và sinh Deep-link `http://localhost:8080/tasks/2/jobs/2?frame={frame_number}` cho từng frame.
4. Bấm **Upload & Analyze**. Hệ thống sẽ nạp dữ liệu và chuyển thẳng sang phân hệ Model QC.

---

### Bước 2: Mở phân hệ Model-Assisted QC
Trên thanh Sidebar bên trái màn hình, nhấn vào mục có biểu tượng lá chắn **`Model QC`**.

---

### Bước 3: Chạy Rà soát Đối soát (Audit Run) & Vận hành Tinh giản
Thanh công cụ phía trên trang Model QC đã được thiết kế lại tinh gọn, phân cấp rõ ràng thành **4 hành động cốt lõi** và **Menu Công cụ nâng cao**:

#### 1. Hành động cốt lõi 1: Chạy Hàng Đợi Streaming (Nút chính "Chạy Queue QC")
* Nhấn nút màu xanh dương nổi bật **"Chạy Queue QC"**.
* Hệ thống kích hoạt tiến trình nền chạy tuần tự từng frame (không làm đơ hay khóa máy chủ như chạy batch toàn bộ).
* Thanh tiến trình thời gian thực (Live Progress Banner) hiển thị % tiến độ, tên frame, số cảnh báo phát hiện và các nút Tạm dừng / Tiếp tục.

#### 2. Hành động cốt lõi 2: Multi-Model WBF (Nút tím Pre-labeling & Ensemble)
* Nhấn nút màu tím **"Multi-Model WBF"**.
* Mở hộp thoại tiền gán nhãn đa mô hình (Weighted Box Fusion) với **Phương án B (Khuyên dùng - Triad Ensemble)**:
  * **Bộ ba Đa Kiến Trúc:**
    * ⚡ **M1: YOLOv8n** (1-Stage CNN Anchor-free, Task-Aligned Assigner, Trọng số 1.0)
    * 🎯 **M2: Faster R-CNN MobileNetV3** (2-Stage CNN RPN + RoIAlign, Trọng số 1.0)
    * 👁️ **M3: RT-DETR-l** (Vision Transformer Deformable Attention, Trọng số 1.2)
  * **1-Click Inference & WBF:** Nhấn **"⚡ Chạy Inference & WBF Phương Án B"** để hệ thống tự động suy luận đồng thời 3 mô hình trên các frame và hợp nhất kết quả.
  * Tùy chỉnh trọng số (Weights) và các ngưỡng IoU / Confidence.
  * Phân tầng đồng thuận 3 cấp: Tier 1 (Đồng thuận cao - Auto accept), Tier 2 (Bất đồng class), Tier 3 (Lệch model).
  * Hỗ trợ nạp thêm file dự đoán JSON của model mới và có nút **1-Click đẩy Pre-label sang CVAT**.
  * Ngoài giao diện, có thể chạy qua dòng lệnh: `python backend/scripts/run_triad_inference.py --dataset-id <ID>`.

#### 3. Hành động cốt lõi 3: Đồng bộ từ CVAT (Inbound Pull)
* Nhấn nút **"Đồng bộ từ CVAT"** để nhận nhãn mới nhất khi annotator vừa chỉnh sửa trên CVAT. Hệ thống tự động đối soát lại và giải phóng cảnh báo (Auto-resolve).

#### 4. Hành động cốt lõi 4: Đẩy nhãn sang CVAT (Outbound Push — An toàn tuyệt đối)
* Nhấn nút **"Đẩy nhãn sang CVAT"** để đồng bộ dữ liệu sang CVAT Task/Job qua REST API.
* Cho phép chọn 3 nguồn dữ liệu linh hoạt:
  * *Gộp thông minh (Merged - Khuyên dùng):* Giữ nguyên 100% nhãn người gán + Bổ sung các vật thể AI phát hiện bị sót.
  * *Dự đoán của AI Model (Predictions):* Đẩy toàn bộ pre-labels AI sang làm nhãn mồi.
  * *Nhãn người gán chuẩn (Annotations):* Xuất nhãn ground truth hiện có.
* Chế độ **Append / Create (Mặc định an toàn)**: Không xóa các nhãn annotator đã vẽ trên CVAT và giữ nguyên 100% video tracks.

#### 5. Menu "Công cụ khác" (Dropdown Menu Nâng cao)
Gom toàn bộ các tính năng phụ trợ, cấu hình và thử nghiệm vào menu dropdown để tránh gây rối mắt:
* **Dữ liệu & Model:**
  * `Upload Predictions JSON`: Nạp kết quả inference từ file bên ngoài.
  * `Chạy Detector Batch (Cũ)`: Chạy detector hàng loạt trên toàn bộ dataset kiểu cũ.
* **Thử nghiệm & Benchmark:**
  * `Simulate Benchmark Sandbox`: Giả lập tỷ lệ sót/sai nhãn để kiểm tra độ nhạy thuật toán.
* **Cấu hình & Hệ thống:**
  * `Cấu hình kết nối CVAT`: Đặt URL, Task ID, Job ID và Personal Access Token (PAT).
  * `Sơ đồ Kiến trúc CVAT × QC`: Mở sơ đồ kiến trúc 3 tầng tham khảo.

#### 6. Kiểm tra lại riêng Frame đang xem (Contextual Action)
* Khi bạn nhấp vào bất kỳ frame nào trong danh sách, nút **"Kiểm tra lại Frame"** sẽ hiển thị ngay bên cạnh nút **"Open in CVAT"** trong thanh tiêu đề của Frame đó, giúp bạn re-check tức thì sau khi sửa nhãn mà không cần tìm kiếm trên thanh menu chính.

---

### Bước 4: Đọc Báo cáo KPI Hiệu quả (Header Dashboard)
Ngay sau khi chạy xong, 4 thẻ KPI động sẽ hiển thị kết quả phân tích:

| Chỉ số KPI | Ý nghĩa nghiệp vụ |
| :--- | :--- |
| **Workload Reduction (%)** | Tỷ lệ phần trăm công sức review được cắt giảm. Ví dụ đạt 75% nghĩa là reviewer chỉ cần kiểm tra 25% số ảnh có nguy cơ lỗi cao. |
| **Cần Review (Flagged)** | Số lượng mẫu có nghi vấn lỗi (hiển thị kèm số lượng ảnh sạch `Clean` đã được bỏ qua). |
| **Sót vật thể (Missing)** | Tổng số đối tượng mà người gán nhãn bỏ quên nhưng Pretrained Model bắt được. |
| **Sai Class (Wrong Class)** | Tổng số đối tượng bị gán nhầm nhãn thuộc các cặp dễ nhầm lẫn giao thông. |

---

### Bước 5: Làm việc với Hàng đợi Ưu tiên (Prioritized Queue - Cột Trái)
* **Xếp hạng thông minh:** Các ảnh được sắp xếp tự động theo thứ tự giảm dần của `qc_score`. Ảnh có lỗi nguy hiểm nhất luôn nằm ở vị trí số **#1** với huy hiệu viền đỏ `HIGH`.
* **Thông tin Frame & CVAT:** Mỗi thẻ ảnh hiển thị rõ ràng nhãn **Frame #N** và nút bấm liên kết trực tiếp sang CVAT.
* **Bộ lọc đa chiều:**
  * Lọc theo mức độ nghiêm trọng: `ALL`, `HIGH`, `MEDIUM`, `LOW`, hoặc `CLEAN`.
  * Lọc theo loại lỗi: `Chỉ sót vật thể` hoặc `Chỉ sai class`.
  * Tìm kiếm tức thời theo tên file (`filename`).
* **Chọn ảnh:** Nhấp vào bất kỳ thẻ ảnh nào trong hàng đợi để chuyển dữ liệu sang khung kiểm tra trực quan ở cột bên phải.

---

### Bước 6: Đối soát Trực quan (Visual Diff) & Xử lý Lỗi (Closed-Loop Resolution)

#### 1. Khung hiển thị trực quan (Dual-Overlay Canvas) & Deep-Link Header
* **Nút CTA chính `[ ↗ Open in CVAT (Frame #N) ]`:** Được đặt nổi bật ngay cạnh tên file. Bấm nút này sẽ mở tab trình duyệt mới, đưa reviewer/annotator vào đúng frame trên CVAT Annotation Editor để sửa lỗi trực tiếp.
* **Công tắc 3 lớp đối chiếu:**
  * **[Nhãn người gán]:** Bounding box viền nét liền màu **Xanh ngọc (Teal)** — đại diện cho Ground Truth hiện tại.
  * **[Model dự đoán]:** Bounding box viền nét đứt màu **Cam (Orange)** kèm chỉ số độ tin cậy $Conf$.
  * **[Vùng nghi vấn]:** Bounding box viền màu **Đỏ nhấp nháy (Red Glow)** làm nổi bật chính xác vị trí phát hiện lỗi.

#### 2. Thẻ Bằng chứng Bất đồng & Các Lựa chọn Xử lý (Hybrid Workflow)
Bên dưới ảnh, hệ thống hiển thị chi tiết từng điểm bất đồng kèm bảng so khớp và 4 nút hành động:
* **Hành động 1 — `[ ↗ Mở trên CVAT ]`:** Mở trực tiếp frame chứa issue này trong CVAT để sửa trên System of Record.
* **Hành động 2 — `[ ✓ Đã sửa trên CVAT (Resolved) ]`:** Đánh dấu issue đã được hoàn tất chỉnh sửa trên CVAT.
* **Hành động 3 — `[ ⚡ 1-Click Sửa tự động (Demo) ]`:** Tự động sửa nhãn / thêm box trực tiếp vào DB, rất tiện lợi khi trình diễn demo cho mentor hoặc kiểm thử nhanh mà không cần bật cụm Docker CVAT.
* **Hành động 4 — `[ ✗ Báo sai (False Positive) ]`:** Bác bỏ cảnh báo sai của Pretrained Model, giữ nguyên nhãn của annotator.
* **Hành động 5 — `[ ↺ Hoàn tác ]`:** Cho phép hoàn tác mọi issue đã xử lý về trạng thái `Pending` nếu cần đánh giá lại.

---

## 7. Đặc tả Hệ thống API Endpoints

Toàn bộ các chức năng QC đều được đóng gói thành RESTful API chuẩn, dễ dàng tích hợp vào bất kỳ hệ thống CI/CD nào:

### 7.1. Chạy đối soát QC
* **`POST /api/datasets/{dataset_id}/qc/detect`**
  * *Chức năng:* Tự động kích hoạt mô hình Pretrained YOLOv8 ONNX và đối soát.
  * *Query params:* `conf_threshold` (mặc định 0.45), `iou_threshold` (mặc định 0.50).
* **`POST /api/datasets/{dataset_id}/qc/synthetic-benchmark`**
  * *Chức năng:* Sinh dự đoán giả lập benchmark để kiểm thử độ nhạy.
  * *Body (JSON):* `{"missing_ratio": 0.15, "wrong_class_ratio": 0.15}`.
* **`POST /api/datasets/{dataset_id}/qc/predictions`**
  * *Chức năng:* Nạp trực tiếp payload kết quả dự đoán của model bên ngoài.

### 7.2. Lấy dữ liệu Báo cáo & Hàng đợi
* **`GET /api/datasets/{dataset_id}/qc/report`**
  * *Chức năng:* Trả về báo cáo KPI tổng quan.
  * *Response:*
    ```json
    {
      "dataset_id": 1,
      "total_samples": 15,
      "flagged_samples": 4,
      "clean_samples": 11,
      "workload_reduction_percent": 73.33,
      "missing_object_issues": 3,
      "wrong_class_issues": 2
    }
    ```
* **`GET /api/datasets/{dataset_id}/qc/queue`**
  * *Chức năng:* Trả về hàng đợi review sắp xếp giảm dần theo điểm `qc_score`.
  * *Query params:* `include_clean=true` (tuỳ chọn bao gồm cả ảnh sạch).

### 7.3. Thao tác Sửa lỗi (Closed-loop Resolution)
* **`PUT /api/qc/issues/{issue_id}`**
  * *Chức năng:* Cập nhật trạng thái duyệt và tự động sửa Ground Truth trong CSDL.
  * *Body (JSON):*
    ```json
    {
      "status": "ACCEPTED",
      "resolution_note": "Chấp nhận đề xuất từ YOLOv8"
    }
    ```

---

## 8. Xử lý Sự cố Thường gặp (Troubleshooting)

### 8.1. Lỗi PowerShell: `File npm.ps1 cannot be loaded because running scripts is disabled`
* **Nguyên nhân:** Chính sách Execution Policy mặc định của Windows khóa thực thi file `.ps1`.
* **Khắc phục:** Mở PowerShell và chạy lệnh sau (chỉ áp dụng cho tài khoản người dùng hiện tại, không yêu cầu quyền Admin):
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
  ```
  Hoặc bạn có thể dùng lệnh `npm.cmd` thay vì `npm`:
  ```powershell
  npm.cmd run dev
  ```

### 8.2. Lỗi trùng cổng `Address already in use` (Port 8000 hoặc 3000)
* **Khắc phục:** Mở PowerShell để tìm và giải phóng tiến trình chiếm cổng:
  ```powershell
  # Giải phóng cổng 8000
  Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process -Force

  # Giải phóng cổng 3000
  Get-Process -Id (Get-NetTCPConnection -LocalPort 3000).OwningProcess | Stop-Process -Force
  ```

### 8.3. Không load được ảnh JPG khi mở trên Web
* **Nguyên nhân:** File ảnh chưa được nạp vào thư mục `uploads/` của dự án.
* **Khắc phục:** Khi thực hiện bước Upload Dataset trên giao diện Web, hãy chọn đồng thời cả file nhãn JSON và các ảnh trong thư mục `demo_data/images`. Hệ thống sẽ tự động sao chép ảnh vào đúng vị trí hiển thị.

### 8.4. Muốn làm sạch Database để kiểm tra lại từ đầu
* **Khắc phục:** Xóa file `annopilot.db` ở thư mục gốc. Khi khởi động lại backend, hệ thống sẽ tự động tạo lại database trắng tinh.
  ```powershell
  Remove-Item -Path "d:\AI\repo\AnnoPilot-qminh\annopilot.db" -Force
  ```

---

*Tài liệu được phát triển bởi Nhóm 2 — Đề tài N2-04D AnnoPilot Model-Assisted QC.*
