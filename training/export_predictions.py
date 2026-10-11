import sys
import json
import glob
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
DEFAULT_WEIGHTS = BASE_DIR / "runs" / "qc_finetuned" / "weights" / "best.pt"
DEMO_IMAGES = REPO_ROOT / "demo_data" / "images"
OUTPUT_JSON = REPO_ROOT / "finetuned_predictions.json"
OUTPUT_DEMO_JSON = REPO_ROOT / "finetuned_predictions_demo.json"

def find_dataset_images_dir():
    """Tự động tìm thư mục ảnh đã giải nén của dataset trong uploads/"""
    uploads_dir = REPO_ROOT / "uploads"
    if not uploads_dir.is_dir():
        return None
    patterns = list(uploads_dir.glob("*/annotations/*/extracted/images"))
    if patterns:
        # Lấy thư mục có nhiều ảnh nhất
        best_dir = max(patterns, key=lambda p: len(list(p.glob("*.*"))))
        return best_dir
    return None

def predict_on_images(model, img_paths, conf_thresh=0.35):
    """Chạy dự đoán trên danh sách ảnh và trả về format chuẩn BDD100K JSON cho AnnoPilot"""
    payload = []
    total_boxes = 0

    print(f"[*] Đang chạy dự đoán trên {len(img_paths)} ảnh (conf={conf_thresh})...")
    # Batch predict
    batch_size = 16
    for i in range(0, len(img_paths), batch_size):
        batch = [str(p) for p in img_paths[i:i+batch_size]]
        results = model.predict(source=batch, conf=conf_thresh, verbose=False)
        for r in results:
            fname = Path(r.path).name
            labels = []
            for box in r.boxes:
                xyxy = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cat_name = model.names.get(cls_id, f"class_{cls_id}")

                labels.append({
                    "category": cat_name,
                    "confidence": round(conf, 3),
                    "box2d": {
                        "x1": round(xyxy[0], 2),
                        "y1": round(xyxy[1], 2),
                        "x2": round(xyxy[2], 2),
                        "y2": round(xyxy[3], 2)
                    }
                })
                total_boxes += 1

            payload.append({
                "name": fname,
                "labels": labels
            })
    return payload, total_boxes

def export_predictions_for_annopilot(weights_path=None, conf_thresh=0.35):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[!] Thiếu thư viện ultralytics.")
        return

    w_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS
    if not w_path.is_file():
        print(f"[!] Không tìm thấy {w_path}, sử dụng yolov8n.pt thay thế...")
        w_path = "yolov8n.pt"

    print("=" * 60)
    print(" XUẤT PREDICTIONS TỪ MODEL ĐÃ FINE-TUNE CHO ANNOPILOT")
    print("=" * 60)
    print(f"[*] Trọng số mô hình: {w_path}")
    model = YOLO(str(w_path))

    # 1. Dự đoán trên Dataset thực tế trong uploads (frame_000000.PNG ... frame_000359.PNG)
    dataset_img_dir = find_dataset_images_dir()
    if dataset_img_dir:
        print(f"\n[1] Tìm thấy thư mục ảnh của Dataset thực tế: {dataset_img_dir}")
        img_paths = sorted(list(dataset_img_dir.glob("*.PNG")) + list(dataset_img_dir.glob("*.png")) + list(dataset_img_dir.glob("*.jpg")))
        if img_paths:
            payload, total_boxes = predict_on_images(model, img_paths, conf_thresh=conf_thresh)
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"[✓] Đã tạo file: {OUTPUT_JSON}")
            print(f"    - Khớp với: Dataset '{dataset_img_dir.parents[3].name}' ({len(payload)} frames)")
            print(f"    - Tổng Bounding Boxes dự đoán: {total_boxes}")

    # 2. Dự đoán trên demo_data/images (nếu có)
    if DEMO_IMAGES.is_dir():
        demo_imgs = sorted(list(DEMO_IMAGES.glob("*.jpg")) + list(DEMO_IMAGES.glob("*.png")))
        if demo_imgs:
            print(f"\n[2] Tìm thấy demo_data: {DEMO_IMAGES} ({len(demo_imgs)} frames)")
            payload_demo, demo_boxes = predict_on_images(model, demo_imgs, conf_thresh=conf_thresh)
            with open(OUTPUT_DEMO_JSON, "w", encoding="utf-8") as f:
                json.dump(payload_demo, f, ensure_ascii=False, indent=2)
            print(f"[✓] Đã tạo file demo: {OUTPUT_DEMO_JSON}")

    print("\n" + "=" * 60)
    print("[✓] XUẤT PREDICTIONS HOÀN TẤT THÀNH CÔNG!")
    print(f"👉 File chính để nạp vào AnnoPilot: {OUTPUT_JSON}")
    print("👉 Hướng dẫn: Mở Web AnnoPilot -> Chọn Dataset -> Vào 'Model QC'")
    print("              Bấm nút 'Upload Predictions JSON' -> Chọn file 'finetuned_predictions.json'")
    print("=" * 60)

if __name__ == "__main__":
    export_predictions_for_annopilot()
