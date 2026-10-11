import sys
from pathlib import Path
import torch
import torch.nn as nn

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "dataset" / "data.yaml"
INIT_WEIGHTS = BASE_DIR / "yolov8n_bdd100k_init.pt"

def build_warmstart_weights():
    """
    Chuyển giao trọng số từ COCO (yolov8n.pt 80 classes) sang 10 classes BDD100K:
    0: pedestrian <- person (0)
    1: rider      <- person (0)
    2: car        <- car (2)
    3: truck      <- truck (7)
    4: bus        <- bus (5)
    5: train      <- train (6)
    6: motorcycle <- motorcycle (3)
    7: bicycle    <- bicycle (1)
    8: traffic light <- traffic light (9)
    9: traffic sign  <- stop sign (11)
    Giúp model có khả năng phát hiện chuẩn xác ngay lập tức mà không bị random weights khi đổi số lớp!
    """
    from ultralytics import YOLO

    print("[*] Đang khởi tạo trọng số Transfer Learning (COCO -> BDD100K)...")
    base_model = YOLO("yolov8n.pt")
    head = base_model.model.model[-1]
    mapping = [0, 0, 2, 7, 5, 6, 3, 1, 9, 11]

    for i in range(len(head.cv3)):
        old_conv = head.cv3[i][-1]
        new_conv = nn.Conv2d(old_conv.in_channels, len(mapping), kernel_size=1)
        new_conv.weight.data = old_conv.weight.data[mapping].clone()
        new_conv.bias.data = old_conv.bias.data[mapping].clone()
        head.cv3[i][-1] = new_conv

    head.nc = 10
    names = {
        0: 'pedestrian', 1: 'rider', 2: 'car', 3: 'truck', 4: 'bus',
        5: 'train', 6: 'motorcycle', 7: 'bicycle', 8: 'traffic light', 9: 'traffic sign'
    }
    base_model.model.names = names

    ckpt = {
        'model': base_model.model,
        'train_args': {},
        'epoch': -1,
        'updates': None
    }
    torch.save(ckpt, INIT_WEIGHTS)
    print(f"[✓] Đã tạo thành công Warm-start weights tại: {INIT_WEIGHTS}")
    return INIT_WEIGHTS

def run_fine_tuning(epochs=25, imgsz=640, batch=2):
    print("=" * 60)
    print(" BẮT ĐẦU QUY TRÌNH FINE-TUNE MODEL YOLOV8 CHO ANNOTATION QC")
    print("=" * 60)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[!] Đang thiếu thư viện ultralytics hoặc torch.")
        return

    if not INIT_WEIGHTS.is_file():
        build_warmstart_weights()

    device = "0" if torch.cuda.is_available() else "cpu"
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[*] Thiết bị huấn luyện: {device} ({device_name})")
    print(f"[*] Trọng số khởi đầu: {INIT_WEIGHTS}")
    print(f"[*] File cấu hình dữ liệu: {DATA_YAML}")

    model = YOLO(str(INIT_WEIGHTS))

    print(f"[*] Bắt đầu huấn luyện tinh chỉnh ({epochs} epochs, batch={batch}, imgsz={imgsz})...")
    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        lr0=0.001,
        lrf=0.01,
        cls=1.2,
        freeze=10,
        project=str(BASE_DIR / "runs"),
        name="qc_finetuned",
        workers=0,
        exist_ok=True,
        plots=True,
        verbose=True
    )

    best_weight = BASE_DIR / "runs" / "qc_finetuned" / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print(f"[✓] HUẤN LUYỆN HOÀN TẤT THÀNH CÔNG!")
    print(f"[✓] Trọng số tốt nhất đã lưu tại: {best_weight}")
    print("=" * 60)
    return best_weight

if __name__ == "__main__":
    run_fine_tuning()
