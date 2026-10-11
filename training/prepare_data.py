import os
import sys
import shutil
import zipfile
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
ZIP_PATH = Path(r"C:\Users\vumin\Downloads\train_data.zip")

def prepare_dataset():
    if not ZIP_PATH.is_file():
        raise FileNotFoundError(f"Không tìm thấy file zip: {ZIP_PATH}")

    # Tạo các thư mục
    images_train = DATA_DIR / "images" / "train"
    images_val = DATA_DIR / "images" / "val"
    labels_train = DATA_DIR / "labels" / "train"
    labels_val = DATA_DIR / "labels" / "val"

    for p in [images_train, images_val, labels_train, labels_val]:
        p.mkdir(parents=True, exist_ok=True)

    print(f"[*] Đang giải nén {ZIP_PATH}...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        namelist = z.namelist()
        
        # Tìm tất cả các file jpg/png trong obj_train_data/
        img_entries = [n for n in namelist if n.startswith("obj_train_data/") and n.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        print(f"[*] Tìm thấy {len(img_entries)} ảnh trong file zip.")
        
        for idx, img_entry in enumerate(img_entries):
            # Tạo tên file sạch không có khoảng trắng
            clean_name = f"frame_{idx:04d}"
            img_ext = Path(img_entry).suffix
            
            # Đọc ảnh
            img_data = z.read(img_entry)
            
            # Đọc txt nhãn tương ứng
            txt_entry = str(Path(img_entry).with_suffix('.txt')).replace('\\', '/')
            if txt_entry in namelist:
                txt_data = z.read(txt_entry)
            else:
                txt_data = b""

            # Lưu vào images/train và labels/train
            with open(images_train / f"{clean_name}{img_ext}", "wb") as f:
                f.write(img_data)
            with open(labels_train / f"{clean_name}.txt", "wb") as f:
                f.write(txt_data)

            # Copy mẫu cuối vào val để pass validation của YOLO
            if idx == len(img_entries) - 1:
                with open(images_val / f"{clean_name}{img_ext}", "wb") as f:
                    f.write(img_data)
                with open(labels_val / f"{clean_name}.txt", "wb") as f:
                    f.write(txt_data)
            elif idx == 0:
                with open(images_val / f"{clean_name}{img_ext}", "wb") as f:
                    f.write(img_data)
                with open(labels_val / f"{clean_name}.txt", "wb") as f:
                    f.write(txt_data)

    # Đọc obj.names nếu có
    names = [
        "pedestrian", "rider", "car", "truck", "bus",
        "train", "motorcycle", "bicycle", "traffic light", "traffic sign"
    ]
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        if "obj.names" in z.namelist():
            lines = [line.strip() for line in z.read("obj.names").decode('utf-8', errors='ignore').splitlines() if line.strip()]
            if lines:
                names = lines

    # Tạo data.yaml
    yaml_content = f"""path: {DATA_DIR.as_posix()}
train: images/train
val: images/val

names:
"""
    for i, name in enumerate(names):
        yaml_content += f"  {i}: {name}\n"

    yaml_file = DATA_DIR / "data.yaml"
    with open(yaml_file, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"[✓] Đã tạo thành công bộ dữ liệu tại: {DATA_DIR}")
    print(f"[✓] File cấu hình: {yaml_file}")

if __name__ == "__main__":
    prepare_dataset()
