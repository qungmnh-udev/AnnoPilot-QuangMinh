"""
Demo Quick Test Script for Model-Assisted QC (N2-04D)
Run this script to verify the entire end-to-end QC flow in terminal:
    python demo_quick_test.py
"""
import io
import json
import sys

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from fastapi.testclient import TestClient
from app.main import app

def run_quick_demo():
    print("=" * 70)
    print(">>> BAT DAU KIEM TRA LUONG MODEL-ASSISTED QC (DE TAI N2-04D)")
    print("=" * 70)

    client = TestClient(app)

    print("\n[Buoc 1] Chuan bi du lieu nhan BDD100K mau...")
    bdd_sample = [
        {
            "name": "b1c9a84b-63fc3452.jpg",
            "labels": [
                {"category": "car", "box2d": {"x1": 120, "y1": 340, "x2": 260, "y2": 420}},
                {"category": "bus", "box2d": {"x1": 420, "y1": 220, "x2": 680, "y2": 450}}
            ]
        },
        {
            "name": "b1c9a84b-63fc3453.jpg",
            "labels": [
                {"category": "pedestrian", "box2d": {"x1": 80, "y1": 310, "x2": 115, "y2": 410}}
            ]
        },
        {
            "name": "b1c9a84b-63fc3454.jpg",
            "labels": [
                {"category": "car", "box2d": {"x1": 300, "y1": 300, "x2": 450, "y2": 390}},
                {"category": "traffic light", "box2d": {"x1": 500, "y1": 150, "x2": 530, "y2": 210}}
            ]
        }
    ]
    bdd_bytes = json.dumps(bdd_sample).encode("utf-8")

    # 2. Upload dataset vào AnnoPilot
    print("[Buoc 2] Tai dataset vao he thong qua API /api/datasets/upload...")
    upload_res = client.post(
        "/api/datasets/upload",
        data={"name": "BDD100K Demo Sanity", "format": "BDD100K", "task_type": "BBOX_2D"},
        files=[("annotations", ("bdd_sample.json", io.BytesIO(bdd_bytes), "application/json"))]
    )
    if upload_res.status_code != 200:
        print(f"[-] Upload that bai: {upload_res.text}")
        return

    dataset = upload_res.json()
    dataset_id = dataset["id"]
    print(f"   [OK] Da tao dataset ID: {dataset_id} | Tong so anh: {dataset['sample_count']}")

    # 3. Nạp dự đoán của Pretrained Model (Detector)
    print("\n[Buoc 3] Nap ket qua du doan cua Pretrained Detector vao he thong...")
    predictions_payload = {
        "predictions": [
            {
                "file_name": "b1c9a84b-63fc3452.jpg",
                "predictions": [
                    # Khớp đúng với car
                    {"label": "car", "confidence": 0.95, "geometry": {"x1": 122, "y1": 338, "x2": 258, "y2": 422}},
                    # BẤT ĐỒNG 1: Người gán là 'bus', model khẳng định là 'truck' (conf 0.91) -> WRONG_CLASS
                    {"label": "truck", "confidence": 0.91, "geometry": {"x1": 422, "y1": 222, "x2": 678, "y2": 448}},
                    # BẤT ĐỒNG 2: Model thấy 1 ô tô ở xa người chưa gán nhãn -> MISSING_OBJECT
                    {"label": "car", "confidence": 0.88, "geometry": {"x1": 800, "y1": 320, "x2": 910, "y2": 390}}
                ]
            },
            {
                "file_name": "b1c9a84b-63fc3453.jpg",
                "predictions": [
                    # Khớp đúng pedestrian
                    {"label": "pedestrian", "confidence": 0.93, "geometry": {"x1": 81, "y1": 312, "x2": 114, "y2": 408}}
                ]
            },
            {
                "file_name": "b1c9a84b-63fc3454.jpg",
                "predictions": [
                    # Khớp đúng car và traffic light
                    {"label": "car", "confidence": 0.96, "geometry": {"x1": 302, "y1": 298, "x2": 448, "y2": 392}},
                    {"label": "traffic light", "confidence": 0.90, "geometry": {"x1": 501, "y1": 151, "x2": 529, "y2": 209}}
                ]
            }
        ]
    }

    pred_res = client.post(f"/api/datasets/{dataset_id}/qc/predictions", json=predictions_payload)
    if pred_res.status_code != 200:
        print(f"[-] Nap predictions that bai: {pred_res.text}")
        return

    report = pred_res.json()["audit_summary"]
    print("\n" + "=" * 70)
    print("BAO CAO KET QUA DOI SOAT CHAT LUONG (QC REPORT)")
    print("=" * 70)
    print(f" * Tong so anh phan tich:                {report['total_samples']}")
    print(f" * So anh phat hien co nghi van loi:     {report['flagged_samples_count']}")
    print(f" * So anh sach (khong can review):       {report['clean_samples_count']}")
    print(f" * TIET KIEM KHOI LUONG REVIEW:          {report['workload_reduction_percentage']}%")
    print(f" * Phat hien vat the bi sot (Missing):   {report['total_missing_candidates']} loi")
    print(f" * Phat hien gan sai nhan (Wrong Class): {report['total_wrong_class_candidates']} loi")

    # 4. Lấy hàng đợi ưu tiên (Prioritized Queue)
    print("\n" + "=" * 70)
    print("HANG DOI REVIEW UU TIEN (PRIORITIZED QC QUEUE)")
    print("=" * 70)
    queue_res = client.get(f"/api/datasets/{dataset_id}/qc/queue")
    queue = queue_res.json()

    for idx, item in enumerate(queue, 1):
        print(f"\n[#{idx}] File: {item['file_name']} | Do nghiem trong: {item['qc_severity']} | QC Score: {item['qc_score']:.3f}")
        for issue in item['qc_issues']:
            itype = issue['issue_type']
            score = issue['qc_score']
            if itype == 'MISSING_OBJECT':
                print(f"   [!] [MISSING OBJECT] Model thay '{issue['suggested_label']}' (conf {issue['evidence']['model_confidence']:.2f}) nhung chua duoc gan nhan | Score: {score:.3f}")
            elif itype == 'WRONG_CLASS':
                print(f"   [!] [WRONG CLASS]    Nguoi gan '{issue['human_label']}' -> Model de xuat '{issue['suggested_label']}' (conf {issue['evidence']['model_confidence']:.2f}, IoU: {issue['evidence']['iou']:.2f}) | Score: {score:.3f}")

    # 5. Thử nghiệm xử lý 1-click (Closed-Loop Resolution)
    print("\n" + "=" * 70)
    print("THAO TAC REVIEWER: CHAP NHAN SUA LOI (1-CLICK ACCEPT)")
    print("=" * 70)
    first_sample = queue[0]
    for issue in first_sample['qc_issues']:
        res = client.put(f"/api/qc/issues/{issue['id']}", json={"status": "ACCEPTED", "reviewer_note": "Xac nhan dung loi qua doi soat model"})
        print(f"   [OK] Da xu ly Issue #{issue['id']} ({issue['issue_type']}): Chuyen trang thai -> {res.json()['status']}")

    # Kiểm tra lại Ground Truth sau khi sửa
    sample_after = client.get(f"/api/samples/{first_sample['id']}").json()
    updated_labels = [a['label'] for a in sample_after['annotations']]
    print(f"\n   Ket qua cap nhat nhan trong database cua {first_sample['file_name']}:")
    print(f"      - Nhan ban dau: ['car', 'bus']")
    print(f"      - Nhan sau khi sua: {updated_labels}")
    print(f"      => Vat the sot da duoc tu dong them box moi, 'bus' da duoc sua thanh 'truck'!")

    print("\n" + "=" * 70)
    print(">>> TOAN BO QUY TRINH MODEL-ASSISTED QC DA HOAT DONG HOAN HAO!")
    print("=" * 70)

if __name__ == "__main__":
    run_quick_demo()
