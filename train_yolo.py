#!/usr/bin/env python3
"""
YOLO TRAINING SCRIPT
Train YOLOv8 để detect hoa với bounding box chính xác

Cách dùng:
    python train_yolo.py

Yêu cầu:
    - Đã annotate ít nhất 100 ảnh bằng annotation_tool.py
    - Dataset ở thư mục yolo_dataset/
"""

from ultralytics import YOLO
from pathlib import Path
import shutil

# ============================================
# CẤU HÌNH
# ============================================

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "yolo_dataset"
MODEL_OUTPUT = BASE_DIR / "flower_yolo.pt"

# Training config - tối ưu cho M2 16GB
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 640
DEVICE = 'mps'  # Apple Silicon GPU


def check_dataset():
    """Kiểm tra dataset đã sẵn sàng chưa"""
    images_dir = DATASET_DIR / "images"
    labels_dir = DATASET_DIR / "labels"
    data_yaml = DATASET_DIR / "data.yaml"

    if not images_dir.exists() or not labels_dir.exists():
        print("❌ Chưa có dataset!")
        print("   Chạy: python annotation_tool.py")
        return False

    num_images = len(list(images_dir.glob("*.jpg")))
    num_labels = len(list(labels_dir.glob("*.txt")))

    print(f"Dataset: {DATASET_DIR}")
    print(f"  Images: {num_images}")
    print(f"  Labels: {num_labels}")

    if num_labels < 50:
        print(f"\n⚠️  Cần annotate thêm! Hiện có {num_labels}, khuyến nghị 100+")
        print("   Chạy: python annotation_tool.py")
        return False

    if not data_yaml.exists():
        print("❌ Thiếu data.yaml!")
        return False

    print("✅ Dataset sẵn sàng!")
    return True


def train():
    """Train YOLOv8"""
    print("""
╔══════════════════════════════════════════════════════════╗
║              YOLO FLOWER TRAINING                        ║
╠══════════════════════════════════════════════════════════╣
║  Model   : YOLOv8n (nano - fast)                         ║
║  Device  : Apple M2 GPU (MPS)                            ║
║  Epochs  : 100                                           ║
║  Output  : flower_yolo.pt                                ║
╚══════════════════════════════════════════════════════════╝
""")

    if not check_dataset():
        return

    print("\n" + "="*50)
    print("BẮT ĐẦU TRAINING")
    print("="*50 + "\n")

    # Load pretrained model
    model = YOLO('yolov8n.pt')

    # Train
    results = model.train(
        data=str(DATASET_DIR / "data.yaml"),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(BASE_DIR / "runs"),
        name="flower_train",
        exist_ok=True,

        # Augmentation
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        degrees=15,
        translate=0.1,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # Early stopping
        patience=20,

        # Save
        save=True,
        save_period=10,
    )

    # Copy best model
    best_model = BASE_DIR / "runs" / "flower_train" / "weights" / "best.pt"
    if best_model.exists():
        shutil.copy(best_model, MODEL_OUTPUT)
        print(f"\n✅ Model saved: {MODEL_OUTPUT}")
        print("\nChạy API: python detection_api.py")
    else:
        print("\n❌ Training failed!")

    return results


if __name__ == "__main__":
    train()
