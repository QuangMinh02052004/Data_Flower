#!/usr/bin/env python3
"""
FLOWER ANNOTATION TOOL
Tool vẽ bounding box cho từng bông hoa để train YOLO

Cách dùng:
    python annotation_tool.py

Controls:
    - Click + Drag: Vẽ bounding box
    - 1-5: Chọn loại hoa
    - Z: Undo
    - S: Save
    - N: Next image
    - P: Previous image
    - Q: Quit
"""

import cv2
import os
import json
from pathlib import Path

# ============================================
# CẤU HÌNH
# ============================================

FLOWERS = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
COLORS = [(255,255,255), (0,200,255), (147,20,255), (0,165,255), (180,105,255)]
FLOWERS_VI = ['Cúc', 'Bồ công anh', 'Hồng', 'Hướng dương', 'Tulip']

BASE_DIR = Path(__file__).parent
DATASET_DIR = BASE_DIR / "yolo_dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"

# Global state
drawing = False
start_x, start_y = 0, 0
boxes = []  # List of (class_id, x1, y1, x2, y2)
current_class = 0
img_original = None
img_display = None


def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, boxes, img_display, img_original

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_display = img_original.copy()
        draw_all_boxes(img_display)
        cv2.rectangle(img_display, (start_x, start_y), (x, y), COLORS[current_class], 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(start_x, x), min(start_y, y)
        x2, y2 = max(start_x, x), max(start_y, y)

        if x2 - x1 > 10 and y2 - y1 > 10:
            boxes.append((current_class, x1, y1, x2, y2))
            print(f"  + {FLOWERS_VI[current_class]} at ({x1},{y1})-({x2},{y2})")

        img_display = img_original.copy()
        draw_all_boxes(img_display)


def draw_all_boxes(img):
    for cls_id, x1, y1, x2, y2 in boxes:
        color = COLORS[cls_id]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{FLOWERS_VI[cls_id]}"
        cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def save_annotation(img_path, img_shape):
    """Save annotation in YOLO format"""
    if not boxes:
        return False

    h, w = img_shape[:2]
    img_name = Path(img_path).stem

    # Copy image to dataset
    img = cv2.imread(str(img_path))
    dest_img = IMAGES_DIR / f"{img_name}.jpg"
    cv2.imwrite(str(dest_img), img)

    # Save labels in YOLO format: class x_center y_center width height (normalized)
    label_path = LABELS_DIR / f"{img_name}.txt"
    with open(label_path, 'w') as f:
        for cls_id, x1, y1, x2, y2 in boxes:
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            box_w = (x2 - x1) / w
            box_h = (y2 - y1) / h
            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")

    print(f"  Saved: {len(boxes)} boxes → {label_path.name}")
    return True


def create_dataset_structure():
    """Create YOLO dataset folders"""
    DATASET_DIR.mkdir(exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)
    LABELS_DIR.mkdir(exist_ok=True)

    # Create data.yaml
    yaml_content = f"""path: {DATASET_DIR.absolute()}
train: images
val: images

names:
  0: daisy
  1: dandelion
  2: rose
  3: sunflower
  4: tulip

nc: 5
"""
    with open(DATASET_DIR / "data.yaml", 'w') as f:
        f.write(yaml_content)

    print(f"Dataset directory: {DATASET_DIR}")


def get_all_images():
    """Get all images from flowers folder"""
    images = []
    flowers_dir = BASE_DIR / "flowers"

    for flower in FLOWERS:
        folder = flowers_dir / flower
        if folder.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                images.extend(folder.glob(ext))

    return sorted(images)


def main():
    global boxes, current_class, img_original, img_display

    print("""
╔══════════════════════════════════════════════════════════╗
║           FLOWER ANNOTATION TOOL FOR YOLO                ║
╠══════════════════════════════════════════════════════════╣
║  Click + Drag : Vẽ bounding box                          ║
║  1-5          : Chọn loại hoa                            ║
║                 1=Cúc, 2=Bồ công anh, 3=Hồng             ║
║                 4=Hướng dương, 5=Tulip                   ║
║  Z            : Undo                                     ║
║  S            : Save                                     ║
║  N            : Next image                               ║
║  P            : Previous image                           ║
║  Q            : Quit                                     ║
╚══════════════════════════════════════════════════════════╝
""")

    create_dataset_structure()

    images = get_all_images()
    if not images:
        print("Không tìm thấy ảnh trong thư mục flowers/")
        return

    print(f"Tìm thấy {len(images)} ảnh")
    print(f"Mục tiêu: Annotate 200-300 ảnh để train YOLO\n")

    # Check existing annotations
    existing = len(list(LABELS_DIR.glob("*.txt")))
    print(f"Đã annotate: {existing} ảnh\n")

    idx = 0
    window_name = "Annotation Tool"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        img_path = images[idx]
        img_original = cv2.imread(str(img_path))

        if img_original is None:
            idx = (idx + 1) % len(images)
            continue

        # Resize if too large
        h, w = img_original.shape[:2]
        max_size = 800
        if max(h, w) > max_size:
            scale = max_size / max(h, w)
            img_original = cv2.resize(img_original, (int(w*scale), int(h*scale)))

        boxes = []
        img_display = img_original.copy()

        print(f"\n[{idx+1}/{len(images)}] {img_path.name}")
        print(f"  Class: {FLOWERS_VI[current_class]} (nhấn 1-5 để đổi)")

        while True:
            # Draw info
            display = img_display.copy()
            info = f"Class: {FLOWERS_VI[current_class]} | Boxes: {len(boxes)} | [{idx+1}/{len(images)}]"
            cv2.putText(display, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            # Number keys 1-5
            if ord('1') <= key <= ord('5'):
                current_class = key - ord('1')
                print(f"  Class → {FLOWERS_VI[current_class]}")

            # Z - Undo
            elif key == ord('z'):
                if boxes:
                    removed = boxes.pop()
                    print(f"  - Removed {FLOWERS_VI[removed[0]]}")
                    img_display = img_original.copy()
                    draw_all_boxes(img_display)

            # S - Save
            elif key == ord('s'):
                if save_annotation(img_path, img_original.shape):
                    existing = len(list(LABELS_DIR.glob("*.txt")))
                    print(f"  Total annotated: {existing}")

            # N - Next
            elif key == ord('n'):
                if boxes:
                    save_annotation(img_path, img_original.shape)
                idx = (idx + 1) % len(images)
                break

            # P - Previous
            elif key == ord('p'):
                idx = (idx - 1) % len(images)
                break

            # Q - Quit
            elif key == ord('q'):
                cv2.destroyAllWindows()
                existing = len(list(LABELS_DIR.glob("*.txt")))
                print(f"\n{'='*50}")
                print(f"Tổng số ảnh đã annotate: {existing}")
                print(f"Dataset: {DATASET_DIR}")
                if existing >= 100:
                    print(f"\nĐủ data! Chạy: python train_yolo.py")
                else:
                    print(f"\nCần annotate thêm {100-existing} ảnh nữa")
                return

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
