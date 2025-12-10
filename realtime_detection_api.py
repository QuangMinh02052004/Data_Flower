"""
Realtime Flower Detection API
- Multi-flower detection in single frame
- Bounding boxes with labels
- WebSocket streaming for realtime tracking
- Flower counting by type
"""

import asyncio
import base64
import io
import json
import os
import time
from contextlib import asynccontextmanager
from typing import List, Dict, Any
from collections import Counter

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras

# Global variables
classifier_model = None
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
vietnamese_names = {
    'daisy': 'Cúc',
    'dandelion': 'Bồ công anh',
    'rose': 'Hồng',
    'sunflower': 'Hướng dương',
    'tulip': 'Tulip'
}

# Detection colors for each flower type
detection_colors = {
    'daisy': '#FFD700',      # Gold
    'dandelion': '#FFA500',  # Orange
    'rose': '#FF1493',       # Deep Pink
    'sunflower': '#FFD700',  # Gold
    'tulip': '#FF6347'       # Tomato
}


def load_model():
    """Load classification model"""
    global classifier_model
    model_path = os.path.join(os.path.dirname(__file__), 'flower_classifier_final.keras')

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}...")
        classifier_model = keras.models.load_model(model_path)
        print("Model loaded successfully!")
    else:
        print(f"WARNING: Model not found at {model_path}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    print("Shutting down...")


app = FastAPI(
    title="Realtime Flower Detection API",
    description="Multi-flower detection with bounding boxes",
    version="3.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_for_classification(image: np.ndarray) -> np.ndarray:
    """Preprocess image region for classification"""
    img = cv2.resize(image, (224, 224))
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def is_likely_skin(region: np.ndarray) -> bool:
    """
    Detect if region is likely human skin to avoid false positives.
    Skin has low saturation, specific hue range, uniform texture.
    """
    if region.size == 0:
        return False

    small = cv2.resize(region, (32, 32))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Skin tone: hue 0-25, saturation 20-80, value 80-255
    skin_mask = (h <= 25) & (s >= 20) & (s <= 80) & (v >= 80)
    skin_ratio = np.mean(skin_mask)

    # If >40% looks like skin AND low color variance, it's probably skin
    if skin_ratio > 0.4:
        # Check texture uniformity (skin is smoother than flowers)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        variance = np.var(gray)
        if variance < 800:  # Low variance = uniform = likely skin
            return True

    return False


def has_green_nearby(image: np.ndarray, bbox: tuple, margin: int = 20) -> bool:
    """
    Check if there's green (leaves) near the bounding box.
    Flowers usually have leaves nearby.
    """
    h, w = image.shape[:2]
    x, y, bw, bh = bbox

    # Expand region to check for green
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(w, x + bw + margin)
    y2 = min(h, y + bh + margin)

    region = image[y1:y2, x1:x2]
    if region.size == 0:
        return False

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    # Green hue: 35-85
    green_mask = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] > 30)
    green_ratio = np.mean(green_mask)

    return green_ratio > 0.05  # At least 5% green


def classify_region(region: np.ndarray, min_confidence: float = 50.0) -> tuple:
    """Classify a single region with skin detection"""
    if classifier_model is None:
        return None, 0.0

    # Skip if looks like skin
    if is_likely_skin(region):
        return None, 0.0

    img = preprocess_for_classification(region)
    predictions = classifier_model.predict(img, verbose=0)
    class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][class_idx]) * 100

    if confidence >= min_confidence:
        return class_names[class_idx], confidence
    return None, confidence


def find_flower_contours(image: np.ndarray) -> List[tuple]:
    """
    Find individual flower regions - SIMPLE & FAST version.
    Tách từng bông hoa bằng connected components + watershed đơn giản.
    """
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # ========== PHÁT HIỆN MÀU HOA ==========
    pink_red = (((hue <= 15) | (hue >= 150)) & (sat >= 40) & (val >= 40))
    yellow = ((hue >= 12) & (hue <= 50) & (sat >= 35) & (val >= 60))
    white = ((sat <= 70) & (val >= 160))
    purple = ((hue >= 115) & (hue <= 165) & (sat >= 30) & (val >= 35))

    flower_mask = (pink_red | yellow | white | purple).astype(np.uint8) * 255

    # Loại bỏ màu xanh lá
    green_mask = ((hue >= 28) & (hue <= 95) & (sat >= 25))
    flower_mask[green_mask] = 0

    # ========== MORPHOLOGY ==========
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    flower_mask = cv2.morphologyEx(flower_mask, cv2.MORPH_OPEN, kernel)

    # ========== DISTANCE TRANSFORM + THRESHOLD ==========
    dist = cv2.distanceTransform(flower_mask, cv2.DIST_L2, 5)

    if dist.max() == 0:
        return []

    # Threshold THẤP để tìm nhiều tâm hoa
    _, sure_fg = cv2.threshold(dist, 0.2 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    # Connected components để đếm số hoa
    num_labels, markers = cv2.connectedComponents(sure_fg)

    # ========== EXTRACT BOUNDING BOXES ==========
    regions = []
    min_area = (h * w) * 0.002
    max_area = (h * w) * 0.3

    for label in range(1, num_labels):
        label_mask = (markers == label).astype(np.uint8) * 255

        # Dilate để mở rộng vùng hoa
        label_mask = cv2.dilate(label_mask, kernel, iterations=3)

        # Giới hạn trong flower_mask
        label_mask = cv2.bitwise_and(label_mask, flower_mask)

        contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                x, y, bw, bh = cv2.boundingRect(contour)

                # Padding
                pad = int(min(bw, bh) * 0.15)
                x = max(0, x - pad)
                y = max(0, y - pad)
                bw = min(w - x, bw + 2 * pad)
                bh = min(h - y, bh + 2 * pad)

                if bw >= 10 and bh >= 10:
                    regions.append((x, y, bw, bh))

    # ========== GRID DETECTION cho bó hoa dày ==========
    flower_ratio = flower_mask.sum() / (255 * h * w)

    if len(regions) < 3 and flower_ratio > 0.2:
        # Chia grid 4x4
        grid_size = 4
        cell_h, cell_w = h // grid_size, w // grid_size

        for row in range(grid_size):
            for col in range(grid_size):
                y1, y2 = row * cell_h, min((row + 1) * cell_h, h)
                x1, x2 = col * cell_w, min((col + 1) * cell_w, w)

                cell_mask = flower_mask[y1:y2, x1:x2]
                cell_ratio = cell_mask.sum() / (255 * cell_mask.size) if cell_mask.size > 0 else 0

                if cell_ratio > 0.25:
                    # Tìm contour trong cell
                    cell_contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in cell_contours:
                        if cv2.contourArea(cnt) > (cell_h * cell_w) * 0.1:
                            cx, cy, cw, ch = cv2.boundingRect(cnt)
                            regions.append((x1 + cx, y1 + cy, cw, ch))

    # Loại bỏ boxes trùng
    regions = remove_overlapping_boxes(regions)

    return regions


def remove_overlapping_boxes(boxes: List[tuple], iou_threshold: float = 0.4) -> List[tuple]:
    """Loại bỏ boxes overlap"""
    if len(boxes) <= 1:
        return boxes

    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
    keep = []

    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [b for b in boxes if box_iou(best, b) < iou_threshold]

    return keep


def box_iou(box1: tuple, box2: tuple) -> float:
    """Tính IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    inter = (x2 - x1) * (y2 - y1)
    union = box1[2] * box1[3] + box2[2] * box2[3] - inter
    return inter / union if union > 0 else 0.0


def selective_search_detection(image: np.ndarray, max_regions: int = 50) -> List[Dict]:
    """
    Detect multiple flowers using color segmentation + contour detection.
    Focus on individual flowers for accurate counting.
    """
    if classifier_model is None:
        return []

    h, w = image.shape[:2]
    detections = []

    # Step 1: Find flower regions using color segmentation
    flower_regions = find_flower_contours(image)

    # Step 2: Classify each detected region
    for (rx, ry, rw, rh) in flower_regions:
        region = image[ry:ry+rh, rx:rx+rw]
        if region.size == 0:
            continue

        # Skip if looks like skin
        if is_likely_skin(region):
            continue

        flower_type, confidence = classify_region(region, min_confidence=55.0)

        if flower_type and confidence >= 55.0:
            bbox = (rx, ry, rw, rh)
            green_bonus = 3.0 if has_green_nearby(image, bbox) else 0.0

            detections.append({
                'class': flower_type,
                'class_vi': vietnamese_names.get(flower_type, flower_type),
                'confidence': round(confidence + green_bonus, 1),
                'bbox': {'x': rx, 'y': ry, 'width': rw, 'height': rh},
                'color': detection_colors.get(flower_type, '#00FF00')
            })

    # Step 3: If no contours found, fall back to grid detection
    if len(detections) == 0:
        grid_sizes = [2, 3]
        for grid_size in grid_sizes:
            cell_h = h // grid_size
            cell_w = w // grid_size

            for row in range(grid_size):
                for col in range(grid_size):
                    y1 = max(0, row * cell_h - cell_h // 5)
                    y2 = min(h, (row + 1) * cell_h + cell_h // 5)
                    x1 = max(0, col * cell_w - cell_w // 5)
                    x2 = min(w, (col + 1) * cell_w + cell_w // 5)

                    region = image[y1:y2, x1:x2]
                    if region.size == 0 or is_likely_skin(region):
                        continue

                    flower_type, confidence = classify_region(region, min_confidence=60.0)

                    if flower_type and confidence >= 60.0:
                        detections.append({
                            'class': flower_type,
                            'class_vi': vietnamese_names.get(flower_type, flower_type),
                            'confidence': round(confidence, 1),
                            'bbox': {'x': x1, 'y': y1, 'width': x2-x1, 'height': y2-y1},
                            'color': detection_colors.get(flower_type, '#00FF00')
                        })

    # Apply NMS to remove overlapping boxes
    detections = non_max_suppression(detections, iou_threshold=0.35)

    # Sort by confidence and limit
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)[:max_regions]

    return detections


def multi_scale_detection(image: np.ndarray) -> List[Dict]:
    """
    Alternative detection using contours + sliding window hybrid.
    Good for detecting multiple flower types.
    """
    if classifier_model is None:
        return []

    h, w = image.shape[:2]
    detections = []

    # First try contour-based detection
    flower_regions = find_flower_contours(image)

    for (rx, ry, rw, rh) in flower_regions:
        region = image[ry:ry+rh, rx:rx+rw]
        if region.size == 0 or is_likely_skin(region):
            continue

        flower_type, confidence = classify_region(region, min_confidence=55.0)

        if flower_type:
            detections.append({
                'class': flower_type,
                'class_vi': vietnamese_names.get(flower_type, flower_type),
                'confidence': round(confidence, 1),
                'bbox': {'x': rx, 'y': ry, 'width': rw, 'height': rh},
                'color': detection_colors.get(flower_type, '#00FF00')
            })

    # Supplement with sliding window if few detections
    if len(detections) < 3:
        window_ratios = [0.3, 0.45]
        step_ratio = 0.3

        for ratio in window_ratios:
            window_size = int(min(h, w) * ratio)
            if window_size < 64:
                continue

            step = int(window_size * step_ratio)

            for y in range(0, h - window_size + 1, step):
                for x in range(0, w - window_size + 1, step):
                    window = image[y:y+window_size, x:x+window_size]

                    if is_likely_skin(window):
                        continue

                    flower_type, confidence = classify_region(window, min_confidence=60.0)

                    if flower_type:
                        detections.append({
                            'class': flower_type,
                            'class_vi': vietnamese_names.get(flower_type, flower_type),
                            'confidence': round(confidence, 1),
                            'bbox': {'x': x, 'y': y, 'width': window_size, 'height': window_size},
                            'color': detection_colors.get(flower_type, '#00FF00')
                        })

    # NMS
    detections = non_max_suppression(detections, iou_threshold=0.3)

    return sorted(detections, key=lambda x: x['confidence'], reverse=True)[:20]


def non_max_suppression(detections: List[Dict], iou_threshold: float = 0.3) -> List[Dict]:
    """Remove overlapping detections"""
    if not detections:
        return []

    # Sort by confidence
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

    keep = []
    while detections:
        best = detections.pop(0)
        keep.append(best)

        remaining = []
        for det in detections:
            if compute_iou(best['bbox'], det['bbox']) < iou_threshold:
                remaining.append(det)
        detections = remaining

    return keep


def compute_iou(box1: Dict, box2: Dict) -> float:
    """Compute IoU between two boxes"""
    x1 = max(box1['x'], box2['x'])
    y1 = max(box1['y'], box2['y'])
    x2 = min(box1['x'] + box1['width'], box2['x'] + box2['width'])
    y2 = min(box1['y'] + box1['height'], box2['y'] + box2['height'])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = box1['width'] * box1['height']
    area2 = box2['width'] * box2['height']
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


# ============= API Endpoints =============

@app.get("/")
async def root():
    return {
        "message": "Realtime Flower Detection API v3.0",
        "endpoints": {
            "/analyze": "POST - Single flower classification",
            "/detect": "POST - Multi-flower detection with bboxes",
            "/ws/realtime": "WebSocket - Realtime detection stream"
        }
    }


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Single flower classification (backward compatible)"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_array = np.array(image)
        flower_type, confidence = classify_region(img_array, min_confidence=30.0)

        if flower_type:
            return {
                "success": True,
                "flower_type": vietnamese_names.get(flower_type, flower_type),
                "flower_type_en": flower_type,
                "confidence": round(confidence, 2),
                "dominant_colors": [],
                "message": f"Phát hiện hoa {vietnamese_names.get(flower_type, flower_type)}"
            }
        else:
            return {
                "success": True,
                "flower_type": "Không xác định",
                "confidence": round(confidence, 2),
                "dominant_colors": [],
                "message": "Không nhận diện được loại hoa"
            }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})


@app.post("/detect")
async def detect_flowers(file: UploadFile = File(...)):
    """Multi-flower detection with bounding boxes"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != 'RGB':
            image = image.convert('RGB')

        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Use grid-based detection for speed
        detections = selective_search_detection(img_bgr)

        # Count by type
        counts = Counter([d['class_vi'] for d in detections])

        # Get main flower (highest confidence)
        main_flower = None
        main_confidence = 0
        if detections:
            best = max(detections, key=lambda x: x['confidence'])
            main_flower = best['class_vi']
            main_confidence = best['confidence']

        return {
            "success": True,
            "detections": detections,
            "counts": dict(counts),
            "total_flowers": len(detections),
            "main_flower": main_flower,
            "main_confidence": main_confidence,
            "image_size": {"width": image.width, "height": image.height}
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})


@app.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    """
    WebSocket for realtime detection
    Receives base64 frames, returns detections with bboxes
    """
    await websocket.accept()
    print("WebSocket client connected for realtime detection")

    frame_count = 0
    last_detections = []

    try:
        while True:
            data = await websocket.receive_text()

            try:
                message = json.loads(data)

                if message.get('type') == 'frame':
                    frame_count += 1

                    # Decode image
                    image_data = message.get('data', '')
                    if ',' in image_data:
                        image_data = image_data.split(',')[1]

                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes))

                    if image.mode != 'RGB':
                        image = image.convert('RGB')

                    img_array = np.array(image)
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                    # Run detection (every frame for smooth tracking)
                    detections = selective_search_detection(img_bgr)

                    # Count flowers
                    counts = Counter([d['class_vi'] for d in detections])

                    # Get main flower
                    main_flower = None
                    main_confidence = 0
                    if detections:
                        best = max(detections, key=lambda x: x['confidence'])
                        main_flower = best['class_vi']
                        main_confidence = best['confidence']

                    last_detections = detections

                    # Send response
                    await websocket.send_json({
                        "type": "detection",
                        "success": True,
                        "detections": detections,
                        "counts": dict(counts),
                        "total_flowers": len(detections),
                        "main_flower": main_flower,
                        "main_confidence": main_confidence,
                        "frame": frame_count,
                        "image_size": {"width": image.width, "height": image.height}
                    })

                elif message.get('type') == 'ping':
                    await websocket.send_json({"type": "pong"})

            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "message": "Invalid JSON"})
            except Exception as e:
                await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        print("WebSocket client disconnected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
