"""
Magnolia Plant Counter — Web Interface
Serves a browser UI for uploading drone images and counting plants.

Usage:
    pip install flask ultralytics numpy Pillow
    python app.py

Then open http://localhost:5000 in your browser.
"""

import os
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from ultralytics import YOLO

app = Flask(__name__, static_folder=".", static_url_path="")

# ── Config ──────────────────────────────────────────────────
MODEL_PATH      = "best.pt"
TILE_SIZE       = 1024
OVERLAP         = 205
CONF            = 0.5
NMS_IOU         = 0.3
MIN_CENTER_DIST = 300

# Load model once at startup
print(f"Loading model: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("Model loaded.")


# ── Tiling + inference (same logic as predict.py) ───────────

def tile_image(img_path, tile_size=1024, overlap=205):
    img = Image.open(img_path)
    w, h = img.size
    img_np = np.array(img)
    tiles = []
    stride = tile_size - overlap
    for row in range(0, h, stride):
        for col in range(0, w, stride):
            tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            r_end = min(row + tile_size, h)
            c_end = min(col + tile_size, w)
            tile[:r_end - row, :c_end - col] = img_np[row:r_end, col:c_end]
            tiles.append((tile, row, col))
    return tiles, w, h


def compute_iou(b1, b2):
    x1_1, y1_1 = b1[0] - b1[2] / 2, b1[1] - b1[3] / 2
    x2_1, y2_1 = b1[0] + b1[2] / 2, b1[1] + b1[3] / 2
    x1_2, y1_2 = b2[0] - b2[2] / 2, b2[1] - b2[3] / 2
    x2_2, y2_2 = b2[0] + b2[2] / 2, b2[1] + b2[3] / 2
    inter = max(0, min(x2_1, x2_2) - max(x1_1, x1_2)) * \
            max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / union if union > 0 else 0


def nms_boxes(boxes, iou_threshold):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        boxes = [b for b in boxes if compute_iou(best, b) < iou_threshold]
    return keep


def center_distance_dedup(boxes, min_dist):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    keep = []
    for box in boxes:
        too_close = False
        for kept in keep:
            dist = np.sqrt((box[0] - kept[0]) ** 2 + (box[1] - kept[1]) ** 2)
            if dist < min_dist:
                too_close = True
                break
        if not too_close:
            keep.append(box)
    return keep


def predict_image(img_path):
    tiles, img_w, img_h = tile_image(img_path, TILE_SIZE, OVERLAP)
    all_boxes = []

    for tile_img, row_offset, col_offset in tiles:
        results = model.predict(tile_img, imgsz=TILE_SIZE, conf=CONF, verbose=False)
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf_val = box.conf[0].cpu().item()
                cx = (x1 + x2) / 2 + col_offset
                cy = (y1 + y2) / 2 + row_offset
                w = x2 - x1
                h = y2 - y1
                all_boxes.append((cx, cy, w, h, conf_val))

    raw_count = len(all_boxes)
    after_nms = nms_boxes(all_boxes, NMS_IOU)
    after_dedup = center_distance_dedup(after_nms, MIN_CENTER_DIST)

    return {
        "raw_detections": raw_count,
        "after_nms": len(after_nms),
        "plant_count": len(after_dedup),
        "image_size": f"{img_w} x {img_h}",
        "tiles_processed": len(tiles),
    }


# ── Routes ──────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/count", methods=["POST"])
def count_plants():
    if "images" not in request.files:
        return jsonify({"error": "No images uploaded"}), 400

    files = request.files.getlist("images")
    if not files or files[0].filename == "":
        return jsonify({"error": "No images selected"}), 400

    results = []
    total_plants = 0

    for f in files:
        # Save to a temp file for processing
        suffix = Path(f.filename).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            f.save(tmp.name)
            tmp_path = tmp.name

        try:
            result = predict_image(tmp_path)
            result["filename"] = f.filename
            results.append(result)
            total_plants += result["plant_count"]
        except Exception as e:
            results.append({
                "filename": f.filename,
                "error": str(e),
                "plant_count": 0,
            })
        finally:
            os.unlink(tmp_path)

    return jsonify({
        "results": results,
        "total_images": len(results),
        "total_plants": total_plants,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
