"""YOLOv8n vegetation-change region detection.

NDVI remains responsible for the quantitative vegetation measurements.  This
module uses the project's trained YOLOv8n weights to locate *probable* change
regions in the NDVI difference image and draws those regions on the later
true-colour satellite scene.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .analysis import ndvi_proxy


WEIGHTS_PATH = Path(__file__).resolve().parent / "models" / "yolov8n_vegetation_change.pt"
MODEL_INPUT_SIZE = 128
CONFIDENCE_THRESHOLD = 0.25


@lru_cache(maxsize=1)
def _model():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("YOLO runtime is not installed on this service.") from exc
    if not WEIGHTS_PATH.is_file():
        raise RuntimeError("The packaged YOLOv8n vegetation-change model is unavailable.")
    return YOLO(str(WEIGHTS_PATH))


def _change_image(earlier_ndvi: str, later_ndvi: str) -> np.ndarray:
    """Create the colour-coded change representation used as YOLO input."""
    before = ndvi_proxy(earlier_ndvi)
    after = ndvi_proxy(later_ndvi)
    if before.shape != after.shape:
        raise ValueError("Images have different dimensions; YOLO change regions cannot be compared.")
    difference = after - before
    valid = np.isfinite(difference)
    strength = np.clip(np.abs(np.nan_to_num(difference)) / 0.25, 0, 1)
    image = np.zeros((*difference.shape, 3), dtype=np.uint8)
    # Red = vegetation decrease; green = vegetation increase. The blue channel
    # preserves confidence strength so the detector receives a rich RGB tile.
    image[..., 0] = np.where(difference < 0, strength * 255, 0).astype(np.uint8)
    image[..., 1] = np.where(difference > 0, strength * 255, 0).astype(np.uint8)
    image[..., 2] = (strength * 100).astype(np.uint8)
    image[~valid] = 0
    return image


def detect_change_regions(earlier_ndvi: str, later_ndvi: str) -> list[dict]:
    """Run the packaged YOLOv8n detector and return normalised bounding boxes."""
    image = _change_image(earlier_ndvi, later_ndvi)
    result = _model().predict(image, imgsz=MODEL_INPUT_SIZE, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
    height, width = image.shape[:2]
    regions: list[dict] = []
    if result.boxes is None:
        return regions
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = float(box.conf[0])
        regions.append({
            "label": "probable vegetation change",
            "confidence": round(confidence, 4),
            "x": round(max(0, x1) / width, 5),
            "y": round(max(0, y1) / height, 5),
            "width": round(min(width, x2) / width - max(0, x1) / width, 5),
            "height": round(min(height, y2) / height - max(0, y1) / height, 5),
        })
    return regions


def yolo_change_overlay(earlier_ndvi: str, later_ndvi: str, later_true_colour: str, output: Path) -> tuple[Path, list[dict]]:
    """Draw YOLO detections over the real later true-colour image."""
    regions = detect_change_regions(earlier_ndvi, later_ndvi)
    with Image.open(later_true_colour) as source:
        canvas = source.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for region in regions:
        left = int(region["x"] * canvas.width)
        top = int(region["y"] * canvas.height)
        right = int((region["x"] + region["width"]) * canvas.width)
        bottom = int((region["y"] + region["height"]) * canvas.height)
        draw.rectangle((left, top, right, bottom), outline=(255, 70, 70), width=max(2, canvas.width // 250))
        draw.text((left + 4, max(0, top - 18)), f"YOLO {region['confidence']:.0%}", fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, "PNG")
    return output, regions
