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


MODEL_PATH = Path(__file__).resolve().parent / "models" / "yolov8n_vegetation_change.onnx"
MODEL_INPUT_SIZE = 128
# The labelled dataset is very small, so a low review threshold is used. The
# boxes are deliberately labelled as *probable* changes in the interface.
CONFIDENCE_THRESHOLD = 0.001


@lru_cache(maxsize=1)
def _model():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(f"YOLO ONNX runtime is unavailable on this service: {exc}") from exc
    if not MODEL_PATH.is_file():
        raise RuntimeError("The packaged YOLOv8n vegetation-change model is unavailable.")
    return ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


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
    resized = np.asarray(Image.fromarray(image).resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.Resampling.BILINEAR), dtype=np.float32)
    inputs = (resized.transpose(2, 0, 1)[None] / 255.0).astype(np.float32)
    session = _model()
    output = session.run(None, {session.get_inputs()[0].name: inputs})[0][0].T
    candidates = []
    for center_x, center_y, box_width, box_height, confidence in output:
        if float(confidence) < CONFIDENCE_THRESHOLD:
            continue
        left = max(0.0, (float(center_x) - float(box_width) / 2) / MODEL_INPUT_SIZE)
        top = max(0.0, (float(center_y) - float(box_height) / 2) / MODEL_INPUT_SIZE)
        right = min(1.0, (float(center_x) + float(box_width) / 2) / MODEL_INPUT_SIZE)
        bottom = min(1.0, (float(center_y) + float(box_height) / 2) / MODEL_INPUT_SIZE)
        candidates.append((float(confidence), left, top, right, bottom))
    # Small deterministic non-maximum suppression prevents overlapping boxes
    # without importing the much heavier PyTorch/Ultralytics runtime.
    regions: list[dict] = []
    for confidence, left, top, right, bottom in sorted(candidates, reverse=True):
        overlaps = False
        for chosen in regions:
            ix1, iy1 = max(left, chosen["x"]), max(top, chosen["y"])
            ix2, iy2 = min(right, chosen["x"] + chosen["width"]), min(bottom, chosen["y"] + chosen["height"])
            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            union = (right - left) * (bottom - top) + chosen["width"] * chosen["height"] - intersection
            if union and intersection / union >= 0.45:
                overlaps = True
                break
        if overlaps:
            continue
        regions.append({
            "label": "probable vegetation change",
            "confidence": round(confidence, 4),
            "x": round(left, 5),
            "y": round(top, 5),
            "width": round(right - left, 5),
            "height": round(bottom - top, 5),
        })
        if len(regions) == 20:
            break
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
