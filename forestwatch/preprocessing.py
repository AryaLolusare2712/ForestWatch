"""Non-destructive satellite-image preparation used before application analysis.

The source dataset is never changed.  Preparation validates each readable image,
applies EXIF orientation, converts it to three standard RGB bands, and converts
pixel values to finite 0–1 floats at the point of analysis.  A manifest in the
output directory records which files were successfully prepared.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .config import settings
from .dataset_scanner import IMAGE_EXTENSIONS, scan_dataset


def prepared_rgb(path: str | Path) -> np.ndarray:
    """Return a validated, orientation-corrected RGB float image in [0, 1]."""
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        pixels = np.asarray(image, dtype=np.float32) / 255.0
    if pixels.ndim != 3 or pixels.shape[-1] != 3 or not np.isfinite(pixels).all():
        raise ValueError("Image did not produce a finite three-band RGB array.")
    return np.clip(pixels, 0.0, 1.0)


def save_preprocessed_preview(path: str | Path) -> Path:
    """Save a lossless visual preview of the standardized local image."""
    source = Path(path)
    pixels = prepared_rgb(source)
    settings.ensure_output_dirs()
    destination = settings.output_path / "processed" / "previews" / f"{source.stem}_preprocessed.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(pixels * 255).astype(np.uint8), mode="RGB").save(destination, optimize=True)
    return destination


def _manifest_path() -> Path:
    settings.ensure_output_dirs()
    return settings.output_path / "processed" / "preprocessing_manifest.json"


def _boundary_mask_path() -> Path:
    settings.ensure_output_dirs()
    return settings.output_path / "processed" / "gorewada_boundary_mask.npy"


def processed_images_path() -> Path:
    """Dedicated location for standardized copies of all source imagery."""
    settings.ensure_output_dirs()
    path = settings.output_path / "processed" / "images"
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_boundary_reference(reference_image: str | Path) -> dict:
    """Create the canonical Gorewada valid-area mask from a supplied base image.

    Pixels that are near black in the reference are outside the study boundary.
    The result is kept separately from the source image and can only be used on
    imagery with exactly the same grid dimensions.
    """
    source = Path(reference_image)
    pixels = prepared_rgb(source)
    threshold = 5 / 255.0
    mask = np.max(pixels, axis=-1) > threshold
    np.save(_boundary_mask_path(), mask.astype(bool))
    # A readable QA preview that keeps all original colours, including black
    # gaps. Interior gaps are later treated as stable for change comparison.
    preview = np.rint(pixels * 255).astype(np.uint8)
    preview_path = settings.output_path / "processed" / "gorewada_preprocessed_boundary_preview.png"
    Image.fromarray(preview, mode="RGB").save(preview_path, optimize=True)
    details = {
        "reference_image": source.name,
        "reference_path": str(source),
        "shape": [int(mask.shape[0]), int(mask.shape[1])],
        "valid_boundary_pixels": int(mask.sum()),
        "boundary_coverage_percent": round(float(mask.mean() * 100), 2),
        "black_threshold_rgb": 5,
        "preview_path": str(preview_path),
        "rule": "Near-black reference pixels are outside the Gorewada boundary.",
    }
    (_boundary_mask_path().with_suffix(".json")).write_text(json.dumps(details, indent=2), encoding="utf-8")
    return details


def valid_data_mask(pixels: np.ndarray) -> tuple[np.ndarray, str]:
    """Return valid image pixels and state how the Gorewada mask was applied."""
    if pixels.ndim != 3 or pixels.shape[-1] != 3:
        raise ValueError("Expected a three-band RGB image.")
    # Any interior black hole is missing coverage, never a vegetation value.
    valid = np.max(pixels, axis=-1) > (5 / 255.0)
    mask_path = _boundary_mask_path()
    if not mask_path.exists():
        return valid, "No boundary reference available; near-black pixels excluded."
    boundary = np.load(mask_path, mmap_mode="r")
    if boundary.shape != valid.shape:
        return valid, f"Boundary reference not applied: image grid {valid.shape[1]} × {valid.shape[0]} does not match reference grid {boundary.shape[1]} × {boundary.shape[0]}."
    return valid & boundary, "Gorewada boundary reference and near-black no-data mask applied."


def boundary_mask_for_shape(shape: tuple[int, int]) -> tuple[np.ndarray, str]:
    """Return the fixed study boundary without interpreting interior pixels."""
    mask_path = _boundary_mask_path()
    if not mask_path.exists():
        return np.ones(shape, dtype=bool), "No boundary reference available."
    boundary = np.load(mask_path, mmap_mode="r")
    if boundary.shape != shape:
        return np.ones(shape, dtype=bool), "Boundary reference does not match this image grid."
    return np.asarray(boundary, dtype=bool), "Gorewada boundary reference applied."


def preprocess_dataset(dataset_path: str | Path | None = None, max_new_images: int | None = None) -> pd.DataFrame:
    """Create standardized PNG copies while preserving every source image."""
    root = Path(dataset_path or settings.dataset_path)
    inventory = scan_dataset(root)
    records: list[dict] = []
    new_exports = 0
    for row in inventory.itertuples():
        source = Path(row.path)
        # The current analytical workflow derives the vegetation time series
        # from NDVI scenes only. Do not create duplicate products for radar,
        # true-colour, or unrelated imagery.
        if row.category != "ndvi" or source.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        # Keep the source folder hierarchy so each prepared product can be
        # traced directly back to its original local file.
        processed_path = processed_images_path() / Path(row.relative_path).with_suffix(".png")
        if not processed_path.exists() and max_new_images is not None and new_exports >= max_new_images:
            break
        if processed_path.exists():
            try:
                with Image.open(processed_path) as prepared:
                    prepared.load()
                    records.append({
                        "relative_path": row.relative_path, "processed_path": str(processed_path),
                        "category": row.category, "date": row.date, "status": "Prepared",
                        "width": prepared.width, "height": prepared.height, "channels": 3,
                        "processing": "previously standardized RGB export reused",
                        "message": "Ready for local analysis (resumed export).",
                        "valid_coverage_percent": None,
                    })
                continue
            except OSError:
                # This is a generated, incomplete output from an interrupted
                # run—not a source dataset file—so it is safe to rebuild.
                processed_path.unlink()
        try:
            pixels = prepared_rgb(source)
            valid, mask_status = valid_data_mask(pixels)
            processed_path.parent.mkdir(parents=True, exist_ok=True)
            # PNG avoids a second lossy JPEG compression; skipping expensive
            # optimizer passes keeps full-dataset preparation practical.
            Image.fromarray(np.rint(pixels * 255).astype(np.uint8), mode="RGB").save(processed_path)
            new_exports += 1
            records.append({
                "relative_path": row.relative_path,
                "processed_path": str(processed_path),
                "category": row.category,
                "date": row.date,
                "status": "Prepared",
                "width": int(pixels.shape[1]),
                "height": int(pixels.shape[0]),
                "channels": 3,
                "processing": "orientation corrected; RGB standardized; pixels normalized to 0–1; no-data mask evaluated",
                "message": f"Ready for local analysis. {mask_status}",
                "valid_coverage_percent": round(float(valid.mean() * 100), 2),
            })
        except (OSError, ValueError, SyntaxError) as exc:
            records.append({
                "relative_path": row.relative_path,
                "processed_path": None,
                "category": row.category,
                "date": row.date,
                "status": "Skipped",
                "width": None,
                "height": None,
                "channels": None,
                "processing": "not processed",
                "message": str(exc),
            })
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_directory": str(root),
        "method": "Non-destructive validation, EXIF orientation correction, RGB standardization, 0–1 pixel normalization, and standardized PNG export.",
        "records": records,
    }
    _manifest_path().write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return pd.DataFrame(records)


def preprocessing_summary() -> tuple[pd.DataFrame, str]:
    """Run preparation and return a small user-facing summary."""
    results = preprocess_dataset()
    prepared = int((results.status == "Prepared").sum()) if not results.empty else 0
    skipped = int((results.status == "Skipped").sum()) if not results.empty else 0
    table = results[["category", "date", "status", "width", "height", "valid_coverage_percent", "message"]].copy() if not results.empty else results
    message = (
        f"### Preprocessing complete\nPrepared **{prepared}** image(s) and skipped **{skipped}** unreadable image(s). "
        "Original satellite files were not changed."
    )
    return table, message
