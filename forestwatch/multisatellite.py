"""Read-only processing for the supplied Copernicus two-date project products."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from .config import settings

DATES = ("2024-12-23", "2025-12-28")

def available() -> bool:
    root = settings.project_data_path
    return root.exists() and (root / "Gorewada_Temporal_Dataset_2024_2025.npy").exists()

def _one(pattern: str) -> Path:
    matches = list(settings.project_data_path.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"Required source product not found: {pattern}")
    return matches[0]

def _read_preview(path: Path, bands=1, size=600):
    with rasterio.open(path) as source:
        height, width = source.height, source.width
        scale = min(1, size / max(height, width))
        output_shape = (bands, max(1, int(height * scale)), max(1, int(width * scale)))
        array = source.read(list(range(1, bands + 1)), out_shape=output_shape, resampling=Resampling.bilinear).astype(np.float32)
        return array, {"path": str(path), "crs": str(source.crs), "width": width, "height": height, "resolution": list(source.res), "nodata": source.nodata}

def _normalise(array):
    low, high = np.nanpercentile(array, (2, 98))
    return np.clip((array - low) / max(high - low, 1e-9), 0, 1)

def _s2_band(date: str, band: str) -> Path:
    compact = date.replace("-", "")
    return _one(f"*{compact}*{band}_10m.jp2")

def sentinel2_ndvi(date: str):
    """Calculate NDVI from official B08 and B04 reflectance products."""
    b04, metadata = _read_preview(_s2_band(date, "B04"))
    b08, _ = _read_preview(_s2_band(date, "B08"))
    red, nir = b04[0], b08[0]
    ndvi = np.divide(nir - red, nir + red, out=np.full_like(nir, np.nan), where=(nir + red) != 0)
    stats = {"date": date, "mean_ndvi": float(np.nanmean(ndvi)), "median_ndvi": float(np.nanmedian(ndvi)),
             "min_ndvi": float(np.nanmin(ndvi)), "max_ndvi": float(np.nanmax(ndvi)), **metadata}
    return ndvi, stats

def _s1_band(date: str, polarisation: str) -> Path:
    compact = date.replace("-", "")
    return _one(f"*{polarisation.lower()}*{compact}*.tiff")

def sentinel1_features(date: str):
    date = "2025-12-30" if date == "2025-12-28" else date
    vv, metadata = _read_preview(_s1_band(date, "vv"))
    vh, _ = _read_preview(_s1_band(date, "vh"))
    vv, vh = vv[0], vh[0]
    ratio = np.divide(vv, vh, out=np.full_like(vv, np.nan), where=vh != 0)
    preview = np.dstack([_normalise(vv), _normalise(vh), _normalise(ratio)])
    return preview, {"date": date, "vv_mean": float(np.nanmean(vv)), "vh_mean": float(np.nanmean(vh)), "vv_vh_ratio_mean": float(np.nanmean(ratio)), **metadata}

def labelled_change_summary():
    root = settings.project_data_path
    candidate = np.load(root / "Gorewada_Candidate_Change_Label_2024_2025.npy", mmap_mode="r")
    high = np.load(root / "Gorewada_HighConfidence_Change_Label_2024_2025.npy", mmap_mode="r")
    valid = high != 255
    return {"grid_shape": list(candidate.shape), "candidate_change_pixels": int((candidate == 1).sum()),
            "high_confidence_change_pixels": int(((high == 1) & valid).sum()), "high_confidence_valid_pixels": int(valid.sum()),
            "unknown_or_masked_pixels": int((high == 255).sum())}

def aligned_change_detection(earlier_date: str = DATES[0], later_date: str = DATES[1]):
    """Area-aware vegetation-change classes on the supplied aligned 10 m grid."""
    cube = np.load(settings.project_data_path / "Gorewada_Temporal_Dataset_2024_2025.npy", mmap_mode="r")
    if earlier_date == later_date:
        raise ValueError("Choose two different dates for change detection.")
    if {earlier_date, later_date} != set(DATES):
        raise ValueError("Area calculation is available only for the two dates in the aligned project grid.")
    if earlier_date == DATES[0]:
        before, after = np.asarray(cube[..., 2]), np.asarray(cube[..., 5])
    else:
        before, after = np.asarray(cube[..., 5]), np.asarray(cube[..., 2])
    change = after - before
    valid_mask = np.load(settings.project_data_path / "Gorewada_HighConfidence_Change_Label_2024_2025.npy", mmap_mode="r") != 255
    threshold = abs(settings.decrease_threshold)
    decrease = (change <= -threshold) & valid_mask
    increase = (change >= threshold) & valid_mask
    stable = ~(decrease | increase) & np.isfinite(change) & valid_mask
    pixel_hectares = 0.01  # aligned project grid is 10 m × 10 m
    def area(mask): return round(float(mask.sum()) * pixel_hectares, 2)
    bound = max(float(np.nanpercentile(np.abs(change), 98)), 1e-6)
    normalised = np.clip(change / bound, -1, 1)
    display = np.zeros((*change.shape, 3), dtype=np.uint8)
    display[..., 0] = np.where(normalised < 0, -normalised * 255, 0).astype(np.uint8)
    display[..., 1] = np.where(normalised >= 0, normalised * 255, 0).astype(np.uint8)
    return display, {"decrease_area_hectares": area(decrease), "increase_area_hectares": area(increase),
                     "stable_area_hectares": area(stable), "decrease_pixels": int(decrease.sum()),
                     "increase_pixels": int(increase.sum()), "stable_pixels": int(stable.sum()),
                     "comparison_dates": f"{earlier_date} to {later_date}", "resolution": "10 m × 10 m"}

def comparison():
    if not available():
        raise FileNotFoundError(f"Project data directory is unavailable: {settings.project_data_path}")
    # The supplied temporal cube is the project’s aligned 10 m Gorewada grid.
    # Channel order is documented in its accompanying notebook.
    cube = np.load(settings.project_data_path / "Gorewada_Temporal_Dataset_2024_2025.npy", mmap_mode="r")
    if cube.ndim != 3 or cube.shape[-1] != 6:
        raise ValueError("Expected a six-channel temporal dataset (VV, VH, NDVI for each date).")
    vv24, vh24, before, vv25, vh25, after = (np.asarray(cube[..., index], dtype=np.float32) for index in range(6))
    change = after - before
    bound = max(float(np.nanpercentile(np.abs(change), 98)), 1e-6)
    normalised = np.clip(change / bound, -1, 1)
    display = np.zeros((*change.shape, 3), dtype=np.uint8)
    display[..., 0] = np.where(normalised < 0, -normalised * 255, 0).astype(np.uint8)
    display[..., 1] = np.where(normalised >= 0, normalised * 255, 0).astype(np.uint8)
    ratio24 = np.divide(vv24, vh24, out=np.full_like(vv24, np.nan), where=vh24 != 0)
    ratio25 = np.divide(vv25, vh25, out=np.full_like(vv25, np.nan), where=vh25 != 0)
    radar_before = np.dstack([_normalise(vv24), _normalise(vh24), _normalise(ratio24)])
    radar_after = np.dstack([_normalise(vv25), _normalise(vh25), _normalise(ratio25)])
    base_metadata = {"grid": "Aligned Gorewada analysis grid", "grid_shape": list(cube.shape[:2]), "resolution_m": 10, "crs": "EPSG:32644"}
    summary = {"sentinel2_before": {"date": DATES[0], "mean_ndvi": float(np.nanmean(before)), "median_ndvi": float(np.nanmedian(before)), **base_metadata},
               "sentinel2_after": {"date": DATES[1], "mean_ndvi": float(np.nanmean(after)), "median_ndvi": float(np.nanmedian(after)), **base_metadata},
               "ndvi_mean_change": float(np.nanmean(change)), "ndvi_change_display_bound": bound,
               "sentinel1_before": {"date": DATES[0], "vv_mean": float(np.nanmean(vv24)), "vh_mean": float(np.nanmean(vh24)), "vv_vh_ratio_mean": float(np.nanmean(ratio24))},
               "sentinel1_after": {"date": "2025-12-30", "vv_mean": float(np.nanmean(vv25)), "vh_mean": float(np.nanmean(vh25)), "vv_vh_ratio_mean": float(np.nanmean(ratio25))},
               "labels": labelled_change_summary()}
    return before, after, display, radar_before, radar_after, summary
