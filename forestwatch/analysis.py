from __future__ import annotations
from pathlib import Path
import re
from datetime import date as calendar_date
from functools import lru_cache
import numpy as np
import pandas as pd
from .config import settings
from .preprocessing import prepared_rgb, valid_data_mask, boundary_mask_for_shape
from .multisatellite import radar_assisted_ndvi_proxy
from .dataset_scanner import scan_dataset

def load_image(path):
    """Load imagery only through the common validation and normalization stage."""
    return prepared_rgb(path) * 255.0

def _date_from_path(path: str) -> str | None:
    match = re.search(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)", Path(path).name)
    return "-".join(match.groups()) if match else None


@lru_cache(maxsize=1)
def _sentinel1_scenes() -> tuple[tuple[calendar_date, str], ...]:
    """Index the supplied rendered S1 scenes once per server process."""
    records = scan_dataset(settings.dataset_path)
    scenes = records.loc[records.category == "sentinel1_unlabelled", ["date", "path"]].dropna()
    return tuple((calendar_date.fromisoformat(row.date), row.path) for row in scenes.itertuples(index=False))


def _resize_nearest(array: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows = np.rint(np.linspace(0, array.shape[0] - 1, shape[0])).astype(int)
    columns = np.rint(np.linspace(0, array.shape[1] - 1, shape[1])).astype(int)
    return array[np.ix_(rows, columns)]


def _sentinel1_image_gap_fill(date: str, visible_proxy: np.ndarray, fill_mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """Use the nearest supplied Sentinel-1 rendered scene to fill optical gaps.

    Source S1 JPEGs do not carry separate VV/VH arrays or georeferencing.  The
    estimate is therefore calibrated on readable pixels of the current optical
    scene and is limited strictly to black interior gaps.
    """
    try:
        target_date = calendar_date.fromisoformat(date)
        radar_date, radar_path = min(_sentinel1_scenes(), key=lambda item: abs((item[0] - target_date).days))
    except (ValueError, TypeError):
        return visible_proxy, {"used": False, "reason": "No dated Sentinel-1 scene is available."}
    radar_rgb = prepared_rgb(radar_path)
    radar_signal = _resize_nearest(np.mean(radar_rgb, axis=-1), visible_proxy.shape)
    overlap = np.isfinite(visible_proxy) & np.isfinite(radar_signal)
    if overlap.sum() < 100:
        return visible_proxy, {"used": False, "reason": "Too little visible NDVI area to calibrate Sentinel-1."}
    coefficients, *_ = np.linalg.lstsq(
        np.column_stack((radar_signal[overlap], np.ones(overlap.sum()))), visible_proxy[overlap], rcond=None
    )
    estimate = radar_signal * coefficients[0] + coefficients[1]
    result = visible_proxy.copy()
    usable = fill_mask & np.isfinite(estimate)
    result[usable] = np.clip(estimate[usable], -1, 1)
    return result, {"used": bool(usable.any()), "filled_pixels": int(usable.sum()),
                    "sentinel1_date": radar_date.isoformat(), "days_from_ndvi": abs((radar_date - target_date).days),
                    "reason": "Interior optical gaps estimated from the nearest Sentinel-1 radar scene."}


def ndvi_proxy(path: str, date: str | None = None, with_metadata: bool = False):
    """Derive a display-image proxy only; JPEG NDVI palettes are not physical NDVI rasters."""
    rgb = load_image(path) / 255.0
    # Preserves an explicit caveat: values derived from rendered imagery are not scientific NDVI.
    values = np.clip((rgb[..., 1] - rgb[..., 0]) / (rgb[..., 1] + rgb[..., 0] + 1e-6), -1, 1)
    valid, _ = valid_data_mask(rgb)
    values[~valid] = np.nan
    boundary, _ = boundary_mask_for_shape(values.shape)
    # An optical black gap inside the boundary has no NDVI information.  For
    # the two dates with aligned S1 data, use the radar estimate rather than
    # treating it as a vegetation loss or a made-up optical measurement.
    scene_date = date or _date_from_path(path) or ""
    gap_mask = boundary & ~valid
    values, radar = radar_assisted_ndvi_proxy(scene_date, values.shape, values, gap_mask)
    if not radar["used"]:
        values, radar = _sentinel1_image_gap_fill(scene_date, values, gap_mask)
    return (values, radar) if with_metadata else values

def ndvi_statistics(path: str, date: str | None = None) -> dict:
    arr, radar = ndvi_proxy(path, date=date, with_metadata=True); valid = np.isfinite(arr)
    values = arr[valid]
    # A colour-rendered JPEG may contain palette endpoints of -1 and +1.
    # Use robust percentiles for display rather than presenting those isolated
    # palette pixels as the true lowest/highest vegetation measurement.
    display_low, display_high = np.percentile(values, [2, 98])
    return {"source": "JPEG NDVI visualization proxy", "scientific_caveat": "Metrics are derived from rendered JPEG pixels; use georeferenced numeric NDVI rasters for scientific measurement.", "radar_gap_fill": radar,
            "mean": float(values.mean()), "median": float(np.median(values)), "min": float(values.min()), "max": float(values.max()),
            "display_low": float(display_low), "display_high": float(display_high), "std": float(values.std()),
            "valid_pixel_percentage": float(valid.mean()*100), "low_vegetation_percentage": float((values < settings.low_ndvi).mean()*100), "healthy_vegetation_percentage": float((values >= settings.healthy_ndvi).mean()*100)}

def ndvi_timeseries(inventory: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for row in inventory.loc[inventory.category == "ndvi"].dropna(subset=["date"]).sort_values("date").itertuples():
        try:
            stats=ndvi_statistics(row.path, row.date)
        except (OSError, ValueError) as exc:
            # A corrupt display JPEG must never prevent analysis of valid scenes.
            rows.append({"date":row.date, "path":row.path, "ndvi_mean":np.nan, "ndvi_median":np.nan,
                         "ndvi_std":np.nan, "data_quality":f"Unreadable image: {exc}"})
            continue
        quality = "NDVI proxy with Sentinel-1 radar filling interior optical gaps" if stats["radar_gap_fill"]["used"] else "JPEG NDVI display proxy"
        rows.append({"date":row.date, "path":row.path, "ndvi_mean":stats["mean"], "ndvi_median":stats["median"], "ndvi_std":stats["std"], "data_quality":quality})
    result=pd.DataFrame(rows)
    if not result.empty: result["ndvi_change"] = result.ndvi_mean.diff()
    return result

def health_scores(series: pd.DataFrame) -> pd.DataFrame:
    result=series.copy()
    if result.empty: return result
    scaled=np.clip((result.ndvi_mean + 1)/2*100, 0, 100)
    penalty=np.clip(-result.ndvi_change.fillna(0)*100, 0, 20)
    result["forest_health_score"]=(scaled-penalty).clip(0,100)
    result["health_category"]=pd.cut(result.forest_health_score, [-1,40,70,100], labels=["Low", "Watch", "Higher"], include_lowest=True).astype(str)
    return result

def change_detection(earlier_path: str, later_path: str) -> dict:
    before, before_radar = ndvi_proxy(earlier_path, with_metadata=True); after, after_radar = ndvi_proxy(later_path, with_metadata=True)
    if before.shape != after.shape: raise ValueError("Images have different dimensions; spatial change map is not reliable.")
    boundary, boundary_status = boundary_mask_for_shape(before.shape)
    diff=after-before
    # A black hole inside the fixed forest boundary is retained as stable for
    # comparisons. It is not counted as vegetation decrease or increase.
    black_gap = boundary & (~np.isfinite(before) | ~np.isfinite(after))
    diff[black_gap] = 0.0
    diff[~boundary] = np.nan
    threshold=abs(settings.decrease_threshold)
    decrease=boundary & (diff <= -threshold); increase=boundary & (diff >= threshold)
    stable=boundary & ~(decrease | increase)
    denominator = max(int(boundary.sum()), 1)
    radar_note = " Sentinel-1 VV/VH filled matched-date interior optical gaps before comparison." if before_radar["used"] or after_radar["used"] else " Black gaps without matching Sentinel-1 are treated as stable (no change)."
    return {"decrease_pixel_percentage":float(decrease.sum()/denominator*100), "increase_pixel_percentage":float(increase.sum()/denominator*100), "stable_pixel_percentage":float(stable.sum()/denominator*100), "difference_map":diff,
            "interpretation":f"Potential vegetation change only. {boundary_status}{radar_note}"}
