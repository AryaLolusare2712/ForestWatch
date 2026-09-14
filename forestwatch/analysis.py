from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from .config import settings

def load_image(path): return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)

def ndvi_proxy(path: str) -> np.ndarray:
    """Derive a display-image proxy only; JPEG NDVI palettes are not physical NDVI rasters."""
    rgb = load_image(path) / 255.0
    # Preserves an explicit caveat: values derived from rendered imagery are not scientific NDVI.
    return np.clip((rgb[..., 1] - rgb[..., 0]) / (rgb[..., 1] + rgb[..., 0] + 1e-6), -1, 1)

def ndvi_statistics(path: str) -> dict:
    arr = ndvi_proxy(path); valid = np.isfinite(arr)
    values = arr[valid]
    return {"source": "JPEG NDVI visualization proxy", "scientific_caveat": "Metrics are derived from rendered JPEG pixels; use georeferenced numeric NDVI rasters for scientific measurement.",
            "mean": float(values.mean()), "median": float(np.median(values)), "min": float(values.min()), "max": float(values.max()), "std": float(values.std()),
            "valid_pixel_percentage": float(valid.mean()*100), "low_vegetation_percentage": float((values < settings.low_ndvi).mean()*100), "healthy_vegetation_percentage": float((values >= settings.healthy_ndvi).mean()*100)}

def ndvi_timeseries(inventory: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for row in inventory.loc[inventory.category == "ndvi"].dropna(subset=["date"]).sort_values("date").itertuples():
        try:
            stats=ndvi_statistics(row.path)
        except (OSError, ValueError) as exc:
            # A corrupt display JPEG must never prevent analysis of valid scenes.
            rows.append({"date":row.date, "path":row.path, "ndvi_mean":np.nan, "ndvi_median":np.nan,
                         "ndvi_std":np.nan, "data_quality":f"Unreadable image: {exc}"})
            continue
        rows.append({"date":row.date, "path":row.path, "ndvi_mean":stats["mean"], "ndvi_median":stats["median"], "ndvi_std":stats["std"], "data_quality":"JPEG display proxy"})
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
    before=ndvi_proxy(earlier_path); after=ndvi_proxy(later_path)
    if before.shape != after.shape: raise ValueError("Images have different dimensions; spatial change map is not reliable.")
    diff=after-before; threshold=abs(settings.decrease_threshold)
    decrease=diff <= -threshold; increase=diff >= threshold; stable=~(decrease | increase)
    return {"decrease_pixel_percentage":float(decrease.mean()*100), "increase_pixel_percentage":float(increase.mean()*100), "stable_pixel_percentage":float(stable.mean()*100), "difference_map":diff,
            "interpretation":"Potential vegetation change only. Verify on the ground; rendered JPEG inputs do not support area estimates or confirmed deforestation claims."}
