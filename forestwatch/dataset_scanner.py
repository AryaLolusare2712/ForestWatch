"""Dataset discovery based on actual paths and filenames, never assumed layout."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import re
import pandas as pd
from PIL import Image

DATE = re.compile(r"(?<!\d)(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)(?!\d)")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

def extract_date(text: str):
    match = DATE.search(text)
    if not match: return None
    try: return datetime.strptime("-".join(match.groups()), "%Y-%m-%d").date().isoformat()
    except ValueError: return None

def classify(path: Path) -> str:
    value = str(path).lower()
    if "ndvi" in value: return "ndvi"
    if "true" in value or "rgb" in value or "truecolor" in value: return "true_color"
    if re.search(r"(^|[_/\\-])vv([_/\\-]|$)", value): return "vv"
    if re.search(r"(^|[_/\\-])vh([_/\\-]|$)", value): return "vh"
    if "sentinel-1" in value or "s1_" in path.name.lower(): return "sentinel1_unlabelled"
    return "unknown"

def image_details(path: Path) -> dict:
    try:
        with Image.open(path) as im:
            return {"width": im.width, "height": im.height, "channels": len(im.getbands()), "dtype": "uint8", "crs": None, "nodata": None, "valid": True, "error": None}
    except Exception as exc:
        return {"width": None, "height": None, "channels": None, "dtype": None, "crs": None, "nodata": None, "valid": False, "error": str(exc)}

def scan_dataset(dataset_path: str | Path, inspect_images: bool = False) -> pd.DataFrame:
    root = Path(dataset_path)
    rows = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        row = {"path": str(path), "relative_path": str(path.relative_to(root)), "filename": path.name,
               "category": classify(path), "date": extract_date(str(path)), "format": path.suffix.lower(), "size_bytes": path.stat().st_size}
        if inspect_images and path.suffix.lower() in IMAGE_EXTENSIONS: row.update(image_details(path))
        rows.append(row)
    return pd.DataFrame(rows)

def inventory_summary(inventory: pd.DataFrame) -> dict:
    dates = pd.to_datetime(inventory["date"], errors="coerce")
    counts = inventory["category"].value_counts().to_dict() if not inventory.empty else {}
    by_date = inventory.groupby(["date", "category"]).size().unstack(fill_value=0) if not inventory.empty else pd.DataFrame()
    def unmatched(a, b):
        if by_date.empty:
            return []
        left = by_date[a] if a in by_date.columns else pd.Series(0, index=by_date.index)
        right = by_date[b] if b in by_date.columns else pd.Series(0, index=by_date.index)
        return sorted(set(by_date.index[left > 0]) - set(by_date.index[right > 0]))
    return {"file_count": int(len(inventory)), "counts": counts, "formats": inventory["format"].value_counts().to_dict(),
            "date_range": {"start": dates.min().date().isoformat() if dates.notna().any() else None, "end": dates.max().date().isoformat() if dates.notna().any() else None},
            "pairing": {"vv_without_vh": unmatched("vv", "vh"), "vh_without_vv": unmatched("vh", "vv"),
                        "ndvi_without_true_color": unmatched("ndvi", "true_color"), "true_color_without_ndvi": unmatched("true_color", "ndvi"),
                        "sentinel1_unlabelled": counts.get("sentinel1_unlabelled", 0)},
            "invalid_files": inventory.loc[inventory.get("valid", pd.Series(True, index=inventory.index)) == False, ["path", "error"]].to_dict("records") if "valid" in inventory else []}
