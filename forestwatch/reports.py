from pathlib import Path
import pandas as pd
from .config import settings

def percentage_display(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert index-like values to readable percentage columns for exports."""
    result = frame.copy()
    for column in ["ndvi_mean", "ndvi_median", "ndvi_std", "ndvi_change"]:
        if column in result:
            result[f"{column}_percent"] = (result[column] * 100).round(2)
            result.drop(columns=column, inplace=True)
    if "forest_health_score" in result:
        result["forest_health_score"] = result["forest_health_score"].round(2)
    return result

def export_metrics(series: pd.DataFrame) -> str:
    settings.ensure_output_dirs(); path=settings.output_path / "metrics" / "ndvi_metrics.csv"
    percentage_display(series.drop(columns=["data_quality"], errors="ignore")).to_csv(path,index=False, float_format="%.2f")
    return str(path)

def generate_html_report(series: pd.DataFrame, alerts: list[dict], period="monthly", report_id: str | None = None) -> str:
    settings.ensure_output_dirs()
    suffix = f"_{report_id}" if report_id else ""
    path=settings.output_path / "reports" / f"{period}{suffix}_report.html"
    latest_frame = percentage_display(series.tail(1).drop(columns=["data_quality"], errors="ignore"))
    latest=latest_frame.iloc[0].to_dict() if not latest_frame.empty else {}
    labels = {
        "date": "Observation date", "ndvi_mean_percent": "Average NDVI (%)", "ndvi_median_percent": "Middle NDVI value (%)",
        "ndvi_std_percent": "NDVI variation (%)", "ndvi_change_percent": "Change from previous observation (%)",
        "forest_health_score": "Vegetation health score", "health_category": "Vegetation health category", "path": "Source image",
    }
    latest_rows = pd.DataFrame(
        [{"Measure": labels.get(key, key.replace("_", " ").title()), "Value": round(value, 2) if isinstance(value, float) else value} for key, value in latest.items()]
    )
    latest_table = latest_rows.to_html(index=False, escape=True) if not latest_rows.empty else "<p>No observation is available.</p>"
    html=f"<html><head><style>body{{font-family:Arial,sans-serif;margin:32px;color:#1f2937}}table{{border-collapse:collapse;width:100%;max-width:800px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #d1d5db}}th{{background:#e8f5e9}}</style></head><body><h1>ForestWatch {period.title()} Report</h1><p>AOI: {settings.aoi_name}</p><h2>Latest observation</h2>{latest_table}<h2>Active alerts</h2>{pd.DataFrame(alerts).to_html(index=False) if alerts else '<p>None generated.</p>'}</body></html>"
    path.write_text(html, encoding="utf-8"); return str(path)
