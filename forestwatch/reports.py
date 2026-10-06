from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from .config import settings
from .analysis import change_detection
from .multisatellite import DATES, available as aligned_data_available, aligned_change_detection

def percentage_display(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert index-like values to readable percentage columns for exports."""
    result = frame.copy()
    for column in ["ndvi_mean", "ndvi_median", "ndvi_std"]:
        if column in result:
            result[f"{column}_percent"] = (result[column] * 100).round(2)
            result.drop(columns=column, inplace=True)
    if "forest_health_score" in result:
        result["forest_health_score"] = result["forest_health_score"].round(2)
    # NDVI-index deltas are a technical value and are intentionally not shown
    # in user-facing reports.  The HTML report uses a spatial changed-area
    # summary instead; the CSV also excludes the raw delta for consistency.
    return result.drop(columns=["ndvi_change"], errors="ignore")

def export_metrics(series: pd.DataFrame, report_id: str | None = None) -> str:
    """Export either all metrics or a user-selected reporting period."""
    settings.ensure_output_dirs()
    suffix = f"_{report_id}" if report_id else ""
    path=settings.output_path / "metrics" / f"ndvi_metrics{suffix}.csv"
    percentage_display(series.drop(columns=["data_quality"], errors="ignore")).to_csv(path,index=False, float_format="%.2f")
    return str(path)

def _changed_area(latest: pd.Series, previous: pd.Series | None) -> tuple[str, str]:
    """Return a defensible spatial change measurement for the report table.

    Only the supplied co-registered 2024-2025 grid has a physical map scale,
    so it is reported in hectares.  Other rendered NDVI scenes have no
    georeferencing; reporting their number of changed image pixels is honest
    and avoids inventing a hectare value.
    """
    if previous is None:
        return "Changed forest area", "No earlier observation is available for comparison."
    earlier_date, later_date = str(previous.get("date", "")), str(latest.get("date", ""))
    try:
        if {earlier_date, later_date} == set(DATES) and aligned_data_available():
            _, summary = aligned_change_detection(earlier_date, later_date)
            area = summary["decrease_area_hectares"] + summary["increase_area_hectares"]
            return "Changed forest area", f"{area:.2f} hectares"
        summary = change_detection(str(previous["path"]), str(latest["path"]))
        difference = summary["difference_map"]
        changed = np.isfinite(difference) & (np.abs(difference) >= abs(settings.decrease_threshold))
        changed_pixels = int(changed.sum())
        return "Changed image area", f"{max(changed_pixels, 0):,} pixels"
    except (OSError, ValueError, KeyError):
        return "Changed forest area", "Comparison is unavailable for these observations."


def generate_html_report(series: pd.DataFrame, alerts: list[dict], period="monthly", report_id: str | None = None,
                         previous_observation: pd.DataFrame | None = None) -> str:
    settings.ensure_output_dirs()
    suffix = f"_{report_id}" if report_id else ""
    path=settings.output_path / "reports" / f"{period}{suffix}_report.html"
    latest_frame = percentage_display(series.tail(1).drop(columns=["data_quality"], errors="ignore"))
    latest=latest_frame.iloc[0].to_dict() if not latest_frame.empty else {}
    labels = {
        "date": "Observation date", "ndvi_mean_percent": "Average NDVI (%)", "ndvi_median_percent": "Middle NDVI value (%)",
        "ndvi_std_percent": "NDVI variation (%)",
        "forest_health_score": "Vegetation health score", "health_category": "Vegetation health category", "path": "Source image",
    }
    latest_rows = pd.DataFrame(
        [{"Measure": labels.get(key, key.replace("_", " ").title()), "Value": round(value, 2) if isinstance(value, float) else value} for key, value in latest.items()]
    )
    previous = None
    if len(series) >= 2:
        previous = series.iloc[-2]
    elif previous_observation is not None and not previous_observation.empty:
        previous = previous_observation.iloc[-1]
    area_label, area_value = _changed_area(series.iloc[-1], previous) if not series.empty else ("Changed forest area", "No observation is available.")
    latest_rows = pd.concat([latest_rows, pd.DataFrame([{"Measure": area_label, "Value": area_value}])], ignore_index=True)
    latest_table = latest_rows.to_html(index=False, escape=True) if not latest_rows.empty else "<p>No observation is available.</p>"
    chart_frame = series.copy()
    chart_frame["date"] = pd.to_datetime(chart_frame["date"], errors="coerce")
    if chart_frame.empty or chart_frame["date"].isna().all():
        trend_chart = "<p>No dated observations available for a trend graph.</p>"
    else:
        trend_chart = pio.to_html(px.line(chart_frame, x="date", y="ndvi_mean", markers=True, title="NDVI-index trend", labels={"date":"Observation date", "ndvi_mean":"NDVI index"}), full_html=False, include_plotlyjs="cdn")
    html=f"<html><head><style>body{{font-family:Arial,sans-serif;margin:32px;color:#1f2937}}table{{border-collapse:collapse;width:100%;max-width:800px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #d1d5db}}th{{background:#e8f5e9}}.chart{{max-width:900px;margin:24px 0}}</style></head><body><h1>ForestWatch {period.title()} Report</h1><p>AOI: {settings.aoi_name}</p><h2>Latest observation</h2>{latest_table}<h2>Vegetation trend</h2><div class='chart'>{trend_chart}</div></body></html>"
    path.write_text(html, encoding="utf-8"); return str(path)
