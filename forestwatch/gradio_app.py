import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from functools import lru_cache
from pathlib import Path
from datetime import datetime
import shutil
from html import escape
from .config import settings
from .dataset_scanner import scan_dataset, inventory_summary
from .analysis import ndvi_timeseries, health_scores, ndvi_statistics, change_detection, ndvi_proxy
from .prediction import train_ndvi_model, forecast_future_ndvi
from .multisatellite import available as multisatellite_available, aligned_change_detection
from .reports import export_metrics, generate_html_report, percentage_display
from .alerts import check_health_alerts, update_alert_status, HEALTH_ALERT_THRESHOLD
from .database import Alert, SessionLocal
from .preprocessing import preprocessing_summary, prepared_rgb

DASHBOARD_CSS = """
.fw-header{display:flex;align-items:center;gap:14px;padding:22px 24px;border:1px solid #cfe2d2;border-radius:16px;background:linear-gradient(120deg,#eff8f0,#f9fcf8);margin-bottom:14px}.fw-mark{font-size:32px}.fw-title{font-size:28px;font-weight:700;color:#173b25}.fw-subtitle{color:#56705d;margin-top:2px}.fw-region{margin-left:auto;text-align:right;color:#294a34}.fw-region span{color:#617565;font-size:13px}.fw-kpis{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:12px}.fw-card{border:1px solid #d6e5d8;border-radius:14px;background:#fff;padding:16px}.fw-card-label{font-size:13px;color:#56705d}.fw-card-value{font-size:26px;font-weight:700;color:#173b25;margin:7px 0}.fw-card-detail{font-size:12px;color:#6c7d70}.fw-alert,.fw-empty{border-radius:12px;padding:14px 16px}.fw-alert{background:#fff5e7;border:1px solid #f0cf98;color:#633d0b}.fw-empty{background:#eff8f0;border:1px solid #cfe2d2;color:#294a34}@media(max-width:760px){.fw-header{align-items:flex-start;flex-wrap:wrap}.fw-region{text-align:left;margin-left:0;width:100%}.fw-kpis{grid-template-columns:repeat(2,minmax(130px,1fr))}}
.gradio-container .tabs{display:grid!important;grid-template-columns:220px minmax(0,1fr)!important;gap:22px;align-items:start}.gradio-container .tabs>.tab-nav{display:flex!important;flex-direction:column!important;align-items:stretch!important;gap:6px;padding:66px 12px 12px!important;border:1px solid #cfe2d2;border-radius:14px;background:#f5faf5;position:sticky;top:12px;min-height:520px}.gradio-container .tabs>.tab-nav button{justify-content:flex-start!important;width:100%;padding:11px 13px!important;border-radius:9px!important}.gradio-container .tabs>.tabitem{min-width:0}#sidebar-refresh{position:fixed!important;left:30px!important;top:154px!important;z-index:10!important;width:194px!important}@media(max-width:760px){.gradio-container .tabs{display:block!important}.gradio-container .tabs>.tab-nav{position:static;display:flex!important;flex-direction:row!important;overflow-x:auto;min-height:auto;padding:10px!important;border-radius:10px;margin-bottom:14px}.gradio-container .tabs>.tab-nav button{width:auto;white-space:nowrap}#sidebar-refresh{position:static!important;width:100%!important;margin:8px 0!important}}
.gradio-container .tabs>.tab-nav button:nth-child(1)::before{content:'⌂  '}.gradio-container .tabs>.tab-nav button:nth-child(2)::before{content:'◫  '}.gradio-container .tabs>.tab-nav button:nth-child(3)::before{content:'◉  '}.gradio-container .tabs>.tab-nav button:nth-child(4)::before{content:'⌕  '}.gradio-container .tabs>.tab-nav button:nth-child(5)::before{content:'♧  '}.gradio-container .tabs>.tab-nav button:nth-child(6)::before{content:'⌖  '}.gradio-container .tabs>.tab-nav button:nth-child(7)::before{content:'↗  '}.gradio-container .tabs>.tab-nav button:nth-child(8)::before{content:'♟  '}.gradio-container .tabs>.tab-nav button:nth-child(9)::before{content:'▤  '}
#theme-toggle button{min-width:108px!important;height:44px!important;border-radius:24px!important;border:1px solid #bdd1c1!important;background:linear-gradient(90deg,#fff8df,#dfe7ff)!important;color:#345036!important;font-size:18px!important;box-shadow:inset 0 1px 3px #00000018!important}
html.fw-dark #sidebar-refresh button{color:#f1faf2!important}
/* Compact conversation-style application sidebar. The native tabs remain the
   working navigation controls; this only changes their visual hierarchy. */
.gradio-container .tabs{grid-template-columns:250px minmax(0,1fr)!important;gap:28px!important}.gradio-container .tabs>.tab-nav{background:#080a09!important;border:1px solid #1d2420!important;border-radius:16px!important;padding:178px 10px 14px!important;min-height:calc(100vh - 130px)!important;position:sticky!important;top:12px!important}.gradio-container .tabs>.tabitem{padding-top:58px!important}.gradio-container .tabs>.tab-nav::before{content:'🌲  ForestWatch';position:absolute;top:20px;left:18px;right:12px;color:#f4f7f4;font-size:15px;font-weight:650;line-height:1.55}.gradio-container .tabs>.tab-nav::after{content:'Satellite Intelligence & Forest Monitoring';position:absolute;top:48px;left:18px;right:12px;color:#9fb6a3;font-size:11px;line-height:1.35}.gradio-container .tabs>.tab-nav button{color:#e9efea!important;background:transparent!important;border:0!important;border-radius:10px!important;font-size:14px!important;font-weight:500!important;padding:10px 12px!important;min-height:40px!important}.gradio-container .tabs>.tab-nav button:hover{background:#202522!important}.gradio-container .tabs>.tab-nav button.selected,.gradio-container .tabs>.tab-nav button[aria-selected='true']{background:#272c29!important;color:#ffffff!important}.gradio-container .tabs>.tab-nav button::before{color:#d8e6da!important;font-size:17px!important}#sidebar-refresh{position:fixed!important;left:31px!important;top:122px!important;z-index:999!important;width:226px!important}#sidebar-refresh button{background:#1f7a48!important;border:0!important;color:#ffffff!important;border-radius:10px!important;font-weight:600!important;min-height:38px!important}#sidebar-refresh button:hover{background:#278c57!important}@media(max-width:760px){.gradio-container .tabs{display:block!important}.gradio-container .tabs>.tab-nav{background:#080a09!important;min-height:auto!important;padding:10px!important}.gradio-container .tabs>.tabitem{padding-top:0!important}.gradio-container .tabs>.tab-nav::before,.gradio-container .tabs>.tab-nav::after{display:none}#sidebar-refresh{position:static!important;width:100%!important;margin:8px 0!important}}
#overview-trend,#overview-trend .plot-container,#overview-trend .js-plotly-plot,#overview-trend .plotly-graph-div{width:100%!important;max-width:100%!important}#overview-trend .svg-container{width:100%!important}
/* The toggle adds fw-dark to the document, body, and Gradio host.  Keep the
   contrast rules here rather than relying on the browser colour scheme. */
html.fw-dark,body.fw-dark,gradio-app.fw-dark{background:#101412!important;color:#edf5ee!important}
html.fw-dark .gradio-container,body.fw-dark .gradio-container,gradio-app.fw-dark .gradio-container{background:#101412!important;color:#edf5ee!important}
html.fw-dark .gradio-container :is(h1,h2,h3,h4,p,span,label,th,td,.prose),body.fw-dark .gradio-container :is(h1,h2,h3,h4,p,span,label,th,td,.prose),gradio-app.fw-dark .gradio-container :is(h1,h2,h3,h4,p,span,label,th,td,.prose){color:#edf5ee!important}
html.fw-dark .gradio-container :is(.block,.form,.panel,.wrap,.gr-box),body.fw-dark .gradio-container :is(.block,.form,.panel,.wrap,.gr-box),gradio-app.fw-dark .gradio-container :is(.block,.form,.panel,.wrap,.gr-box){background-color:#171c19!important;border-color:#344039!important}
html.fw-dark .fw-header,body.fw-dark .fw-header,gradio-app.fw-dark .fw-header{background:linear-gradient(120deg,#16251b,#1c2920)!important;border-color:#385642!important}
html.fw-dark .fw-title,html.fw-dark .fw-card-value,body.fw-dark .fw-title,body.fw-dark .fw-card-value,gradio-app.fw-dark .fw-title,gradio-app.fw-dark .fw-card-value{color:#f2faf3!important}
html.fw-dark .fw-subtitle,html.fw-dark .fw-card-label,html.fw-dark .fw-card-detail,html.fw-dark .fw-region,html.fw-dark .fw-region span,body.fw-dark .fw-subtitle,body.fw-dark .fw-card-label,body.fw-dark .fw-card-detail,body.fw-dark .fw-region,body.fw-dark .fw-region span,gradio-app.fw-dark .fw-subtitle,gradio-app.fw-dark .fw-card-label,gradio-app.fw-dark .fw-card-detail,gradio-app.fw-dark .fw-region,gradio-app.fw-dark .fw-region span{color:#bfd0c2!important}
html.fw-dark .fw-card,body.fw-dark .fw-card,gradio-app.fw-dark .fw-card{background:#18201b!important;border-color:#385642!important}
html.fw-dark #theme-toggle button,body.fw-dark #theme-toggle button,gradio-app.fw-dark #theme-toggle button{background:linear-gradient(90deg,#313d63,#1d2440)!important;color:#f6f8ff!important;border-color:#7280ae!important}
"""

@lru_cache(maxsize=1)
def _cached_data(): return scan_dataset(settings.dataset_path)
def data(): return _cached_data().copy()
@lru_cache(maxsize=1)
def _cached_timeline(): return health_scores(ndvi_timeseries(_cached_data()))
def timeline(): return _cached_timeline().copy()

def refresh_dataset_cache():
    _cached_data.cache_clear()
    _cached_timeline.cache_clear()

def as_percent(value):
    return round(float(value) * 100, 2) if pd.notna(value) else None
def _active_alerts():
    """Read alert records without creating or changing any alert state."""
    session = SessionLocal()
    try:
        return session.query(Alert).filter(Alert.status.in_(["OPEN", "IN PROGRESS"])).order_by(Alert.date.desc()).all()
    finally:
        session.close()

def _status(label, ready, detail):
    state = "ready" if ready else "limited"
    icon = "●" if ready else "●"
    text = "Ready" if ready else "Limited"
    return [label, text, detail, state, icon]

def overview():
    """Compose the Overview solely from the existing dataset and backend outputs."""
    inventory, ts = data(), timeline().dropna(subset=["ndvi_mean"])
    alerts = _active_alerts()
    if ts.empty:
        cards = [("Latest NDVI", "—", "Vegetation health indicator"), ("Forest health score", "—", "System monitoring score"),
                 ("Healthy vegetation", "—", "Coverage above configured threshold"), ("Satellite observations", "0", "Dated NDVI scenes")]
        latest_date = "No dated observation"
    else:
        latest = ts.sort_values("date").iloc[-1]
        try:
            coverage = ndvi_statistics(latest.path)["healthy_vegetation_percentage"]
        except (OSError, ValueError):
            coverage = None
        cards = [("Latest NDVI", f"{as_percent(latest.ndvi_mean):.2f}%", "Vegetation health indicator"),
                 ("Forest health score", f"{float(latest.forest_health_score):.2f}%", str(latest.health_category)),
                 ("Healthy vegetation", f"{coverage:.2f}%" if coverage is not None else "—", "Coverage above configured threshold"),
                 ("Satellite observations", str(len(ts)), "Dated NDVI scenes")]
        latest_date = str(latest.date)
    card_html = "".join(f"<div class='fw-card'><div class='fw-card-label'>{escape(label)}</div><div class='fw-card-value'>{escape(value)}</div><div class='fw-card-detail'>{escape(detail)}</div></div>" for label, value, detail in cards)
    header = f"""
    <div class='fw-header'><div class='fw-mark'>🌲</div><div><div class='fw-title'>ForestWatch</div><div class='fw-subtitle'>Satellite-Based Forest Monitoring &amp; Prediction</div></div><div class='fw-region'><b>{escape(settings.aoi_name)}</b><br><span>Latest observation: {escape(latest_date)}</span></div></div>
    <div class='fw-kpis'>{card_html}</div>"""
    if alerts:
        newest = alerts[0]
        alert_html = f"<div class='fw-alert'><b>{len(alerts)} active alert(s)</b><br><span>{escape(str(newest.date))} · {escape(newest.severity)} · {escape(newest.reason)}</span></div>"
    else:
        alert_html = "<div class='fw-empty'>No recent alerts detected.</div>"
    return header, plot_ts(), health_distribution_plot(ts), alert_html
def dates(): return data().loc[lambda x:x.category=='ndvi','date'].dropna().sort_values().unique().tolist()
def view_ndvi(date):
    row=data().query("category == 'ndvi' and date == @date").iloc[0]
    try:
        # Return decoded pixels so Gradio does not retry a corrupt file path.
        import numpy as np
        preview = (prepared_rgb(row.path) * 255).astype(np.uint8)
        stats = ndvi_statistics(row.path)
        return preview, ndvi_scene_table(stats), stats["mean"], stats["std"], stats["healthy_vegetation_percentage"]
    except (OSError, ValueError) as exc:
        return None, pd.DataFrame([["Scene status", f"Unable to read this image: {exc}"]], columns=["Measure", "Value"]), None, None, None

def ndvi_scene_table(stats):
    return pd.DataFrame([
        ["Average NDVI", f"{as_percent(stats['mean']):.2f}%"], ["Middle NDVI value", f"{as_percent(stats['median']):.2f}%"],
        ["Lower NDVI range (2nd percentile)", f"{as_percent(stats['display_low']):.2f}%"], ["Upper NDVI range (98th percentile)", f"{as_percent(stats['display_high']):.2f}%"],
        ["Variation across scene", f"{as_percent(stats['std']):.2f}%"], ["Valid image pixels", f"{stats['valid_pixel_percentage']:.2f}%"],
        ["Lower-vegetation pixels", f"{stats['low_vegetation_percentage']:.2f}%"], ["Higher-vegetation pixels", f"{stats['healthy_vegetation_percentage']:.2f}%"],
    ], columns=["Scene measure", "Value"])

def health_distribution_plot(series=None):
    frame = timeline() if series is None else series
    frame = frame.dropna(subset=["health_category"])
    if frame.empty:
        return None
    counts = frame["health_category"].value_counts().rename_axis("Health category").reset_index(name="Observations")
    return px.pie(counts, names="Health category", values="Observations", hole=0.48,
                  color="Health category", color_discrete_map={"Higher":"#2e8b57", "Watch":"#e3a42c", "Low":"#d9534f"},
                  title="Observation health distribution")

def vegetation_coverage_plot(stats):
    healthy = max(0.0, float(stats["healthy_vegetation_percentage"]))
    low = max(0.0, float(stats["low_vegetation_percentage"]))
    middle = max(0.0, 100.0 - healthy - low)
    chart = pd.DataFrame({"Vegetation class":["Higher vegetation", "Middle vegetation", "Lower vegetation"],
                          "Coverage":[healthy, middle, low]})
    return px.pie(chart, names="Vegetation class", values="Coverage", hole=0.48,
                  color="Vegetation class", color_discrete_map={"Higher vegetation":"#2e8b57", "Middle vegetation":"#e3a42c", "Lower vegetation":"#d9534f"},
                  title="Vegetation coverage")

def view_ndvi_with_chart(date):
    image, table, mean, std, healthy = view_ndvi(date)
    if image is None:
        return image, table, mean, std, healthy, None
    row = data().query("category == 'ndvi' and date == @date").iloc[0]
    return image, table, mean, std, healthy, vegetation_coverage_plot(ndvi_statistics(row.path))
def visual_ndvi_timeline():
    """Add display-only estimates where no March–October scene exists.

    Two values per absent month are linearly interpolated from surrounding real
    observations. They are intentionally kept out of the analytical timeline,
    alerts, reports, training, and date selectors.
    """
    actual = timeline().dropna(subset=["ndvi_mean"])[["date", "ndvi_mean"]].copy()
    if actual.empty:
        return actual
    actual["date"] = pd.to_datetime(actual["date"])
    actual = actual.sort_values("date").drop_duplicates("date")
    actual["point_type"] = "Actual satellite observation"
    extra = []
    for year in range(actual.date.min().year, actual.date.max().year + 1):
        for month in range(3, 11):
            month_mask = (actual.date.dt.year == year) & (actual.date.dt.month == month)
            if month_mask.any():
                continue
            for day in (10, 20):
                estimate_date = pd.Timestamp(year=year, month=month, day=day)
                if actual.date.min() < estimate_date < actual.date.max():
                    estimate = float(np.interp(estimate_date.value, actual.date.astype("int64"), actual.ndvi_mean))
                    extra.append({"date": estimate_date, "ndvi_mean": estimate, "point_type": "Estimated from surrounding observations"})
    return pd.concat([actual, pd.DataFrame(extra)], ignore_index=True).sort_values("date")

def plot_ts():
    points = visual_ndvi_timeline()
    if points.empty:
        return None
    points["ndvi_percent"] = points["ndvi_mean"] * 100
    actual = points[points.point_type == "Actual satellite observation"]
    estimated = points[points.point_type != "Actual satellite observation"]
    figure = go.Figure()
    figure.add_scatter(x=points.date, y=points.ndvi_percent, mode="lines", name="Connected trend", line={"color":"#8b9ab0", "width":2}, hoverinfo="skip")
    figure.add_scatter(x=actual.date, y=actual.ndvi_percent, mode="markers", name="Actual satellite observation", marker={"color":"#356ae6", "size":7}, customdata=actual[["point_type"]], hovertemplate="%{x|%d %b %Y}<br>NDVI: %{y:.2f}%<br>%{customdata[0]}<extra></extra>")
    if not estimated.empty:
        figure.add_scatter(x=estimated.date, y=estimated.ndvi_percent, mode="markers", name="Estimated point (no image)", marker={"color":"#e3a42c", "size":8, "symbol":"diamond"}, customdata=estimated[["point_type"]], hovertemplate="%{x|%d %b %Y}<br>Estimated NDVI: %{y:.2f}%<br>%{customdata[0]}<extra></extra>")
    figure.update_layout(title="Historical NDVI observations", xaxis_title="Observation date", yaxis_title="NDVI (%)", autosize=True, width=None, height=520, margin={"l":65,"r":30,"t":70,"b":65}, legend_title_text="Point type")
    return figure
def change_distribution_plot(result):
    chart = pd.DataFrame({"Change class":["Vegetation decrease", "Stable", "Vegetation increase"],
                          "Coverage":[result["decrease_pixel_percentage"], result["stable_pixel_percentage"], result["increase_pixel_percentage"]]})
    return px.pie(chart, names="Change class", values="Coverage", hole=0.48,
                  color="Change class", color_discrete_map={"Vegetation decrease":"#d9534f", "Stable":"#66788a", "Vegetation increase":"#2e8b57"},
                  title="Selected-period change distribution")

def run_change(earlier_date, later_date):
    try:
        aligned_dates = {"2024-12-23", "2025-12-28"}
        if {earlier_date, later_date} == aligned_dates and multisatellite_available():
            preview, result = aligned_change_detection(earlier_date, later_date)
            table = pd.DataFrame([
                ["Vegetation decrease area", f"{result['decrease_area_hectares']:.2f} hectares"],
                ["Vegetation increase area", f"{result['increase_area_hectares']:.2f} hectares"],
                ["Stable vegetation area", f"{result['stable_area_hectares']:.2f} hectares"],
                ["Analysis resolution", result["resolution"]],
            ], columns=["What was measured", "Result"])
            chart_result = {"decrease_pixel_percentage": result["decrease_pixels"] / max(sum(result[k] for k in ("decrease_pixels", "increase_pixels", "stable_pixels")), 1) * 100,
                            "increase_pixel_percentage": result["increase_pixels"] / max(sum(result[k] for k in ("decrease_pixels", "increase_pixels", "stable_pixels")), 1) * 100,
                            "stable_pixel_percentage": result["stable_pixels"] / max(sum(result[k] for k in ("decrease_pixels", "increase_pixels", "stable_pixels")), 1) * 100}
            return table, "### Change result\nVegetation decrease, increase, and stable area were calculated from the aligned 10 m project grid.", change_distribution_plot(chart_result)
        if earlier_date == later_date:
            raise ValueError("Choose two different dates for change detection.")
        earlier = data().query("category == 'ndvi' and date == @earlier_date").iloc[0]
        later = data().query("category == 'ndvi' and date == @later_date").iloc[0]
        result = change_detection(earlier.path, later.path)
        import numpy as np
        difference = result["difference_map"]
        bound = max(float(np.nanpercentile(np.abs(difference), 98)), 1e-6)
        normalised = np.clip(difference / bound, -1, 1)
        preview = np.zeros((*normalised.shape, 3), dtype=np.uint8)
        preview[..., 0] = np.where(normalised < 0, -normalised * 255, 0).astype(np.uint8)
        preview[..., 1] = np.where(normalised >= 0, normalised * 255, 0).astype(np.uint8)
        table = pd.DataFrame([
            ["Vegetation decrease coverage", f"{result['decrease_pixel_percentage']:.2f}% of image"],
            ["Vegetation increase coverage", f"{result['increase_pixel_percentage']:.2f}% of image"],
            ["Stable vegetation coverage", f"{result['stable_pixel_percentage']:.2f}% of image"],
        ], columns=["What was measured", "Result"])
        return table, "### Change result\nVegetation decrease, increase, and stable coverage were calculated for the selected scenes.", change_distribution_plot(result)
    except (IndexError, FileNotFoundError, OSError, ValueError) as exc:
        return pd.DataFrame([["Status", f"Comparison unavailable: {exc}"]], columns=["What was measured", "Result"]), "### Change result\nSelect two different project dates, then run the area calculation.", None

def export_change_csv(earlier_date, later_date):
    """Export only pixels classified as vegetation increase or decrease."""
    import numpy as np
    settings.ensure_output_dirs()
    threshold = abs(settings.decrease_threshold)
    aligned_dates = {"2024-12-23", "2025-12-28"}
    if {earlier_date, later_date} == aligned_dates and multisatellite_available():
        cube = np.load(settings.project_data_path / "Gorewada_Temporal_Dataset_2024_2025.npy", mmap_mode="r")
        before, after = (np.asarray(cube[..., 2]), np.asarray(cube[..., 5])) if earlier_date == "2024-12-23" else (np.asarray(cube[..., 5]), np.asarray(cube[..., 2]))
        valid_mask = np.load(settings.project_data_path / "Gorewada_HighConfidence_Change_Label_2024_2025.npy", mmap_mode="r") != 255
        area_value, area_unit = 100, "square metres"
    else:
        earlier = data().query("category == 'ndvi' and date == @earlier_date").iloc[0]
        later = data().query("category == 'ndvi' and date == @later_date").iloc[0]
        before, after = ndvi_proxy(earlier.path), ndvi_proxy(later.path)
        if before.shape != after.shape:
            raise ValueError("The selected images have different dimensions and cannot be exported as pixel changes.")
        area_value, area_unit = None, "not available"
        valid_mask = np.ones(before.shape, dtype=bool)
    difference = after - before
    decrease, increase = (difference <= -threshold) & valid_mask, (difference >= threshold) & valid_mask
    rows, columns = np.where(decrease | increase)
    types = np.where(decrease[rows, columns], "Vegetation decrease", "Vegetation increase")
    changed = pd.DataFrame({"row": rows, "column": columns, "change_type": types,
                            "ndvi_before_percent": (before[rows, columns] * 100).round(2), "ndvi_after_percent": (after[rows, columns] * 100).round(2),
                            "change_magnitude_percent": (np.abs(difference[rows, columns]) * 100).round(2)})
    if area_value is not None:
        changed["pixel_area"] = area_value
        changed["pixel_area_unit"] = area_unit
    path = settings.output_path / "metrics" / f"change_{earlier_date}_to_{later_date}.csv"
    changed.to_csv(path, index=False, float_format="%.2f")
    return str(path), f"Created CSV with {len(changed):,} changed pixels."

def inspect_uploads(files):
    from pathlib import Path
    from PIL import Image
    rows = []
    for file in files or []:
        path = Path(file)
        try:
            with Image.open(path) as image:
                rows.append([path.name, path.suffix.lower(), f"{image.width} × {image.height}", ", ".join(image.getbands()), "Ready for review"])
        except (OSError, ValueError):
            rows.append([path.name, path.suffix.lower(), "—", "—", "File added; raster processing is required"])
    return pd.DataFrame(rows, columns=["Uploaded file", "Format", "Dimensions", "Channels", "Status"])

def add_uploads_to_dataset(files, image_type, acquisition_date):
    """Persist user uploads as additive dated dataset files; never overwrite sources."""
    if not files:
        return "### No images added\nSelect one or more images first."
    try:
        date = datetime.strptime(str(acquisition_date), "%Y-%m-%d").date().isoformat()
    except ValueError:
        return "### No images added\nEnter the acquisition date as YYYY-MM-DD."
    category_tokens = {"Sentinel-1 VV": "VV", "Sentinel-1 VH": "VH", "Sentinel-2 NDVI": "NDVI", "Sentinel-2 True Color": "TrueColor"}
    token = category_tokens[image_type]
    destination = settings.dataset_path / "imported" / token
    destination.mkdir(parents=True, exist_ok=True)
    added = []
    for index, file in enumerate(files, start=1):
        source = Path(file)
        suffix = source.suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".jp2"}:
            continue
        target = destination / f"Gorewada_{token}_{date}_{index}{suffix}"
        copy_index = 2
        while target.exists():
            target = destination / f"Gorewada_{token}_{date}_{index}_{copy_index}{suffix}"
            copy_index += 1
        shutil.copy2(source, target)
        added.append(target.name)
    if not added:
        return "### No images added\nUse a supported satellite-image format: JPG, PNG, TIFF, or JP2."
    refresh_dataset_cache()
    return f"### Images added to dataset\nAdded **{len(added)}** {image_type} image(s) dated **{date}** to `dataset/imported/{token}`."

def potential_degradation(earlier_date, later_date):
    table, message, _ = run_change(earlier_date, later_date)
    if table.iloc[0, 0] == "Status":
        return table, message
    finding = table.copy()
    finding.insert(0, "Detection", "Potential Forest Degradation")
    return finding, "### Detection result\nPotential vegetation change is shown for the selected period. Review the changed-data CSV for pixel-level records."

def gis_grid_view(earlier_date, later_date):
    """Display real true-colour imagery plus actual NDVI-index change metrics."""
    try:
        if earlier_date == later_date:
            raise ValueError("Choose two different observation dates.")
        earlier = data().query("category == 'ndvi' and date == @earlier_date").iloc[0]
        later = data().query("category == 'ndvi' and date == @later_date").iloc[0]
        earlier_true = data().query("category == 'true_color' and date == @earlier_date").iloc[0]
        later_true = data().query("category == 'true_color' and date == @later_date").iloc[0]
        result = change_detection(earlier.path, later.path)
        actual_change = result["difference_map"]
        figure = px.imshow(actual_change, color_continuous_scale="RdYlGn", zmin=-0.4, zmax=0.4,
                           title=f"NDVI-index change: {earlier_date} to {later_date}",
                           labels={"x": "Grid column", "y": "Grid row", "color": "NDVI index change"})
        before_image = (prepared_rgb(earlier_true.path) * 255).astype("uint8")
        after_image = (prepared_rgb(later_true.path) * 255).astype("uint8")
        return before_image, after_image, figure, (f"### Selected period: {earlier_date} to {later_date}\n"
                         f"Decrease: **{result['decrease_pixel_percentage']:.2f}%** · "
                         f"Increase: **{result['increase_pixel_percentage']:.2f}%** · "
                         f"Stable: **{result['stable_pixel_percentage']:.2f}%**\n\n"
                         "Map colours show the actual NDVI-index difference, not a −100% to +100% normalised display.")
    except (IndexError, OSError, ValueError) as exc:
        return None, None, None, f"### GIS view unavailable\n{exc}"

def generate_report(period, report_id):
    try:
        value = str(report_id).strip()
        dates_series = timeline().copy()
        dates_series["date"] = pd.to_datetime(dates_series["date"])
        if period == "weekly":
            start = pd.to_datetime(f"{value}-1", format="%G-W%V-%u")
            end = start + pd.Timedelta(days=6)
            selected = dates_series[(dates_series.date >= start) & (dates_series.date <= end)]
            expected = "YYYY-Www (example: 2025-W05)"
        elif period == "monthly":
            start = pd.to_datetime(value, format="%Y-%m")
            end = start + pd.offsets.MonthEnd(1)
            selected = dates_series[(dates_series.date >= start) & (dates_series.date <= end)]
            expected = "YYYY-MM (example: 2025-02)"
        else:
            start = pd.to_datetime(value, format="%Y")
            selected = dates_series[dates_series.date.dt.year == start.year]
            expected = "YYYY (example: 2025)"
        report_path = generate_html_report(selected, [], period, value.replace("/", "-"))
        csv_path = settings.output_path / "metrics" / f"ndvi_metrics_{period}_{value.replace('/', '-')}.csv"
        percentage_display(selected.drop(columns=["data_quality"], errors="ignore")).to_csv(csv_path, index=False, float_format="%.2f")
        return str(report_path), str(csv_path), f"{period.title()} report for **{value}** generated with {len(selected)} observation(s)."
    except (TypeError, ValueError):
        return None, None, f"Enter the reporting period as {expected if 'expected' in locals() else 'YYYY-MM'}"

def refresh_health_alerts():
    alerts = check_health_alerts(timeline())
    return alerts, f"### Alert threshold: health score below {HEALTH_ALERT_THRESHOLD:.0f}\nFound **{len(alerts)}** alert(s)."

def change_alert_status(alert_date, status):
    result = update_alert_status(alert_date, status)
    alerts, _ = refresh_health_alerts()
    return alerts, result

def alert_date_choices():
    alerts = check_health_alerts(timeline())
    return gr.update(choices=alerts["Date"].tolist() if not alerts.empty else [])
def train(existing_model, epochs):
    epochs = max(0, int(epochs))
    if epochs == 0:
        if existing_model is None:
            return None, pd.DataFrame([["Status", "Enter at least 1 epoch to train the model."]], columns=["Evaluation measure", "Result"]), None, None, None, None, "### Model not trained\nSet training epochs to 1 or more."
        return existing_model, pd.DataFrame([["Status", "No new trees were added because epochs is 0."]], columns=["Evaluation measure", "Result"]), None, None, None, None, "### Model unchanged\nSet training epochs to 1 or more to improve the model."
    trees_added = epochs * 200
    trained, metrics, pred=train_ndvi_model(timeline(), "Random Forest", existing_model, additional_trees=trees_added)
    fig=px.line(pred,x="date",y=["actual","predicted"],title="Actual vs predicted NDVI") if not pred.empty else None
    details = pd.DataFrame([
        ["Model", metrics["model"]], ["Forest trees trained", metrics["trees"]], ["Training observations", metrics["training_rows"]], ["Test observations", metrics["test_rows"]],
        ["Average error (MAE)", round(metrics["mae"], 2) if metrics["mae"] is not None else "Not available"],
        ["Root mean square error (RMSE)", round(metrics["rmse"], 2) if metrics["rmse"] is not None else "Not available"],
        ["Fit score (R²)", round(metrics["r2"], 2) if metrics["r2"] is not None else "Not available"],
    ], columns=["Evaluation measure", "Result"])
    return trained, details, fig, metrics.get("mae"), metrics.get("rmse"), metrics.get("r2"), f"### Model trained\nCompleted **{epochs} epoch(s)** by adding **{trees_added} trees** using the same historical data."

def future_forecast(model, horizon):
    try:
        # Training remains a backend concern: always create a fresh model from
        # the current dataset before each user-requested forecast.
        model, _, _ = train_ndvi_model(timeline(), "Random Forest", additional_trees=600)
        forecast = forecast_future_ndvi(model, timeline(), horizon)
        historical = timeline().dropna(subset=["ndvi_mean"])[["date", "ndvi_mean"]].rename(columns={"date":"observation", "ndvi_mean":"value"})
        forecast_plot = forecast.rename(columns={"forecast_date":"observation", "predicted_ndvi":"value"})
        historical["value"] *= 100; forecast_plot["value"] *= 100
        historical["series"] = "Historical NDVI"; forecast_plot["series"] = "Future prediction"
        combined = pd.concat([historical, forecast_plot])
        fig = px.scatter(
            combined, x="observation", y="value", color="series", symbol="series",
            title="Historical observations and future NDVI prediction",
            labels={"observation": "Date", "value": "NDVI (%)", "series": "Series"},
        )
        # Start the forecast line at the last real observation, then join every
        # future prediction point in chronological order.
        forecast_line = pd.concat([historical.tail(1), forecast_plot], ignore_index=True)
        fig.add_scatter(
            x=forecast_line["observation"], y=forecast_line["value"], mode="lines+markers",
            name="Future prediction trend", line={"color": "#ef553b"}, marker={"color": "#ef553b", "size": 7},
        )
        forecast_display = forecast.assign(predicted_ndvi_percent=(forecast.predicted_ndvi * 100).round(2)).drop(columns="predicted_ndvi")
        return model, forecast_display, fig, f"Generated {len(forecast)} future prediction(s)."
    except ValueError as exc:
        return model, None, None, str(exc)

def filter_inventory(category, start_date, end_date):
    inventory = data()
    if category != "All imagery":
        inventory = inventory[inventory.category == category]
    if start_date:
        inventory = inventory[inventory.date >= start_date]
    if end_date:
        inventory = inventory[inventory.date <= end_date]
    return inventory[["date", "category", "filename", "format", "size_bytes", "relative_path"]]

def build_app():
    initial_dates=dates()
    with gr.Blocks(title="ForestWatch") as demo:
        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                gr.Markdown("# 🌲 ForestWatch\nSatellite-Based Forest Monitoring & Prediction")
            with gr.Column(scale=1, min_width=120):
                theme_toggle=gr.Button("☀  /  ☾", elem_id="theme-toggle")
        refresh=gr.Button("↻ Refresh Data", variant="primary", elem_id="sidebar-refresh")
        with gr.Tab("Overview"):
            dashboard_header = gr.HTML()
            gr.Markdown("### Forest Health Trend\nBlue circles are real satellite observations. For March–October months with no imagery, two amber diamonds are display-only estimates interpolated from the surrounding real observations.")
            overview_trend = gr.Plot(label="Forest Health Trend", elem_id="overview-trend")
            overview_health_pie = gr.Plot(label="Health-category distribution")
            gr.Markdown("### Recent Alerts")
            overview_alerts = gr.HTML()
            refresh.click(overview, outputs=[dashboard_header, overview_trend, overview_health_pie, overview_alerts])
            demo.load(overview, outputs=[dashboard_header, overview_trend, overview_health_pie, overview_alerts])
            theme_toggle.click(
                fn=None,
                outputs=theme_toggle,
                js="""() => {
                    const app = document.querySelector('gradio-app');
                    const nextDark = !document.documentElement.classList.contains('fw-dark');
                    [document.documentElement, document.body, app].filter(Boolean).forEach((node) => {
                        node.classList.toggle('fw-dark', nextDark);
                    });
                    try { localStorage.setItem('forestwatch-theme', nextDark ? 'dark' : 'light'); } catch (_) {}
                    return nextDark ? '☀  Light mode' : '☾  Dark mode';
                }""",
            )
        with gr.Tab("Satellite Image Upload"):
            gr.Markdown("## Satellite Image Upload\nFirst prepare the local dataset. Preparation validates images, corrects orientation, standardizes RGB bands, and normalizes pixels for analysis without modifying the original satellite files. Prepared copies are saved in `data/processed/images`.")
            preprocessing_status = gr.Markdown("Dataset preparation has not been run in this session.")
            preprocess_table = gr.Dataframe(interactive=False, label="Preprocessing results")
            gr.Button("Preprocess local satellite data", variant="primary").click(
                preprocessing_summary, outputs=[preprocess_table, preprocessing_status]
            )
            gr.Markdown("### Add satellite images\nChoose the satellite image type and acquisition date, then add uploaded image files to the ForestWatch dataset.")
            uploads = gr.File(file_count="multiple", file_types=["image", ".tif", ".tiff", ".jp2"], label="Satellite images")
            with gr.Row():
                upload_type = gr.Dropdown(["Sentinel-1 VV", "Sentinel-1 VH", "Sentinel-2 NDVI", "Sentinel-2 True Color"], value="Sentinel-2 NDVI", label="Image type")
                upload_date = gr.Textbox(placeholder="YYYY-MM-DD", label="Acquisition date")
            upload_table = gr.Dataframe(interactive=False, label="Uploaded-image details")
            with gr.Row():
                gr.Button("Inspect uploaded images").click(inspect_uploads, uploads, upload_table)
                add_button = gr.Button("Add images to dataset", variant="primary")
            upload_status = gr.Markdown()
            add_button.click(add_uploads_to_dataset, [uploads, upload_type, upload_date], upload_status)
        with gr.Tab("NDVI Analysis"):
            gr.Markdown("## NDVI Analysis\n1. Choose a date.  2. Load the satellite scene.  3. Read the plain-language scene summary and historical trend. Amber diamonds in the trend are display-only estimates for missing March–October imagery.")
            choice=gr.Dropdown(initial_dates, value=initial_dates[-1] if initial_dates else None, label="Observation date")
            inspect = gr.Button("Load NDVI observation", variant="primary")
            with gr.Row():
                image=gr.Image(label="NDVI image", height=430)
                with gr.Column():
                    mean = gr.Number(label="Mean NDVI")
                    std = gr.Number(label="NDVI variability")
                    healthy = gr.Number(label="Healthy vegetation (%)")
                    stats=gr.Dataframe(interactive=False, label="Scene statistics")
            coverage_pie = gr.Plot(label="Vegetation coverage")
            inspect.click(view_ndvi_with_chart,choice,[image,stats,mean,std,healthy,coverage_pie])
            trend_button = gr.Button("Load historical NDVI trend")
            trend_plot = gr.Plot(label="Historical trend")
            trend_button.click(plot_ts, outputs=trend_plot)
        with gr.Tab("Change Detection"):
            gr.Markdown("## NDVI Change Detection")
            with gr.Row():
                earlier_date = gr.Dropdown(initial_dates, value=initial_dates[0] if initial_dates else None, label="Earlier observation date")
                later_date = gr.Dropdown(initial_dates, value=initial_dates[-1] if initial_dates else None, label="Later observation date")
            change_table=gr.Dataframe(interactive=False,label="Change summary"); change_message=gr.Markdown()
            change_pie = gr.Plot(label="Change distribution")
            gr.Button("Calculate vegetation change", variant="primary").click(run_change,[earlier_date,later_date], [change_table,change_message,change_pie])
            gr.Markdown("### Download changed data")
            download_button = gr.Button("Create changed-data CSV")
            changed_csv = gr.File(label="Changed-data CSV")
            csv_status = gr.Markdown()
            download_button.click(export_change_csv, [earlier_date, later_date], [changed_csv, csv_status])
        with gr.Tab("Potential Degradation Detection"):
            gr.Markdown("## Potential Forest Degradation\nChoose two observations to identify potential vegetation decrease or increase. This is a monitoring signal, not a confirmed deforestation finding.")
            with gr.Row():
                detect_earlier = gr.Dropdown(initial_dates, value=initial_dates[0] if initial_dates else None, label="Earlier observation date")
                detect_later = gr.Dropdown(initial_dates, value=initial_dates[-1] if initial_dates else None, label="Later observation date")
            detection_table = gr.Dataframe(interactive=False, label="Detection results")
            detection_message = gr.Markdown()
            gr.Button("Run potential-degradation detection", variant="primary").click(potential_degradation, [detect_earlier, detect_later], [detection_table, detection_message])
        with gr.Tab("GIS Visualization"):
            gr.Markdown("## GIS Visualization\nCompare the actual true-colour satellite scenes for two dates. The calculated NDVI-index change is shown separately below.")
            with gr.Row():
                gis_earlier = gr.Dropdown(initial_dates, value=initial_dates[0] if initial_dates else None, label="Earlier observation date")
                gis_later = gr.Dropdown(initial_dates, value=initial_dates[-1] if initial_dates else None, label="Later observation date")
            with gr.Row():
                gis_before_image = gr.Image(label="Actual true-colour scene — earlier date", height=430)
                gis_after_image = gr.Image(label="Actual true-colour scene — later date", height=430)
            gis_plot = gr.Plot(label="Actual NDVI-index change grid")
            gis_info = gr.Markdown()
            gr.Button("Load GIS visualization", variant="primary").click(gis_grid_view, [gis_earlier, gis_later], [gis_before_image, gis_after_image, gis_plot, gis_info])
        with gr.Tab("Prediction"):
            gr.Markdown("## Future NDVI Prediction\nChoose how many months to predict, then generate the forecast.")
            trained_model = gr.State(value=None)
            with gr.Row():
                horizon=gr.Slider(1,24,value=6,step=1,label="Number of future monthly predictions")
                forecast_button=gr.Button("Generate future prediction", variant="primary")
            forecast_table=gr.Dataframe(headers=["forecast_date","predicted_ndvi_percent"],interactive=False,label="Future NDVI predictions (%)")
            forecast_plot=gr.Plot(label="Historical vs future NDVI")
            forecast_status=gr.Markdown()
            forecast_button.click(future_forecast,[trained_model,horizon],[trained_model,forecast_table,forecast_plot,forecast_status])
        with gr.Tab("Alerts"):
            gr.Markdown("## Vegetation Health Alerts")
            alerts_button = gr.Button("Check vegetation health alerts", variant="primary")
            alerts_table = gr.Dataframe(interactive=False, label="Active health alerts")
            alerts_status = gr.Markdown()
            alerts_button.click(refresh_health_alerts, outputs=[alerts_table, alerts_status])
            with gr.Row():
                alert_date = gr.Dropdown([], label="Alert date")
                alert_workflow = gr.Dropdown(["OPEN", "IN PROGRESS", "CLOSED"], value="OPEN", label="Alert status")
                update_status = gr.Button("Update alert status")
            update_message = gr.Markdown()
            # The date list refreshes from the alerts table after a check.
            alerts_button.click(alert_date_choices, outputs=alert_date)
            update_status.click(change_alert_status, [alert_date, alert_workflow], [alerts_table, update_message])
        with gr.Tab("Reports"):
            gr.Markdown("## Forest Monitoring Reports\nChoose the report type and enter the exact period you want to report on.")
            with gr.Row():
                report_period = gr.Radio(["weekly", "monthly", "yearly"], value="monthly", label="Report type")
                report_id = gr.Textbox(value="2025-02", label="Week / month / year", placeholder="Weekly: 2025-W05 · Monthly: 2025-02 · Yearly: 2025")
                report_button = gr.Button("Generate report", variant="primary")
            html_report = gr.File(label="Monitoring report (HTML)")
            metrics_report = gr.File(label="NDVI metrics (CSV)")
            report_status = gr.Markdown()
            report_button.click(generate_report, [report_period, report_id], [html_report, metrics_report, report_status])
    return demo

if __name__ == "__main__": build_app().launch(css=DASHBOARD_CSS)
