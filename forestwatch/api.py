from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np
import pandas as pd
import json
from .config import settings
from .dataset_scanner import scan_dataset, inventory_summary
from .analysis import ndvi_timeseries, health_scores, ndvi_statistics, change_detection
from .prediction import train_ndvi_model, forecast_future_ndvi
from .schemas import ChangeRequest, PredictionRequest, Credentials
from .auth import register, login, user_from_token
from .notifications import deliver_new_alerts
from .model_comparison import compare_cnn_yolo
from .database import initialise_database, SessionLocal, Alert
from .reports import export_metrics, generate_html_report
from .alerts import check_health_alerts, update_alert_status
from pathlib import Path
from datetime import datetime
import shutil
from .multisatellite import available as multisatellite_available, comparison as multisatellite_comparison

app=FastAPI(title="ForestWatch API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])
initialise_database()

def inventory(): return scan_dataset(settings.dataset_path)
def series(): return health_scores(ndvi_timeseries(inventory()))
def json_records(frame):
    """FastAPI-safe records: JavaScript JSON cannot represent NaN or NaT."""
    clean = frame.copy().replace([np.inf, -np.inf], np.nan)
    # Pandas serialises floating NaN as JSON null; loading it again gives plain
    # Python None values that Starlette can safely return.
    return json.loads(clean.to_json(orient="records", date_format="iso"))
def record_for_date(date, category="ndvi"):
    rows=inventory(); subset=rows[(rows.category==category)&(rows.date==date)]
    if subset.empty: raise HTTPException(404, detail=f"No {category} image found for {date}")
    return subset.iloc[0]

@app.get("/")
def root():
    return {
        "application": "ForestWatch API",
        "status": "running",
        "docs": "/docs",
        "gradio": "Run `python -m forestwatch.gradio_app` in a second terminal to start the visual interface.",
    }

@app.get("/api/health")
def health(): return {"status":"ok","application":"ForestWatch"}
@app.post("/api/auth/register")
def auth_register(credentials: Credentials):
    try: return register(credentials.email, credentials.password)
    except ValueError as exc: raise HTTPException(422, detail=str(exc))
@app.post("/api/auth/login")
def auth_login(credentials: Credentials):
    try: return login(credentials.email, credentials.password)
    except ValueError as exc: raise HTTPException(401, detail=str(exc))
@app.get("/api/aoi")
def aoi(): return {"name":settings.aoi_name}
@app.get("/api/dataset/inventory")
def dataset_inventory():
    data=inventory(); return {"summary":inventory_summary(data),"files":data.to_dict("records")}
@app.get("/api/satellite/images")
def images(category: str | None=None):
    data=inventory(); data=data[data.category==category] if category else data; return data.to_dict("records")
@app.post("/api/uploads")
async def upload_images(files: list[UploadFile] = File(...), image_type: str = Form(...), acquisition_date: str = Form(...)):
    try: date = datetime.strptime(acquisition_date, "%Y-%m-%d").date().isoformat()
    except ValueError: raise HTTPException(422, detail="Use acquisition date format YYYY-MM-DD.")
    tokens = {"Sentinel-1 VV":"VV", "Sentinel-1 VH":"VH", "Sentinel-2 NDVI":"NDVI", "Sentinel-2 True Color":"TrueColor"}
    if image_type not in tokens: raise HTTPException(422, detail="Choose a supported satellite image type.")
    destination = settings.dataset_path / "imported" / tokens[image_type]; destination.mkdir(parents=True, exist_ok=True)
    added=[]
    for index, file in enumerate(files, 1):
        suffix=Path(file.filename or "").suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".jp2"}: continue
        target=destination / f"Gorewada_{tokens[image_type]}_{date}_{index}{suffix}"; copy_index=2
        while target.exists(): target=destination / f"Gorewada_{tokens[image_type]}_{date}_{index}_{copy_index}{suffix}"; copy_index+=1
        with target.open("wb") as output: shutil.copyfileobj(file.file, output)
        added.append(target.name)
    if not added: raise HTTPException(422, detail="Upload JPG, PNG, TIFF, or JP2 image files.")
    return {"added":len(added), "date":date, "files":added}
@app.post("/api/satellite/process")
def process(): return {"metrics_csv":export_metrics(series()), "message":"Computed from available NDVI JPEG display imagery."}
@app.get("/api/ndvi/latest")
def ndvi_latest():
    data=series()
    if data.empty: raise HTTPException(404,detail="No dated NDVI scenes found")
    row=data.iloc[-1]; return {"date":row.date,"statistics":ndvi_statistics(row.path),"path":row.path}
@app.get("/api/ndvi/timeseries")
def ndvi_ts(): return json_records(series())
@app.get("/api/ndvi/dates")
def ndvi_dates():
    """Fast date list for UI selectors; avoids calculating every scene first."""
    records = inventory()
    dates = records.loc[records.category == "ndvi", "date"].dropna().drop_duplicates().sort_values().tolist()
    return {"dates": dates}
@app.get("/api/ndvi/statistics")
def ndvi_stats(): return ndvi_latest()
@app.get("/api/radar/latest")
def radar_latest(): return {"message":"No separately labeled VV/VH files detected. Sentinel-1 imagery is retained as unlabelled JPEG scenes; radar metrics are unavailable."}
@app.get("/api/radar/timeseries")
def radar_ts(): return []
@app.get("/api/multisatellite/status")
def multisatellite_status(): return {"available": multisatellite_available(), "project_data_path": str(settings.project_data_path)}
@app.post("/api/multisatellite/compare")
def multisatellite_compare():
    if not multisatellite_available(): raise HTTPException(404, detail="Supplied Copernicus project data is not available.")
    *_, summary = multisatellite_comparison()
    return summary
@app.get("/api/forest-health/latest")
def forest_latest():
    data=series()
    if data.empty: raise HTTPException(404,detail="No observations")
    return data.iloc[-1].to_dict()
@app.get("/api/forest-health/timeseries")
def forest_ts(): return json_records(series())
@app.post("/api/change-detection/run")
def change_run(request: ChangeRequest):
    try:
        result=change_detection(record_for_date(request.earlier_date).path,record_for_date(request.later_date).path); result.pop("difference_map"); return result
    except ValueError as exc: raise HTTPException(422,detail=str(exc))
@app.get("/api/change-detection")
def change_info(): return {"message":"Submit two available NDVI dates to /api/change-detection/run."}
@app.get("/api/gis/dashboard")
def gis_dashboard(earlier_date: str, later_date: str):
    """Date-selectable GIS summary for the React dashboard."""
    try:
        result = change_detection(record_for_date(earlier_date).path, record_for_date(later_date).path)
        result.pop("difference_map", None)
        return {"earlier_date": earlier_date, "later_date": later_date, "period": f"{earlier_date} to {later_date}", **result}
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc))
@app.get("/api/gis/true-colour/{date}")
def gis_true_colour(date: str):
    """Serve the actual dated true-colour scene for the GIS frontend."""
    row = record_for_date(date, "true_color")
    return FileResponse(row.path, media_type="image/jpeg", filename=row.filename)
@app.post("/api/predictions/train")
def prediction_train(request: PredictionRequest):
    try: _,metrics,_=train_ndvi_model(series(),request.model); return metrics
    except ValueError as exc: raise HTTPException(422,detail=str(exc))
@app.post("/api/predictions/generate")
def prediction_generate(request: PredictionRequest): return prediction_train(request)
@app.get("/api/predictions/forecast")
def predictions_forecast(horizon: int = 6):
    try:
        model, _, _ = train_ndvi_model(series(), "Random Forest", additional_trees=600)
        output = forecast_future_ndvi(model, series(), horizon)
        output["predicted_ndvi_percent"] = (output.predicted_ndvi * 100).round(2)
        return json_records(output)
    except ValueError as exc: raise HTTPException(422, detail=str(exc))
@app.get("/api/predictions")
def predictions(): return {"message":"Train a model before generating a forecast."}
@app.post("/api/models/cnn-yolo-compare")
def cnn_yolo_compare(epochs: int = 5):
    try:
        table, note = compare_cnn_yolo(epochs)
        return {"results": table.to_dict("records"), "note": note}
    except ValueError as exc: raise HTTPException(422, detail=str(exc))
@app.get("/api/alerts")
def alerts():
    db=SessionLocal(); items=db.query(Alert).all(); db.close(); return [{"alert_id":x.alert_id,"date":x.date,"severity":x.severity,"reason":x.reason,"ndvi_change":x.ndvi_change,"recommended_action":x.recommended_action,"status":x.status} for x in items]
@app.post("/api/alerts/{alert_date}/status")
def alert_status(alert_date: str, status: str): return {"message": update_alert_status(alert_date, status)}
@app.post("/api/alerts/check")
def alerts_check(authorization: str | None = Header(default=None)):
    active = check_health_alerts(series())
    token = authorization.removeprefix("Bearer ").strip() if authorization else None
    user = user_from_token(token)
    delivery = deliver_new_alerts(user) if user else {"sent": 0, "configured": False, "message": "Sign in to receive email alerts."}
    return {"threshold": 35, "active_alerts": int(len(active)), "message": "Alerts are created when vegetation health score is below 35.", "email_delivery": delivery}
@app.post("/api/reports/{period}")
def report(period: str):
    if period not in {"monthly","yearly"}: raise HTTPException(422,detail="period must be monthly or yearly")
    return {"html_report":generate_html_report(series(),alerts(),period),"metrics_csv":export_metrics(series())}

def report_content(period: str, report_date: str | None):
    if period not in {"monthly", "yearly"}:
        raise HTTPException(422, detail="Choose monthly or yearly.")
    content = series()
    if not report_date:
        return content, None
    dates = pd.to_datetime(content["date"], errors="coerce")
    pattern = r"^\d{4}-\d{2}$" if period == "monthly" else r"^\d{4}$"
    if not __import__("re").match(pattern, report_date):
        required = "YYYY-MM" if period == "monthly" else "YYYY"
        raise HTTPException(422, detail=f"Use {required} for the selected report.")
    selected = content.loc[dates.dt.strftime("%Y-%m" if period == "monthly" else "%Y") == report_date].copy()
    if selected.empty:
        raise HTTPException(404, detail=f"No NDVI observations are available for {report_date}.")
    return selected, report_date

@app.get("/api/reports/available-periods")
def report_available_periods():
    records = inventory().loc[lambda frame: (frame.category == "ndvi") & frame.date.notna(), "date"]
    values = pd.to_datetime(records, errors="coerce").dropna()
    return {"months": sorted(values.dt.strftime("%Y-%m").unique().tolist()),
            "years": sorted(values.dt.strftime("%Y").unique().tolist())}

@app.get("/api/reports/{period}/download")
def report_download(period: str, format: str = "html", report_date: str | None = None):
    content, report_id = report_content(period, report_date); settings.ensure_output_dirs()
    label = report_id or "all-observations"
    if format == "csv":
        path=Path(export_metrics(content, report_id)); return FileResponse(path, media_type="text/csv", filename=f"forestwatch_{period}_{label}_metrics.csv")
    if format == "html":
        path=Path(generate_html_report(content, alerts(), period, report_id)); return FileResponse(path, media_type="text/html", filename=f"forestwatch_{period}_{label}_report.html")
    raise HTTPException(422, detail="Format must be html or csv.")
@app.get("/api/reports")
def reports(): return {"reports_directory":str(settings.output_path / "reports")}
