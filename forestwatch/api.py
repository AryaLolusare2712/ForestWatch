from fastapi import FastAPI, HTTPException
from .config import settings
from .dataset_scanner import scan_dataset, inventory_summary
from .analysis import ndvi_timeseries, health_scores, ndvi_statistics, change_detection
from .prediction import train_ndvi_model
from .schemas import ChangeRequest, PredictionRequest
from .database import initialise_database, SessionLocal, Alert
from .reports import export_metrics, generate_html_report
from .multisatellite import available as multisatellite_available, comparison as multisatellite_comparison

app=FastAPI(title="ForestWatch API", version="1.0.0")
initialise_database()

def inventory(): return scan_dataset(settings.dataset_path)
def series(): return health_scores(ndvi_timeseries(inventory()))
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
@app.get("/api/aoi")
def aoi(): return {"name":settings.aoi_name}
@app.get("/api/dataset/inventory")
def dataset_inventory():
    data=inventory(); return {"summary":inventory_summary(data),"files":data.to_dict("records")}
@app.get("/api/satellite/images")
def images(category: str | None=None):
    data=inventory(); data=data[data.category==category] if category else data; return data.to_dict("records")
@app.post("/api/satellite/process")
def process(): return {"metrics_csv":export_metrics(series()), "message":"Computed from available NDVI JPEG display imagery."}
@app.get("/api/ndvi/latest")
def ndvi_latest():
    data=series()
    if data.empty: raise HTTPException(404,detail="No dated NDVI scenes found")
    row=data.iloc[-1]; return {"date":row.date,"statistics":ndvi_statistics(row.path),"path":row.path}
@app.get("/api/ndvi/timeseries")
def ndvi_ts(): return series().to_dict("records")
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
def forest_ts(): return series().to_dict("records")
@app.post("/api/change-detection/run")
def change_run(request: ChangeRequest):
    try:
        result=change_detection(record_for_date(request.earlier_date).path,record_for_date(request.later_date).path); result.pop("difference_map"); return result
    except ValueError as exc: raise HTTPException(422,detail=str(exc))
@app.get("/api/change-detection")
def change_info(): return {"message":"Submit two available NDVI dates to /api/change-detection/run."}
@app.post("/api/predictions/train")
def prediction_train(request: PredictionRequest):
    try: _,metrics,_=train_ndvi_model(series(),request.model); return metrics
    except ValueError as exc: raise HTTPException(422,detail=str(exc))
@app.post("/api/predictions/generate")
def prediction_generate(request: PredictionRequest): return prediction_train(request)
@app.get("/api/predictions")
def predictions(): return {"message":"Train a model before generating a forecast."}
@app.get("/api/alerts")
def alerts():
    db=SessionLocal(); items=db.query(Alert).all(); db.close(); return [{"alert_id":x.alert_id,"date":x.date,"severity":x.severity,"reason":x.reason,"ndvi_change":x.ndvi_change,"recommended_action":x.recommended_action,"status":x.status} for x in items]
@app.post("/api/alerts/check")
def alerts_check(): return {"message":"Alert generation requires georeferenced numeric observations; no alert was fabricated from JPEG visualizations."}
@app.post("/api/reports/{period}")
def report(period: str):
    if period not in {"weekly","monthly"}: raise HTTPException(422,detail="period must be weekly or monthly")
    return {"html_report":generate_html_report(series(),alerts(),period),"metrics_csv":export_metrics(series())}
@app.get("/api/reports")
def reports(): return {"reports_directory":str(settings.output_path / "reports")}
