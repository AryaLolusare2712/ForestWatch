from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_ndvi_model(series: pd.DataFrame, model_name="Random Forest", existing_model=None, additional_trees: int = 200):
    values=series.dropna(subset=["ndvi_mean"]).sort_values("date").reset_index(drop=True)
    if len(values) < 8: raise ValueError("At least 8 observations are required for an honest chronological evaluation.")
    values["lag1"]=values.ndvi_mean.shift(1); values["lag2"]=values.ndvi_mean.shift(2); values["month"]=pd.to_datetime(values.date).dt.month
    data=values.dropna(); X=data[["lag1","lag2","month"]]; y=data.ndvi_mean
    split=max(1, int(len(data)*.8)); train_x,test_x=X.iloc[:split],X.iloc[split:]; train_y,test_y=y.iloc[:split],y.iloc[split:]
    if model_name == "Random Forest" and isinstance(existing_model, RandomForestRegressor):
        model = existing_model
        model.warm_start = True
        model.n_estimators += int(additional_trees)
    else:
        model=RandomForestRegressor(n_estimators=int(additional_trees), random_state=42, warm_start=True) if model_name == "Random Forest" else LinearRegression()
    model.fit(train_x,train_y); pred=model.predict(test_x) if len(test_x) else np.array([])
    metrics={"model":model_name,"training_rows":len(train_x),"test_rows":len(test_x),"trees": model.n_estimators if model_name == "Random Forest" else None, "mae":float(mean_absolute_error(test_y,pred)) if len(pred) else None,"rmse":float(mean_squared_error(test_y,pred)**.5) if len(pred) else None,"r2":float(r2_score(test_y,pred)) if len(pred)>1 else None}
    return model,metrics,pd.DataFrame({"date":data.date.iloc[split:],"actual":test_y,"predicted":pred})

def forecast_future_ndvi(model, series: pd.DataFrame, horizon: int = 6) -> pd.DataFrame:
    """Generate a recursive future forecast after a model has been trained."""
    history = series.dropna(subset=["ndvi_mean"]).sort_values("date").copy()
    if len(history) < 2:
        raise ValueError("At least two valid NDVI observations are required to forecast.")
    horizon = int(horizon)
    if not 1 <= horizon <= 24:
        raise ValueError("Forecast horizon must be between 1 and 24 observations.")
    last_date = pd.to_datetime(history.iloc[-1].date)
    values = history.ndvi_mean.tolist()
    future = []
    for step in range(1, horizon + 1):
        date = last_date + pd.DateOffset(months=step)
        features = pd.DataFrame([[values[-1], values[-2], date.month]], columns=["lag1", "lag2", "month"])
        estimate = float(model.predict(features)[0])
        values.append(estimate)
        future.append({"forecast_date": date.date().isoformat(), "predicted_ndvi": estimate})
    return pd.DataFrame(future)
