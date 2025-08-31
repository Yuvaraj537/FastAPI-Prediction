from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objs as go
from plotly.offline import plot as plotly_plot
from pathlib import Path
import pickle
import io
import mysql.connector
from mysql.connector import Error
# uvicorn main:app --reload

app = FastAPI(title="Prophet Bitcoin Forecaster")

# Folders
BASE_DIR = Path(__file__).parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
DOWNLOADS_DIR = STATIC_DIR

# Mount static and templates
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# MySQL config
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "bitcoin_forecast"
}

# ---------------- Utilities ----------------

def load_csv_bytes_to_df(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    # Flexible column names
    date_col = next((c for c in df.columns if c.lower() in ["date","ds"]), None)
    close_col = next((c for c in df.columns if c.lower() in ["close","y"]), None)
    if not date_col or not close_col:
        raise ValueError("CSV must have 'Date' and 'Close' columns")
    data = df[[date_col, close_col]].copy()
    data = data.rename(columns={date_col:"ds", close_col:"y"})
    data["ds"] = pd.to_datetime(data["ds"], errors="coerce")
    data["y"] = pd.to_numeric(data["y"], errors="coerce")
    data = data.dropna(subset=["ds","y"]).sort_values("ds").reset_index(drop=True)
    data["y_orig"] = data["y"]
    data["y"] = np.log(data["y"])
    return data

def train_and_forecast(data: pd.DataFrame, periods: int=12, freq: str="M"):
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    # Inverse log transform
    forecast["yhat"] = np.exp(forecast["yhat"])
    forecast["yhat_lower"] = np.exp(forecast["yhat_lower"])
    forecast["yhat_upper"] = np.exp(forecast["yhat_upper"])
    actual = data[["ds"]].copy()
    actual["y"] = np.exp(data["y"])
    merged = pd.merge(forecast, actual, on="ds", how="left")
    # Safety clip
    max_clip = merged["yhat"].max()*2
    merged["yhat_upper"] = merged["yhat_upper"].clip(upper=max_clip)
    merged["yhat_lower"] = merged["yhat_lower"].clip(lower=0)
    eval_df = merged.dropna(subset=["y","yhat"])
    rmse = float(np.sqrt(mean_squared_error(eval_df["y"], eval_df["yhat"]))) if not eval_df.empty else None
    mae = float(mean_absolute_error(eval_df["y"], eval_df["yhat"])) if not eval_df.empty else None
    return model, merged, rmse, mae

def make_plot(merged: pd.DataFrame) -> str:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged["ds"], y=merged["y"], mode="lines+markers", name="Actual"))
    fig.add_trace(go.Scatter(x=merged["ds"], y=merged["yhat"], mode="lines+markers", name="Predicted"))
    fig.add_trace(go.Scatter(x=merged["ds"], y=merged["yhat_upper"], mode="lines", name="Predicted Upper"))
    fig.add_trace(go.Scatter(x=merged["ds"], y=merged["yhat_lower"], mode="lines", name="Predicted Lower"))
    fig.update_layout(title="Actual vs Predicted", xaxis_title="Date", yaxis_title="Price", template="plotly_white", hovermode="x unified")
    return plotly_plot(fig, include_plotlyjs="cdn", output_type="div")

def save_to_mysql(merged: pd.DataFrame):
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS forecast_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                ds DATE NOT NULL,
                y DOUBLE,
                yhat DOUBLE NOT NULL,
                yhat_lower DOUBLE NOT NULL,
                yhat_upper DOUBLE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Insert rows
        for _, row in merged.iterrows():
            ds_val = pd.to_datetime(row["ds"]).date()
            y_val = row["y"] if not pd.isna(row["y"]) else None
            cursor.execute("""
                INSERT INTO forecast_data (ds, y, yhat, yhat_lower, yhat_upper)
                VALUES (%s,%s,%s,%s,%s)
            """, (ds_val, y_val, row["yhat"], row["yhat_lower"], row["yhat_upper"]))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Inserted {len(merged)} rows successfully!")
    except Error as e:
        print("MySQL Error:", e)
        raise e

# ---------------- Pydantic model ----------------

class ForecastRequest(BaseModel):
    periods: int = 12
    freq: str = "M"
    csv_text: Optional[str] = None

# ---------------- Routes ----------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "chart_div": None})

@app.post("/train-forecast", response_class=HTMLResponse)
async def train_forecast(
    request: Request,
    file: UploadFile = File(None),
    periods: int = Form(12),
    freq: str = Form("M")
):
    if file and file.filename:
        df = load_csv_bytes_to_df(await file.read())
    else:
        fallback = BASE_DIR / "coin_Bitcoin.xls"
        if not fallback.exists():
            return templates.TemplateResponse("index.html", {"request": request, "error": "CSV missing", "chart_div": None}, status_code=400)
        df = pd.read_csv(fallback)
    try:
        data = prepare_data(df)
        model, merged, rmse, mae = train_and_forecast(data, periods=periods, freq=freq)
        save_to_mysql(merged)  # <-- save to MySQL
        with open(BASE_DIR / "prophet_bitcoin_model.pkl", "wb") as f:
            pickle.dump(model, f)
        chart_div = make_plot(merged)
        head_rows = merged.head(5).to_dict(orient="records")
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "chart_div": chart_div, "rmse": f"{rmse:,.2f}" if rmse else None,
             "mae": f"{mae:,.2f}" if mae else None, "download_href": "/static/cleaned_forecast.csv",
             "head_rows": head_rows, "periods": periods, "freq": freq}
        )
    except Exception as e:
        return templates.TemplateResponse({"request": request, "error": str(e), "chart_div": None}, status_code=400)

@app.post("/api/forecast", response_model=dict)
async def api_forecast(req: ForecastRequest):
    if req.csv_text:
        df = pd.read_csv(io.StringIO(req.csv_text))
    else:
        fallback = BASE_DIR / "coin_Bitcoin.xls"
        if not fallback.exists():
            return JSONResponse({"error":"CSV not found"}, status_code=400)
        df = pd.read_csv(fallback)
    try:
        data = prepare_data(df)
        model, merged, rmse, mae = train_and_forecast(data, periods=req.periods, freq=req.freq)
        save_to_mysql(merged)  # <-- save to MySQL
        recent_actuals = merged[["ds","y"]].dropna().tail(5)
        future_predictions = merged[["ds","yhat","yhat_lower","yhat_upper"]].tail(req.periods)
        recent_actuals["ds"] = recent_actuals["ds"].astype(str)
        future_predictions["ds"] = future_predictions["ds"].astype(str)
        return {"recent_actuals": recent_actuals.to_dict(orient="records"),
                "future_predictions": future_predictions.to_dict(orient="records")}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
