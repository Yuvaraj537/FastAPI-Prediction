from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load trained LSTM model
model = load_model("tesla_lstm_model.h5")
scaler = MinMaxScaler(feature_range=(0,1))

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "prediction": None})

@app.post("/predict")
def predict(request: Request, prices: str = Form(...)):
    try:
        price_list = [float(x.strip()) for x in prices.split(",")]
        if len(price_list) != 60:
            return templates.TemplateResponse(
                "index.html",
                {"request": request, "prediction": None, "error": "Enter exactly 60 prices, separated by commas."}
            )

        # Prepare data for LSTM
        data = np.array(price_list).reshape(-1,1)
        scaled_data = scaler.fit_transform(data)  # for demo; ideally save/load scaler
        X_input = scaled_data.reshape(1, 60, 1)

        # Predict next day
        pred_scaled = model.predict(X_input)
        pred_price = scaler.inverse_transform(pred_scaled)[0,0]

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "prediction": f"${round(pred_price,2)}", "error": None}
        )
    except Exception as e:
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "prediction": None, "error": f"Error: {e}"}
        )
