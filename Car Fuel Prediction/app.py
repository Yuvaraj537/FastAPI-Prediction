from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import pickle
import pandas as pd

app = FastAPI()

# Load the ML model
with open("carprice_model.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Setup templates directory
templates = Jinja2Templates(directory="templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
def predict(
    request: Request,
    maker_year: int = Form(...),
    mileage_kmpl: float = Form(...),
    engine_cc: float = Form(...),
    fuel_type: str = Form(...),
    brand: str = Form(...),
    transmission: str = Form(...)
):
    # Create sample DataFrame with correct column names as used in training
    sample = pd.DataFrame({
        "make_year": [maker_year],
        "mileage_kmpl": [mileage_kmpl],
        "engine_cc": [engine_cc],
        "fuel_type": [fuel_type],
        "brand": [brand],
        "transmission": [transmission]
    })

    # Predict
    result = pipeline.predict(sample)[0]

    # Return result template
    return templates.TemplateResponse("result.html", {
        "request": request,
        "price_usd": round(result)
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)
