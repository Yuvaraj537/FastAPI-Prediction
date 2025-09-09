from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pickle
import pandas as pd
import mysql.connector

# ------------------- FastAPI Setup ------------------- #
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ------------------- Load Model ------------------- #
with open("carsales_model.pkl", "rb") as f:
    pipeline = pickle.load(f)

# ------------------- MySQL Connection ------------------- #
db = mysql.connector.connect(
    host="localhost",
    user="root",          # change to your MySQL username
    password="root",  # change to your MySQL password
    database="car_prediction"
)
cursor = db.cursor()

# ------------------- Routes ------------------- #
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    maker_year: int = Form(...),
    mileage_kmpl: float = Form(...),
    engine_cc: float = Form(...),
    fuel_type: str = Form(...),
    brand: str = Form(...),
    transmission: str = Form(...),
    service_history: str = Form(...),
    accidents_reported: int = Form(...),
    insurance_valid: str = Form(...),
    buyer_type: str = Form(...)
):
    # ✅ Match training column names
    sample = pd.DataFrame({
        "Make Year": [maker_year],
        "Mileage (kmpl)": [mileage_kmpl],
        "Engine CC": [engine_cc],
        "Fuel Type": [fuel_type],
        "Brand": [brand],
        "Transmission": [transmission],
        "Service History": [service_history],
        "Accidents Reports": [accidents_reported],   # fixed name
        "Insurance Valid": [insurance_valid],
        "Buyer Type": [buyer_type]
    })

    try:
        predicted_price = pipeline.predict(sample)[0]
        result = round(float(predicted_price), 2)
        error = None

        # ------------------- Save to MySQL ------------------- #
        query = """
            INSERT INTO predictions (
                make_year, mileage_kmpl, engine_cc, fuel_type, brand,
                transmission, service_history, accidents_reported,
                insurance_valid, buyer_type, predicted_price_usd
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            maker_year, mileage_kmpl, engine_cc, fuel_type, brand,
            transmission, service_history, accidents_reported,
            insurance_valid, buyer_type, result
        )
        cursor.execute(query, values)
        db.commit()

    except Exception as e:
        result = None
        error = str(e)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "price_usd": result,
        "error": error
    })

# ------------------- Run Server ------------------- #
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
