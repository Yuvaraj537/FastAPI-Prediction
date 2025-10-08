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
with open("car sales price_model.pkl", "rb") as f:
    pipeline = pickle.load(f)

# ------------------- MySQL Connection ------------------- #
db = mysql.connector.connect(
    host="localhost",
    user="root",          
    password="root",  
    database="car_prediction"     # ✅ Make sure this DB exists
)
cursor = db.cursor()

# ------------------- Routes ------------------- #
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    make_year: int = Form(...),
    mileage_kmpl: float = Form(...),
    engine_cc: float = Form(...),
    fuel_type: str = Form(...),
    owner_count: int = Form(...),
    brand: str = Form(...),
    transmission: str = Form(...),
    color: str = Form(...),
    service_history: str = Form(...),
    accidents_reported: int = Form(...),
    insurance_valid: str = Form(...)
):
    # ✅ Prepare sample DataFrame with exact column names
    sample = pd.DataFrame({
        "make_year": [make_year],
        "mileage_kmpl": [mileage_kmpl],
        "engine_cc": [engine_cc],
        "fuel_type": [fuel_type],
        "owner_count": [owner_count],
        "brand": [brand],
        "transmission": [transmission],
        "color": [color],
        "service_history": [service_history],
        "accidents_reported": [accidents_reported],
        "insurance_valid": [insurance_valid],
    })

    try:
        # ✅ Prediction
        predicted_price = pipeline.predict(sample)[0]
        result = round(float(predicted_price), 2)
        error = None

        # ✅ Ensure table exists
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            make_year INT,
            mileage_kmpl FLOAT,
            engine_cc FLOAT,
            fuel_type VARCHAR(50),
            owner_count INT,
            brand VARCHAR(100),
            transmission VARCHAR(50),
            color VARCHAR(50),
            service_history VARCHAR(50),
            accidents_reported INT,
            insurance_valid VARCHAR(10),
            predicted_price_usd FLOAT
        )
        """)
        db.commit()

        # ✅ Save to MySQL
        query = """
            INSERT INTO predictions (
                make_year, mileage_kmpl, engine_cc, fuel_type, owner_count, brand,
                transmission, color, service_history, accidents_reported,
                insurance_valid, predicted_price_usd
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (
            make_year, mileage_kmpl, engine_cc, fuel_type, owner_count, brand,
            transmission, color, service_history, accidents_reported,
            insurance_valid, result
        )
        cursor.execute(query, values)
        db.commit()

    except Exception as e:
        result = None
        error = str(e)

    return templates.TemplateResponse("result.html", {
        "request": request,
        "price_inr": result,
        "error": error
    })


# ------------------- Run Server ------------------- #
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
