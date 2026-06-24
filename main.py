from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import traceback
import os
import joblib
import re
import numpy as np
from sklearn.impute import SimpleImputer
import httpx

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# モデル読み込み
MODEL_PATH = "model.pkl"
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("Loaded model from", MODEL_PATH)
    except Exception as e:
        print("Failed to load model:", e)
        model = None
else:
    print("No model file found at", MODEL_PATH, "- using dummy predictor")

MLIT_KEY = os.getenv("MLIT_API_KEY")

# -------------------------
# Utility functions
# -------------------------

def clean_number(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def normalize_payload(data: dict):
    prefecture = data.get("都道府県") or data.get("prefecture") or ""
    city = data.get("市区町村") or data.get("city") or ""
    district = data.get("地区") or data.get("district") or ""

    return {
        "年度": data.get("年度", ""),
        "面積": data.get("面積", 0),
        "築年数": data.get("築年数", data.get("年数", 0)),
        "駅距離": data.get("駅距離", 0),
        "道路幅": data.get("道路幅", 0),
        "都道府県": prefecture,
        "市区町村": city,
        "市区町村名": city,
        "地区": district,
        "地域": data.get("地域", ""),
        "容積率": data.get("容積率", 0),
        "建ぺい率": data.get("建ぺい率", 0),
    }

def convert_year_quarter(x):
    if not x or not isinstance(x, str):
        return {"年度_raw": x, "year": 0, "quarter": 0}
    m = re.search(r"(\d{4}).*第\s*?(\d)\s*四半期", x)
    if m:
        return {"年度_raw": x, "year": int(m.group(1)), "quarter": int(m.group(2))}
    m2 = re.search(r"(\d{4})", x)
    if m2:
        return {"年度_raw": x, "year": int(m2.group(1)), "quarter": 0}
    return {"年度_raw": x, "year": 0, "quarter": 0}

def numeric_impute(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        imputer = SimpleImputer(strategy="mean")
        df[num_cols] = imputer.fit_transform(df[num_cols])
    return df

# -------------------------
# Dummy predictor
# -------------------------

def dummy_predict(df: pd.DataFrame):
    base_unit = 200000
    factor_map = {
        "第一種低層住居専用地域": 1.2,
        "第一種住居地域": 1.0,
        "商業地域": 1.8,
        "住宅地": 0.9
    }
    results = []
    for _, row in df.iterrows():
        youto = row.get("用途", "住宅地")
        factor = factor_map.get(youto, 1.0)
        area = float(row.get("面積", 0) or 0)
        price = area * base_unit * factor
        results.append(price)
    return results

# -------------------------
# Prediction logic
# -------------------------

def predict_logic(data: dict):
    try:
        payload = normalize_payload(data)

        yq = convert_year_quarter(payload["年度"])
        payload["年度_raw"] = yq["年度_raw"]
        payload["年度_year"] = yq["year"]
        payload["年度_quarter"] = yq["quarter"]

        city = payload["市区町村"]
        district = payload["地区"] or ""
        district_full = f"{city}_{district}" if district else city
        payload["地区_full"] = district_full

        model_input = {
            "年度": payload["年度_raw"],
            "年度_raw": payload["年度_raw"],
            "年度_year": payload["年度_year"],
            "年度_quarter": payload["年度_quarter"],
            "面積": payload["面積"],
            "築年数": payload["築年数"],
            "駅距離": payload["駅距離"],
            "道路幅": payload["道路幅"],
            "都道府県": payload["都道府県"],
            "市区町村": payload["市区町村"],
            "市区町村名": payload["市区町村名"],
            "地区": payload["地区_full"],
            "地域": payload.get("地域", "")
        }

        df = pd.DataFrame([model_input])

        df["面積"] = df["面積"].apply(clean_number)
        df["築年数"] = df["築年数"].apply(clean_number)
        df["駅距離"] = df["駅距離"].apply(clean_number)
        df["道路幅"] = df["道路幅"].apply(clean_number)

        if "年度" in df.columns:
            df = df.drop(columns=["年度"])

        df = numeric_impute(df)

        prefecture = payload["都道府県"]
        if prefecture == "東京都":
            youto = "第一種住居地域"
        else:
            youto = "住宅地"

        df["用途"] = youto

        if model is not None:
            pred = model.predict(df)[0]
        else:
            pred = dummy_predict(df)[0]

        return {"predicted_price": pred, "used": payload}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={
            "error": "internal server error",
            "message": str(e),
            "received": data
        })

# -------------------------
# /predict endpoint
# -------------------------

@app.post("/predict")
async def predict_endpoint(request: Request):
    payload = await request.json()
    try:
        result = predict_logic(payload)
        return result
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": "internal server error", "message": str(e)})

# -------------------------
# MLIT API 統合
# -------------------------

@app.get("/mlit/trade")
async def mlit_trade(pref: str, city: str, district: str = "", from_year: int = 20201, to_year: int = 20244):
    if MLIT_KEY is None:
        return {"error": "MLIT_API_KEY is not set in Render environment variables"}

    base_url = "https://www.land.mlit.go.jp/webland/api/TradeList"
    params = {
        "from": from_year,
        "to": to_year,
        "area": pref,
        "city": city
    }
    if district:
        params["district"] = district

    headers = {"Ocp-Apim-Subscription-Key": MLIT_KEY}

    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(base_url, params=params, headers=headers)
        return r.json()
