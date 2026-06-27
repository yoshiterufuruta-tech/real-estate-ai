from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
import json
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

model = joblib.load(BASE_DIR / "model.pkl")

preprocess = model.named_steps["preprocess"]
regressor = model.named_steps["regressor"]

with open(STATIC_DIR / "city_avg_price.json", encoding="utf-8") as f:
    city_avg_price = json.load(f)

with open(STATIC_DIR / "district_avg_price.json", encoding="utf-8") as f:
    district_avg_price = json.load(f)

feature_columns = preprocess.get_feature_names_out()

class PredictRequest(BaseModel):
    市区町村名: str
    地区名: str
    面積: float
    延床面積: float
    築年数: float
    駅距離: float
    道路幅: float
    道路種類: str
    建ぺい率: float
    容積率: float
    用途: str
    構造: str
    
@app.post("/predict")
def predict(req: PredictRequest):

    city_avg = city_avg_price.get(req.市区町村名, 0)
    district_avg = district_avg_price.get(req.地区名, 0)

    raw = pd.DataFrame([{
        "都道府県名": "東京都",  # ← 学習データが東京都のみなので固定
        "市区町村名": req.市区町村名,
        "地区名": req.地区名,
        "面積": req.面積,
        "延床面積": req.延床面積,
        "築年数": req.築年数,
        "駅距離": req.駅距離,
        "道路幅": req.道路幅,
        "道路種類": req.道路種類,
        "建ぺい率": req.建ぺい率,
        "容積率": req.容積率,
        "用途": req.用途,
        "構造": req.構造,
        "市区町村平均価格": city_avg,
        "地区平均価格": district_avg,
        "市区町村平均価格_log": np.log1p(city_avg),
        "地区平均価格_log": np.log1p(district_avg)
    }])
    
    raw["駅距離_log"] = np.log1p(raw["駅距離"])
    raw["面積_sqrt"] = np.sqrt(raw["面積"])
    raw["延床面積_sqrt"] = np.sqrt(raw["延床面積"]) 
    
    X = preprocess.transform(raw)

    pred = regressor.predict(X)[0]
    pred = pred * (122.1 / 119.2)
    pred_list_price = pred * 1.25S
    pred = max(pred, 0)

    return {
        "predicted_price": int(pred),
        "predicted_list_price": int(pred_list_price)
    }
