from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
import json
import pandas as pd

# ============================
# FastAPI アプリ本体
# ============================

app = FastAPI()

# static フォルダを公開
app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================
# モデル読み込み
# ============================

model = joblib.load("model.pkl")

# ============================
# JSON 読み込み（train_model.py で生成）
# ============================

with open("static/city_avg_price.json", encoding="utf-8") as f:
    city_avg_price = json.load(f)

with open("static/district_avg_price.json", encoding="utf-8") as f:
    district_avg_price = json.load(f)

# ============================
# 入力データ形式
# ============================

class PredictRequest(BaseModel):
    市区町村名: str
    地区名: str
    面積: float
    築年数: float
    駅距離: float
    道路幅: float
    建ぺい率: float
    容積率: float
    用途: str

# ============================
# 推定 API
# ============================

@app.post("/predict")
def predict(req: PredictRequest):

    # 平均価格特徴量（train_model.py と揃える）
    city_avg = city_avg_price.get(req.市区町村名, 0)
    district_avg = district_avg_price.get(req.地区名, 0)

    data = {
        # ★ モデルが要求している列をすべて揃える ★
        "都道府県名": "東京都",  
        "市区町村名": req.市区町村名,
        "地区名": req.地区名,
        "面積": req.面積,
        "築年数": req.築年数,
        "駅距離": req.駅距離,
        "道路幅": req.道路幅,
        "建ぺい率": req.建ぺい率,
        "容積率": req.容積率,
        "用途": req.用途,

        "市区町村平均価格": city_avg,
        "地区平均価格": district_avg,
        "市区町村平均価格_log": np.log1p(city_avg),
        "地区平均価格_log": np.log1p(district_avg),
    }

    df = pd.DataFrame([data])
    pred = model.predict(df)[0]

    return {"predicted_price": int(pred)}
