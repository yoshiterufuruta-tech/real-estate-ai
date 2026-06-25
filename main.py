from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
app.mount("/static", StaticFiles(directory="static"), name="static")

import joblib
import numpy as np
import json

app = FastAPI()

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

with open("static/city_to_districts.json", encoding="utf-8") as f:
    city_to_districts = json.load(f)

# ============================
# 入力データ形式
# ============================

class PredictRequest(BaseModel):
    市区町村名: str
    地区名: str
    面積: float
    駅距離: float
    道路幅: float
    建ぺい率: float
    容積率: float
    用途: str
    築年数: float

# ============================
# 推定 API
# ============================

@app.post("/predict")
def predict(req: PredictRequest):

    # ----------------------------
    # 1. 入力データを DataFrame 形式に
    # ----------------------------
    data = {
        "市区町村名": req.市区町村名,
        "地区名": req.地区名,
        "面積": req.面積,
        "駅距離": req.駅距離,
        "道路幅": req.道路幅,
        "建ぺい率": req.建ぺい率,
        "容積率": req.容積率,
        "用途": req.用途,
        "築年数": req.築年数,
    }

    # ----------------------------
    # 2. train_model.py と同じ特徴量を生成
    # ----------------------------

    # 市区町村平均価格
    city_avg = city_avg_price.get(req.市区町村名, 0)
    data["市区町村平均価格"] = city_avg
    data["市区町村平均価格_log"] = np.log1p(city_avg)

    # 地区平均価格
    district_avg = district_avg_price.get(req.地区名, 0)
    data["地区平均価格"] = district_avg
    data["地区平均価格_log"] = np.log1p(district_avg)

    # ----------------------------
    # 3. モデル推定
    # ----------------------------
    import pandas as pd
    df = pd.DataFrame([data])

    pred = model.predict(df)[0]

    return {"predicted_price": int(pred)}
