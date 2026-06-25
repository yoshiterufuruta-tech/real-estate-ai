from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
import json
import pandas as pd
from pathlib import Path

# ============================
# パス設定
# ============================

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# ============================
# FastAPI アプリ本体
# ============================

app = FastAPI()

# static フォルダを公開
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ============================
# モデル読み込み
# ============================

model = joblib.load(BASE_DIR / "model.pkl")

# 前処理部分と LightGBM 部分を分離
preprocess = model.named_steps["preprocess"]
regressor = model.named_steps["regressor"]

# ============================
# JSON 読み込み（train_model.py で生成）
# ============================

with open(STATIC_DIR / "city_avg_price.json", encoding="utf-8") as f:
    city_avg_price = json.load(f)

with open(STATIC_DIR / "district_avg_price.json", encoding="utf-8") as f:
    district_avg_price = json.load(f)

# ColumnTransformer の出力列名を取得（これが正しい列順）
feature_columns = preprocess.get_feature_names_out()

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

@app.post("/predict")
def predict(req: PredictRequest):

    # train_model.py と同じ特徴量を生成
    city_avg = city_avg_price.get(req.市区町村名, 0)
    district_avg = district_avg_price.get(req.地区名, 0)

    raw = pd.DataFrame([{
        "都道府県名": "東京都",  # ← 学習データが東京都のみなので固定
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
        "地区平均価格_log": np.log1p(district_avg)
    }])
    
    # ColumnTransformer に通す
    X = preprocess.transform(raw)

    # 推定
    pred = model.predict(X)[0]

    # マイナス予測は 0 に補正（安全策）
    pred = max(pred, 0)

    return {"predicted_price": int(pred)}
