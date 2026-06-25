from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# ============================
# モデル読み込み
# ============================

model = joblib.load("model.pkl")

# ============================
# 静的ファイル（index.html, JSON）を配信
# ============================

app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================
# 入力データモデル
# ============================

class PredictRequest(BaseModel):
    都道府県: str
    市区町村: str
    地区: str
    面積: float
    築年数: float
    駅距離: float
    道路幅: float

# ============================
# /predict（完全版）
# ============================

@app.post("/predict")
def predict(req: PredictRequest):

    data = pd.DataFrame([{
        "都道府県名": req.都道府県,
        "市区町村名": req.市区町村,
        "地区名": req.地区,
        "面積": req.面積,
        "築年数": req.築年数,
        "駅距離": req.駅距離,
        "道路幅": req.道路幅,
        "建ぺい率": 0, 
        "容積率": 0,
        "用途": ""  
    }])

    # 予測
    pred = model.predict(data)[0]

    return {
        "predicted_price": int(pred),
        "used": data.to_dict(orient="records")[0]
    }

# ============================
# ルート（index.html を返す）
# ============================

@app.get("/")
def root():
    return {"message": "不動産価格AI API 動作中。/static/index.html を開いてください。"}
