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
# 静的ファイル
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
    
@app.post("/predict")
def predict(req: PredictRequest):

    combined_district = req.市区町村 + "_" + req.地区

    data = pd.DataFrame([{
        "都道府県名": req.都道府県,
        "市区町村名": req.市区町村,
        "地区名": combined_district,
        "面積": req.面積,
        "築年数": req.築年数,
        "駅距離": req.駅距離,
        "道路幅": req.道路幅,

        "地区平均価格": 0,
        "市区町村平均価格": 0,

        "建ぺい率": 0,
        "容積率": 0,
        "用途": ""
    }])

    pred = model.predict(data)[0]

    # マイナスは 0 に補正
    pred = max(pred, 0)

    return {"predicted_price": int(pred)}

# ============================
# ルート
# ============================

@app.get("/")
def root():
    return {"message": "Real Estate AI Running"}
