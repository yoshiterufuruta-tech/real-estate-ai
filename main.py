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

# ============================
# /predict（完全版）
# ============================

@app.post("/predict")
def predict(req: PredictRequest):

    # ★ 学習時と同じ「市区町村_地区」形式に変換
    combined_district = req.市区町村 + "_" + req.地区

    # ★ 学習データを読み込み（平均価格を計算するため）
    df = pd.read_csv("Tokyo_20244_20254.csv", encoding="cp932")
    df["地区名"] = df["市区町村名"] + "_" + df["地区名"].fillna("")

    # ============================
    # 地区平均価格
    # ============================
    district_mean = df[df["地区名"] == combined_district]["取引価格（総額）"].mean()

    # データが無い地区 → 市区町村平均で補完
    if pd.isna(district_mean):
        district_mean = df[df["市区町村名"] == req.市区町村]["取引価格（総額）"].mean()

    # それでも無い場合 → 全体平均
    if pd.isna(district_mean):
        district_mean = df["取引価格（総額）"].mean()

    # ============================
    # 市区町村平均価格
    # ============================
    city_mean = df[df["市区町村名"] == req.市区町村]["取引価格（総額）"].mean()

    if pd.isna(city_mean):
        city_mean = df["取引価格（総額）"].mean()

    # ============================
    # 推論用 DataFrame
    # ============================

    data = pd.DataFrame([{
        "都道府県名": req.都道府県,
        "市区町村名": req.市区町村,
        "地区名": combined_district,
        "面積": req.面積,
        "築年数": req.築年数,
        "駅距離": req.駅距離,
        "道路幅": req.道路幅,

        # ★ 学習時と同じ特徴量を渡す（ここが最重要）
        "地区平均価格": district_mean,
        "市区町村平均価格": city_mean,

        "建ぺい率": 0,
        "容積率": 0,
        "用途": ""
    }])

    # ============================
    # 予測
    # ============================

    pred = model.predict(data)[0]

    # マイナスは 0 に補正
    pred = max(pred, 0)

    return {
        "predicted_price": int(pred)
    }

# ============================
# ルート
# ============================

@app.get("/")
def root():
    return {"message": "Real Estate AI Running"}
