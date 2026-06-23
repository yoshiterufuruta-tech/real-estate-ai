# main.py
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import traceback
import os
import joblib

app = FastAPI()

# CORS（開発用。必要に応じて制限）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# static 配下に index.html を置く想定
app.mount("/static", StaticFiles(directory="static"), name="static")

# モデル読み込み（model.pkl があれば読み込む）
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

# --- ユーティリティ（必要に応じて実装を学習時に合わせてください） ---
def convert_year_quarter(x):
    return x

def clean_number(x):
    try:
        return float(x)
    except Exception:
        return 0.0

# --- 用途地域マップ（例。必要に応じて拡張） ---
tokyo_default = {
    "世田谷区": ("第一種低層住居専用地域", 50, 100),
    "渋谷区": ("商業地域", 80, 400),
    "港区": ("商業地域", 80, 600),
    "新宿区": ("商業地域", 80, 600),
    "杉並区": ("第一種低層住居専用地域", 50, 100),
    "練馬区": ("第一種低層住居専用地域", 50, 100),
    "足立区": ("第一種住居地域", 60, 200),
    "江戸川区": ("第一種住居地域", 60, 200),
    "八王子市": ("第一種低層住居専用地域", 40, 80),
    "町田市": ("第一種低層住居専用地域", 50, 100),
    "立川市": ("商業地域", 80, 400),
    "武蔵野市": ("第一種中高層住居専用地域", 60, 200),
    "三鷹市": ("第一種中高層住居専用地域", 60, 200),
    "調布市": ("第一種住居地域", 60, 200),
    "府中市": ("第一種住居地域", 60, 200),
}

setagaya_map = {
    "三宿": ("第一種低層住居専用地域", 50, 100),
    "三軒茶屋": ("商業地域", 80, 400),
    "上北沢": ("第一種低層住居専用地域", 50, 100),
    "上馬": ("第一種中高層住居専用地域", 60, 200),
    "北沢": ("第一種住居地域", 60, 200),
}

shibuya_map = {
    "神宮前": ("第一種住居地域", 60, 200),
    "代々木": ("商業地域", 80, 400),
    "恵比寿": ("商業地域", 80, 400),
    "広尾": ("第一種中高層住居専用地域", 60, 200),
}

# --- ダミー予測（モデルが無い場合のフォールバック） ---
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

# --- 予測ロジック（堅牢化済み） ---
def predict_logic(data: dict):
    try:
        # 受信キーの互換対応
        prefecture = data.get("都道府県") or data.get("都道府県名") or data.get("prefecture") or ""
        city = data.get("市区町村") or data.get("市区町村名") or data.get("市区町村名") or data.get("city") or ""
        # 地区は複数キーをチェック（地域, 地区名, district など）
        district = data.get("地区") or data.get("地区名") or data.get("地域") or data.get("district") or ""

        # 空値対策
        prefecture = prefecture or ""
        city = city or ""
        district = district or ""

        # 地名衝突対策：市区町村を前置して一意化
        district_full = f"{city}_{district}" if district else city

        # モデルが期待するカラムを明示的に作成（学習時のカラム名に合わせて必要なら追加）
        model_input = {
            "年度": data.get("年度", "2024年第1四半期"),
            "面積": data.get("面積", 0),
            "築年数": data.get("築年数", data.get("年数", 0)),
            "駅距離": data.get("駅距離", 0),
            "道路幅": data.get("道路幅", 0),
            "都道府県": prefecture,
            "市区町村": city,
            # 学習時に '市区町村名' や '地域' を参照しているモデルがある場合に備えて両方用意
            "市区町村名": city,
            "地区": district_full,
            "地域": data.get("地域") or data.get("地域名") or ""
        }

        df = pd.DataFrame([model_input])

        # 前処理
        df["年度"] = df["年度"].apply(convert_year_quarter)
        df["面積"] = df["面積"].apply(clean_number)
        df["駅距離"] = df["駅距離"].apply(clean_number)
        df["道路幅"] = df["道路幅"].apply(clean_number)

        # 用途地域の決定（東京都向け）
        if prefecture == "東京都":
            if city == "世田谷区" and (district in setagaya_map):
                youto, kenpei, youseki = setagaya_map[district]
            elif city == "渋谷区" and (district in shibuya_map):
                youto, kenpei, youseki = shibuya_map[district]
            elif city in tokyo_default:
                youto, kenpei, youseki = tokyo_default[city]
            else:
                youto, kenpei, youseki = ("第一種住居地域", 60, 200)
        else:
            youto, kenpei, youseki = ("住宅地", 60, 200)

        # 欠けている用途関連を埋める
        df["用途"] = youto
        df["建ぺい率"] = kenpei
        df["容積率"] = youseki

        # 明示的に地区カラムを入れてモデルに渡す
        df["地区"] = district_full

        # デバッグ出力（確認用）
        print("DEBUG model input df:", df.to_dict(orient="records"))

        # 予測
        if model is not None:
            pred = model.predict(df)[0]
        else:
            pred = dummy_predict(df)[0]

        return {"predicted_price": pred, "used": {"都道府県": prefecture, "市区町村": city, "地区": district_full, "用途": youto}}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={
            "error": "internal server error",
            "message": str(e),
            "received": data
        })

# エンドポイント
@app.post("/predict")
async def predict_endpoint(request: Request):
    payload = await request.json()
    print("DEBUG received payload:", payload)   # デバッグ用ログ
    return predict_logic(payload)

# ルートで静的ページを返す（開発用）
@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("static/index.html")
