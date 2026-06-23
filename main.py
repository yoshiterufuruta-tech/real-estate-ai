# main.py
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import traceback
import os
import joblib  # pip install joblib
import json

app = FastAPI()

# CORS（開発用。必要に応じて origin を制限する）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# static 配下に index.html を置く想定
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- モデル読み込み（なければダミー） ---
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

# --- ユーティリティ（あなたの既存関数があれば置き換えてください） ---
def convert_year_quarter(x):
    # 例: "2024年第1四半期" -> 2024.0 + 0.0 など、実装は学習時に合わせる
    return x

def clean_number(x):
    try:
        return float(x)
    except Exception:
        return 0.0

# --- 東京都デフォルト・地区マップ（必要に応じて拡張してください） ---
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

# --- ダミー予測関数（モデルが無い場合のフォールバック） ---
def dummy_predict(df: pd.DataFrame):
    # 単純に面積 * 単価（用途で変える）
    base_unit = 200000  # 1㎡あたりの基準単価（仮）
    # 用途ごとに倍率を変える（仮）
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

# --- predict 実装（堅牢化済み） ---
def predict_logic(data: dict):
    try:
        # 互換キー対応
        prefecture = data.get("都道府県") or data.get("都道府県名") or data.get("prefecture") or ""
        city = data.get("市区町村") or data.get("市区町村名") or data.get("city") or data.get("市区町村名") or ""
        district = data.get("地区") or data.get("地区名") or data.get("district") or data.get("地区名") or ""

        # 空値対策
        prefecture = prefecture or ""
        city = city or ""
        district = district or ""

        # 地名衝突対策：市区町村を前置して一意化
        district_full = f"{city}_{district}" if district else city

        # モデル入力用 DataFrame を明示的に作る（学習時のカラムに合わせて必要に応じて調整）
        model_input = {
            "年度": data.get("年度", "2024年第1四半期"),
            "面積": data.get("面積", 0),
            "駅距離": data.get("駅距離", 0),
            "道路幅": data.get("道路幅", 0),
            "都道府県": prefecture,
            "市区町村": city,
            "地区": district_full
        }
        df = pd.DataFrame([model_input])

        # 前処理
        df["年度"] = df["年度"].apply(convert_year_quarter)
        df["面積"] = df["面積"].apply(clean_number)
        df["駅距離"] = df["駅距離"].apply(clean_number)
        df["道路幅"] = df["道路幅"].apply(clean_number)

        # 用途地域の決定（東京都向けロジック）
        if prefecture == "東京都":
            # 注意：district は元の地区名（結合前）でマップ照合する
            if city == "世田谷区" and district in setagaya_map:
                youto, kenpei, youseki = setagaya_map[district]
            elif city == "渋谷区" and district in shibuya_map:
                youto, kenpei, youseki = shibuya_map[district]
            elif city in tokyo_default:
                youto, kenpei, youseki = tokyo_default[city]
            else:
                youto, kenpei, youseki = ("第一種住居地域", 60, 200)
        else:
            youto, kenpei, youseki = ("住宅地", 60, 200)

        df["用途"] = youto
        df["建ぺい率"] = kenpei
        df["容積率"] = youseki

        # 明示的に地区カラムを入れてモデルに渡す
        df["地区"] = district_full

        # デバッグ出力（本番では logger に切替）
        print("DEBUG model input df:", df.to_dict(orient="records"))

        # 予測
        if model is not None:
            # モデルの入力カラム順や前処理が学習時と一致するように調整してください
            pred = model.predict(df)[0]
        else:
            pred = dummy_predict(df)[0]

        return {"predicted_price": pred, "used": {"都道府県": prefecture, "市区町村": city, "地区": district_full, "用途": youto}}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- エンドポイント ---
@app.post("/predict")
async def predict_endpoint(request: Request):
    payload = await request.json()
    print("DEBUG received payload:", payload)   # 一時的に追加して受信内容を確認
    return predict_logic(payload)

# --- ルートで静的ページを返す（開発用） ---
@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse("static/index.html")
