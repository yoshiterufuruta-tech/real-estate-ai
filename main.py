from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import duckdb
import pandas as pd
import pickle
import re

app = FastAPI()

# 静的ファイル（index.html）を提供
app.mount("/static", StaticFiles(directory="static"), name="static")

# DuckDB 接続
con = duckdb.connect("land.duckdb")

# モデル読み込み
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# 年度変換
def convert_year_quarter(s):
    match = re.match(r"(\d+)年第(\d)四半期", str(s))
    if match:
        year = int(match.group(1))
        q = int(match.group(2))
        return year + q * 0.1
    return None

# 数値クリーニング
def clean_number(x):
    if x is None:
        return None
    s = str(x)
    s = s.replace(",", "").replace("㎡", "").replace("以上", "")
    try:
        return float(s)
    except:
        return None

@app.get("/cities")
def get_cities(pref: str):
    df = con.execute(f"""
        SELECT DISTINCT 市区町村名
        FROM tokyo
        WHERE 都道府県名 = '{pref}'
        ORDER BY 市区町村名
    """).df()
    return df["市区町村名"].tolist()

@app.get("/districts")
def get_districts(city: str):
    df = con.execute(f"""
        SELECT DISTINCT 地区名
        FROM tokyo
        WHERE 市区町村名 = '{city}'
        ORDER BY 地区名
    """).df()
    return df["地区名"].tolist()

@app.post("/predict")
def predict(data: dict):
    # 互換性のため複数キーをチェック
    prefecture = data.get("都道府県") or data.get("都道府県名") or data.get("prefecture")
    city = data.get("市区町村") or data.get("市区町村名") or data.get("city")
    district = data.get("地区") or data.get("地区名") or data.get("district")

    # 地区が空なら空文字にしておく
    city = city or ""
    district = district or ""

    # 市区町村と地区を結合して一意化（地名衝突対策）
    district_full = f"{city}_{district}" if district else city

    df = pd.DataFrame([data])
    # 必要な前処理（既存の関数を使う）
    df["年度"] = df.get("年度", "2024年第1四半期")
    df["面積"] = clean_number(df.get("面積", 0))
    df["駅距離"] = clean_number(df.get("駅距離", 0))
    df["道路幅"] = clean_number(df.get("道路幅", 0))

    # 用途地域決定ロジック（例）
    if prefecture == "東京都":
        if city == "世田谷区" and district in setagaya_map:
            youto, kenpei, youseki = setagaya_map[district]
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

    pred = model.predict(df)[0]
    return {"predicted_price": pred}
