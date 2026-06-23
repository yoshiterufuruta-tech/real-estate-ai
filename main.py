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
    df = pd.DataFrame([data])

    # 必要な前処理
    df["年度"] = df["年度"].apply(convert_year_quarter)
    df["面積"] = df["面積"].apply(clean_number)
    df["駅距離"] = df["駅距離"].apply(clean_number)
    df["道路幅"] = df["道路幅"].apply(clean_number)

    # モデルが必要とするカラムを補完（デフォルト値）
    if "建ぺい率" not in df:
        df["建ぺい率"] = 60
    if "容積率" not in df:
        df["容積率"] = 200
    if "用途" not in df:
        df["用途"] = "住宅"
    if "地域" not in df:
        df["地域"] = "住宅地"

    pred = model.predict(df)[0]
    return {"predicted_price": pred}
