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

    # 前処理
    df["年度"] = df["年度"].apply(convert_year_quarter)
    df["面積"] = df["面積"].apply(clean_number)
    df["駅距離"] = df["駅距離"].apply(clean_number)
    df["道路幅"] = df["道路幅"].apply(clean_number)

    prefecture = data.get("都道府県")
    city = data.get("市区町村")
    district = data.get("地区")

    # 東京23区＋市部のデフォルト用途地域
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

    # 世田谷区の地区マップ（例）
    setagaya_map = {
        "三宿": ("第一種低層住居専用地域", 50, 100),
        "三軒茶屋": ("商業地域", 80, 400),
        "上北沢": ("第一種低層住居専用地域", 50, 100),
        "上馬": ("第一種中高層住居専用地域", 60, 200),
        "北沢": ("第一種住居地域", 60, 200),
    }

    # 渋谷区の地区マップ（例）
    shibuya_map = {
        "神宮前": ("第一種住居地域", 60, 200),
        "代々木": ("商業地域", 80, 400),
        "恵比寿": ("商業地域", 80, 400),
        "広尾": ("第一種中高層住居専用地域", 60, 200),
    }

    # 用途地域の決定
    if prefecture == "東京都":
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
    df["地域"] = "住宅地"

    pred = model.predict(df)[0]
    return {"predicted_price": pred}
