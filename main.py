import pandas as pd
import numpy as np
import joblib
import json
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from lightgbm import LGBMRegressor

# ============================
# 1. CSV 読み込み
# ============================

df = pd.read_csv(
    r"C:\Users\yoshi\AI\Tokyo_20244_20254.csv",
    encoding="cp932"
)

# ============================
# 2. 必要な列だけ抽出
# ============================

use_cols = [
    "都道府県名",
    "市区町村名",
    "地区名",
    "面積（㎡）",
    "建築年",
    "最寄駅：距離（分）",
    "前面道路：幅員（ｍ）",
    "建ぺい率（％）",
    "容積率（％）",
    "用途",
    "取引価格（総額）"
]

df = df[use_cols].copy()

# ============================
# 3. ★ 根本バグ修正：築年数を正しく計算
# ============================

# 本来は取引時点の年を使うべきだが、ここでは 2024 とする
df["築年数"] = 2024 - df["建築年"]

# 建築年はもう不要
df = df.drop(columns=["建築年"])

# ============================
# 4. 列名変換
# ============================

df = df.rename(columns={
    "面積（㎡）": "面積",
    "最寄駅：距離（分）": "駅距離",
    "前面道路：幅員（ｍ）": "道路幅",
    "建ぺい率（％）": "建ぺい率",
    "容積率（％）": "容積率",
    "取引価格（総額）": "価格"
})

# ============================
# 5. 地区名を強化（市区町村 + 地区）
# ============================

df["地区名"] = df["市区町村名"] + "_" + df["地区名"].fillna("")

# ============================
# 6. 地区平均価格・市区町村平均価格
# ============================

df["地区平均価格"] = df.groupby("地区名")["価格"].transform("mean")
df["市区町村平均価格"] = df.groupby("市区町村名")["価格"].transform("mean")

# ============================
# 7. 目的変数と説明変数
# ============================

y = df["価格"]
X = df.drop(columns=["価格"])

# ============================
# 8. 数値列・カテゴリ列
# ============================

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()

# 地区名・市区町村名はカテゴリ扱い
for col in ["市区町村名", "地区名"]:
    if col in numeric_cols:
        numeric_cols.remove(col)
    if col not in categorical_cols:
        categorical_cols.append(col)

# ============================
# 9. 前処理 + LightGBM
# ============================

preprocess = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="mean"), numeric_cols),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_cols)
    ]
)

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("regressor", LGBMRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=-1,
        random_state=42
    ))
])

# ============================
# 10. 学習
# ============================

model.fit(X, y)

# ============================
# 11. 保存
# ============================

joblib.dump(model, "model.pkl")

# ============================
# 12. JSON（UI 用）
# ============================

city_to_districts = (
    df.groupby("市区町村名")["地区名"]
      .apply(lambda x: sorted(set(x.dropna())))
      .to_dict()
)

with open("static/city_to_districts.json", "w", encoding="utf-8") as f:
    json.dump(city_to_districts, f, ensure_ascii=False, indent=2)

cities = sorted(set(df["市区町村名"].dropna()))
with open("static/cities.json", "w", encoding="utf-8") as f:
    json.dump(cities, f, ensure_ascii=False, indent=2)
