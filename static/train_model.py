import pandas as pd
import numpy as np
import joblib
import json
import os
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
# 3. 列名をモデル用にリネーム
# ============================

df = df.rename(columns={
    "面積（㎡）": "面積",
    "建築年": "築年数",
    "最寄駅：距離（分）": "駅距離",
    "前面道路：幅員（ｍ）": "道路幅",
    "建ぺい率（％）": "建ぺい率",
    "容積率（％）": "容積率",
    "取引価格（総額）": "価格"
})

# ============================
# 4. 目的変数と説明変数
# ============================

y = df["価格"]
X = df.drop(columns=["価格"])

# ============================
# 5. 数値列・カテゴリ列を自動判定
# ============================

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()

# ============================
# 6. LightGBM 用前処理
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

# ============================
# 7. LightGBM モデル
# ============================

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("regressor", LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=-1,
        random_state=42
    ))
])

# ============================
# 8. 学習
# ============================

model.fit(X, y)

# ============================
# 9. モデル保存（超軽量）
# ============================

joblib.dump(model, "model.pkl", compress=3)
print("model.pkl を生成しました（LightGBM版・圧縮済み）")

# ============================
# 10. static フォルダ作成
# ============================

os.makedirs("static", exist_ok=True)

# ============================
# 11. 市区町村 → 地区名辞書を生成
# ============================

city_to_districts = (
    df.groupby("市区町村名")["地区名"]
      .apply(lambda x: sorted(set(x.dropna())))
      .to_dict()
)

with open("static/city_to_districts.json", "w", encoding="utf-8") as f:
    json.dump(city_to_districts, f, ensure_ascii=False, indent=2)

print("city_to_districts.json を生成しました")

# ============================
# 12. 市区町村一覧も生成
# ============================

cities = sorted(set(df["市区町村名"].dropna()))

with open("static/cities.json", "w", encoding="utf-8") as f:
    json.dump(cities, f, ensure_ascii=False, indent=2)

print("cities.json を生成しました")
