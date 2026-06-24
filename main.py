from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import traceback
import os
import joblib
import re
import numpy as np
from sklearn.impute import SimpleImputer
import httpx

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# モデル読み込み
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

# MLIT API KEY
MLIT_KEY = os.getenv("MLIT_API_KEY")

# -------------------------
# Utility functions
# -------------------------

def clean_number(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def normalize_payload(data: dict):
    prefecture = data.get("都道府県") or data.get("prefecture") or ""
    city = data.get("市区町村") or data.get("city") or ""
    district = data.get("地区") or data.get("district") or ""
    return {
        "年度": data.get("年度", ""),
        "面積": data.get("面積", 0),
        "築年数": data.get("築年数", data.get("年数", 0)),
        "駅距離": data.get("駅距離", 0),
        "道路幅": data.get("道路幅", 0),
        "都道府県": prefecture,
        "市区町村": city,
