import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from catboost import CatBoostRegressor
import lightgbm as lgb

from sklearn.ensemble import IsolationForest

def train_fraud_model(df):
    features = df[["age", "vehicle_age", "premium", "accidents"]]

    fraud_model = IsolationForest(contamination=0.05, random_state=42)
    fraud_model.fit(features)

    return fraud_model

def preprocess(df):
    df = df.copy()

    le1 = LabelEncoder()
    le2 = LabelEncoder()

    df["vehicle_type"] = le1.fit_transform(df["vehicle_type"])
    df["region"] = le2.fit_transform(df["region"])

    X = df.drop("claim_amount", axis=1)
    y = df["claim_amount"]

    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_models(df):
    X_train, X_test, y_train, y_test = preprocess(df)

    # ✅ FIX: models dictionary defined properly
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100),
        "Ridge": Ridge(),
        "CatBoost": CatBoostRegressor(verbose=0),
        "LightGBM": lgb.LGBMRegressor()
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)

        results[name] = round(mae, 2)
        trained_models[name] = model

    best_model_name = min(results, key=results.get)

    return trained_models[best_model_name], results