"""
prepare_app_model.py

Creates the model files used by the RA-Predict web app, from the XGBoost model that is reported in the manuscript.

Run it in the folder that contains  Best_XGBoost_Model.pkl  and  RA_Training_Data.csv :

    python prepare_app_model.py

It writes the folder  app_model/  with
    RA_Predict_XGB.json   the model in XGBoost's native format (safer across xgboost versions than a pickle)
    background.csv        100 training patients (50 RA, 50 control), used by the probability-scale SHAP explanation
    feature_names.json    the model's own column names (two of them end with a space) and the display names
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import xgboost

SEED = 42
BACKGROUND_N = 100
APP_DIR = "app_model"

train = pd.read_csv("RA_Training_Data.csv")
ALL = [c for c in train.columns if c != "RA"]
Xtr, ytr = train[ALL], train["RA"].values

model = joblib.load("Best_XGBoost_Model.pkl")
assert model.n_features_in_ == 17, "the model does not use 17 features"

rng = np.random.RandomState(SEED)
idx = np.concatenate([
    rng.choice(np.where(ytr == 1)[0], BACKGROUND_N // 2, replace=False),
    rng.choice(np.where(ytr == 0)[0], BACKGROUND_N // 2, replace=False),
])
background = Xtr.iloc[idx]

os.makedirs(APP_DIR, exist_ok=True)
model.save_model(os.path.join(APP_DIR, "RA_Predict_XGB.json"))
background.to_csv(os.path.join(APP_DIR, "background.csv"), index=False)
with open(os.path.join(APP_DIR, "feature_names.json"), "w") as f:
    json.dump({"model_columns": ALL, "display_names": [c.strip() for c in ALL]}, f, indent=2)

# check: the saved JSON model reproduces the predictions of the pickle
check = xgboost.XGBClassifier()
check.load_model(os.path.join(APP_DIR, "RA_Predict_XGB.json"))
diff = np.abs(check.predict_proba(Xtr)[:, 1] - model.predict_proba(Xtr)[:, 1]).max()
print(f"xgboost {xgboost.__version__} | wrote {APP_DIR}/RA_Predict_XGB.json, background.csv ({len(background)} patients), feature_names.json")
print(f"max difference between the JSON model and the pickle on the training set: {diff:.2e}")
print("model parameters:", {k: model.get_params()[k] for k in ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree")})
