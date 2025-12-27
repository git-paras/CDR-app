import numpy as np
import pandas as pd
import joblib

# ---------- Load artifacts ----------
MODEL_PATH = r"C:\Users\paras\Desktop\project\artifacts\model_xgb.pkl"
SCALER_PATH = r"C:\Users\paras\Desktop\project\artifacts\scaler.joblib"
PREPROCESS_PATH = r"C:\Users\paras\Desktop\project\artifacts\preprocessing_values.joblib"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
preprocess_cfg = joblib.load(PREPROCESS_PATH)

FINAL_FEATURES = preprocess_cfg["final_features"]
PERCENTILE_CAPS = preprocess_cfg["percentile_caps"]
FIXED_DELINQ_CAP = preprocess_cfg["fixed_delinquency_cap"]
LOG_FEATURES = preprocess_cfg["log_transform_features"]

DEFAULT_THRESHOLD = 0.7619

# ---------- Preprocessing ----------
def preprocess_input(raw_input: dict) -> pd.DataFrame:
    df = pd.DataFrame([raw_input])


    # Cap delinquency features
    for col in ["Late3059", "Late90", "Late6089", "Deps"]:
        df[col] = df[col].clip(upper=FIXED_DELINQ_CAP)

    # Percentile caps
    for col, cap in PERCENTILE_CAPS.items():
        df[col] = df[col].clip(upper=cap)

    # Log transforms
    for col in LOG_FEATURES:
        df[f"{col}_log"] = np.log1p(df[col])

    # Drop raw columns
    df = df.drop(columns=LOG_FEATURES)

    # Enforce feature order
    df = df[FINAL_FEATURES]

    # Scale
    df_scaled = scaler.transform(df)
    return pd.DataFrame(df_scaled, columns=FINAL_FEATURES)

# ---------- Prediction ----------
def predict_default(raw_input: dict, threshold: float = DEFAULT_THRESHOLD) -> dict:
    X = preprocess_input(raw_input)

    prob_default = model.predict_proba(X)[0, 1]
    decision = prob_default >= threshold

    return {
        "probability": float(prob_default),
        "prediction": "Default" if decision else "No Default",
        "threshold_used": threshold
    }