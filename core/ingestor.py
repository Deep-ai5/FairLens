"""
ingestor.py
Handles all three input modes:
  1. CSV only  → trains a lightweight LogisticRegression internally
  2. CSV + model file (.pkl / .joblib) → loads the model
  3. Text description → Gemini interprets and generates a synthetic demo dataset
"""

import io
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from core.domain_detector import detect_domain, get_protected_attrs
from core.engine import run_audit


DOMAIN_OUTCOME_MAP = {
    "hr":        ["hired", "shortlisted", "selected", "outcome", "label", "result"],
    "banking":   ["approved", "loan_status", "default", "outcome", "label", "result"],
    "education": ["admitted", "accepted", "outcome", "label", "result", "pass"],
}


def ingest_input(uploaded_csv, uploaded_model, text_description):
    """
    Main entry. Returns audit_result dict or raises with a user-friendly message.
    """
    if uploaded_csv is not None:
        df = _load_csv(uploaded_csv)
        domain = detect_domain(df.columns.tolist())
        outcome_col = _find_outcome_col(df, domain)

        if uploaded_model is not None:
            model = _load_model(uploaded_model)
            # Attach predictions to df using the external model
            feature_cols = [c for c in df.columns if c not in [outcome_col]]
            X = df[feature_cols].copy()
            for col in X.select_dtypes(include="object").columns:
                X[col] = X[col].astype("category").cat.codes
            X = X.fillna(0)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if hasattr(model, "scaler_"):
                    X_scaled = model.scaler_.transform(X)
                else:
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    model.scaler_ = scaler
                if not hasattr(model, "feature_names_"):
                    model.feature_names_ = X.columns.tolist()
            y = df[outcome_col].values
            from sklearn.preprocessing import LabelEncoder
            if df[outcome_col].dtype == object:
                y = LabelEncoder().fit_transform(y)
            # Convert X_scaled to DataFrame to allow column selection
            if not isinstance(X_scaled, pd.DataFrame):
                X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

            # Step 1: Get the feature names the model was trained on
            if hasattr(model, "feature_names_in_"):
                expected_features = list(model.feature_names_in_)
            else:
                # Fallback: manually list the 5 features the model was trained on
                expected_features = ["feature1", "feature2", "feature3", "feature4", "feature5"]  # ← replace with real names

            # Step 2: Check all expected features exist in your dataframe
            missing = [f for f in expected_features if f not in X_scaled.columns]
            if missing:
                raise ValueError(f"CSV is missing columns the model needs: {missing}")

            # Step 3: Select only the model's features (drops the extra columns)
            X_model = X_scaled[expected_features]

            # Step 4: Predict
            proba = model.predict_proba(X_model)[:, 1] if hasattr(model, "predict_proba") else y.astype(float)
            df = df.copy()
            df["_y"]    = y
            df["_pred"] = y        # ground truth for fairness metrics
            df["_proba"] = proba

            # Step 5: Attach the aligned feature matrix to the model so
            # engine.py / SHAP always receives the correct 5-column input.
            model.X_audit_ = X_model
            model.feature_names_ = expected_features
        else:
            model, df = _train_internal_model(df, outcome_col)

        protected = get_protected_attrs(domain)
        # Pass X_audit_ (feature-aligned matrix) so engine.py/SHAP uses
        # the correct columns — not the raw df which has more cols.
        X_for_audit = getattr(model, "X_audit_", None)
        result = run_audit(df, model, outcome_col, domain, protected, X_override=X_for_audit)
        return result, domain, df, model

    elif text_description and len(text_description.strip()) > 20:
        # Generate synthetic dataset via Gemini, then audit it
        from core.gemini_client import describe_to_dataset
        df, domain = describe_to_dataset(text_description)
        outcome_col = _find_outcome_col(df, domain)
        model, df = _train_internal_model(df, outcome_col)
        protected = get_protected_attrs(domain)
        result = run_audit(df, model, outcome_col, domain, protected)
        return result, domain, df, model

    else:
        st.warning("Please upload a CSV dataset or describe your AI system.")
        return None, None, None, None


def _load_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        return df
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        raise


def _load_model(uploaded_file):
    ext = uploaded_file.name.split(".")[-1].lower()
    try:
        data = uploaded_file.read()
        # Try joblib first, fallback to pickle
        try:
            obj = joblib.load(io.BytesIO(data))
        except Exception:
            obj = pickle.load(io.BytesIO(data))

        # Handle tuple format: (model, X_df, y_series, extras)
        # This is a common export pattern from training scripts
        if isinstance(obj, tuple):
            model = next((item for item in obj if hasattr(item, "predict")), None)
            if model is None:
                st.error("Could not find a model with a predict() method inside the uploaded file.")
                raise ValueError("No model in tuple")
            # If X_df is in the tuple, attach feature names to model
            for item in obj:
                if hasattr(item, "columns") and hasattr(model, "feature_names_in_"):
                    model.feature_names_ = list(item.columns)
                    break
            return model

        # Handle dict format: {"model": ..., "scaler": ...}
        if isinstance(obj, dict):
            model = obj.get("model") or obj.get("clf") or obj.get("estimator")
            if model is None:
                model = next((v for v in obj.values() if hasattr(v, "predict")), None)
            if model is None:
                st.error("Could not find a model inside the uploaded dictionary.")
                raise ValueError("No model in dict")
            if "scaler" in obj:
                model.scaler_ = obj["scaler"]
            return model

        # Plain model object
        if hasattr(obj, "predict"):
            return obj

        st.error("Uploaded file does not contain a recognizable model object.")
        raise ValueError("Unrecognized format")

    except Exception as e:
        st.error(f"Could not load model: {e}")
        raise


def _find_outcome_col(df, domain):
    candidates = DOMAIN_OUTCOME_MAP.get(domain, []) + ["outcome", "label", "target", "y"]
    for col in df.columns:
        if col in candidates:
            return col
    # Last column as fallback
    return df.columns[-1]


def _train_internal_model(df, outcome_col):
    """
    Trains a LogisticRegression on the uploaded CSV.
    Encodes categoricals, scales numerics.
    Returns (model, df_with_predictions).
    """
    X = df.drop(columns=[outcome_col]).copy()
    y = df[outcome_col].copy()

    # Encode target if needed
    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        y = y.values

    # Encode categoricals in features
    encoders = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # Fill NaNs
    X = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    # Attach predictions back to df
    all_proba = clf.predict_proba(X_scaled)[:, 1]
    df = df.copy()
    # Store ground truth in both _y and _pred — fairness is measured on
    # historical outcome disparity (what actually happened), not model predictions.
    # Model predictions (all_preds) are available via _proba thresholding for simulation.
    df["_pred"]  = y          # ground truth outcomes for fairness metrics
    df["_proba"] = all_proba  # model probability scores for simulation + SHAP
    df["_y"]    = y

    # Store feature names and scaler on model for SHAP
    clf.feature_names_ = X.columns.tolist()
    clf.scaler_ = scaler
    clf.encoders_ = encoders
    clf.outcome_col_ = outcome_col

    return clf, df