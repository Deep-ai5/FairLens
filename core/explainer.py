"""
explainer.py
SHAP-based feature importance and candidate-level explanation.
"""

import numpy as np
import pandas as pd
import shap
import streamlit as st


@st.cache_data(show_spinner=False)
def compute_shap(_model, _df, outcome_col, max_samples=500):
    """
    Compute SHAP values for the model.
    Returns (shap_values array, feature_names list, X_sample DataFrame).
    Caches result so it only runs once per audit.
    """
    # ── Use the pre-aligned feature matrix stored by ingestor (external model path)
    # This guarantees we pass exactly the columns the model was trained on.
    if hasattr(_model, "X_audit_"):
        X_audit = _model.X_audit_
        feature_cols = list(X_audit.columns) if hasattr(X_audit, "columns") else list(
            getattr(_model, "feature_names_in_",
            getattr(_model, "feature_names_", [f"f{i}" for i in range(X_audit.shape[1])])))
        X = pd.DataFrame(X_audit, columns=feature_cols).copy()
        # Sample for speed
        if len(X) > max_samples:
            X = X.sample(max_samples, random_state=42)
        X_arr = X.values  # already scaled by ingestor
    else:
        # Internal model path — build X from df the normal way
        feature_cols = [
            c for c in _df.columns
            if c not in [outcome_col, "_y", "_pred", "_proba"]
        ]
        X = _df[feature_cols].copy()

        # Encode categoricals
        for col in X.select_dtypes(include="object").columns:
            X[col] = X[col].astype("category").cat.codes
        X = X.fillna(0)

        # Sample for speed
        if len(X) > max_samples:
            X = X.sample(max_samples, random_state=42)

        # Scale if scaler is attached
        X_arr = X.values
        if hasattr(_model, "scaler_"):
            X_arr = _model.scaler_.transform(X_arr)

    try:
        # Tree-based models (RandomForest, XGBoost, etc.)
        if hasattr(_model, "estimators_") or "Forest" in type(_model).__name__ or "Tree" in type(_model).__name__:
            explainer  = shap.TreeExplainer(_model)
            shap_raw   = explainer.shap_values(X_arr)
            # Binary classification returns a list [class0, class1] — take class 1
            shap_values = shap_raw[1] if isinstance(shap_raw, list) else shap_raw
        else:
            explainer   = shap.LinearExplainer(_model, X_arr, feature_perturbation="interventional")
            shap_values = explainer.shap_values(X_arr)
    except Exception:
        # Fallback to KernelExplainer (slower but universal)
        background  = shap.kmeans(X_arr, min(10, len(X_arr)))
        explainer   = shap.KernelExplainer(_model.predict_proba, background)
        shap_values = explainer.shap_values(X_arr, nsamples=50)[1]

    return shap_values, feature_cols, X


def get_feature_importance(shap_values, feature_names):
    """
    Returns a DataFrame with mean |SHAP| per feature, sorted descending.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    return pd.DataFrame({
        "feature":    feature_names,
        "importance": mean_abs,
    }).sort_values("importance", ascending=False).reset_index(drop=True)


def get_candidate_shap(shap_values, feature_names, X_sample, candidate_idx):
    """
    Returns SHAP values for a single candidate (by positional index in X_sample).
    """
    if candidate_idx >= len(X_sample):
        candidate_idx = len(X_sample) - 1
    sv = shap_values[candidate_idx]
    return pd.DataFrame({
        "feature": feature_names,
        "shap":    sv,
        "value":   X_sample.iloc[candidate_idx].values,
    }).sort_values("shap", key=abs, ascending=False)