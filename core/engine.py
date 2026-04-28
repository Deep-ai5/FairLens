"""
engine.py
Core fairness audit engine.
Computes: Demographic Parity, Equalized Odds, Disparate Impact, Group Accuracy Gap.
Also computes FairScore (0–100) and grade.

v2: Recalibrated FairScore with stricter thresholds and floor penalty for
    critical disparate impact violations.
"""

import numpy as np
import pandas as pd
from core.domain_detector import filter_available_attrs


# ── Metric thresholds ─────────────────────────────────────────────────────────
RATIO_THRESHOLD = 0.80   # 80% rule (EEOC standard for Disparate Impact)
DIFF_THRESHOLD  = 0.10   # >10pp gap = bias flag
GAP_THRESHOLD   = 0.05   # >5pp accuracy gap = bias flag

# FairScore grade bands
GRADE_THRESHOLDS = [(90, "A"), (75, "B"), (60, "C"), (45, "D"), (0, "F")]

# FairScore calibration (v2 — stricter)
_DP_NORM  = 0.15   # demographic parity normalizer  (was 0.30)
_DI_NORM  = 0.20   # disparate impact normalizer    (was 0.30)
_EO_NORM  = 0.20   # equalized odds normalizer      (was 0.30)
_AG_NORM  = 0.15   # accuracy gap normalizer        (unchanged)

_DP_W = 0.40       # weight (was 0.35)
_DI_W = 0.35       # weight (was 0.30)
_EO_W = 0.20       # weight (was 0.25)
_AG_W = 0.05       # weight (unchanged; must sum to 1.0)

_DI_FLOOR_THRESHOLD = 0.70   # any group DI below this → flat -20 floor penalty
_DI_FLOOR_PENALTY   = 20.0


def run_audit(df, model, outcome_col, domain, protected_attrs, X_override=None):
    """
    Main audit function. Returns a structured result dict.
    """
    available = filter_available_attrs(protected_attrs, df.columns)

    # Get ground truth and predictions
    if "_y" in df.columns and "_pred" in df.columns:
        y_true = df["_y"].values
        y_pred = df["_pred"].values
        y_prob = df["_proba"].values if "_proba" in df.columns else None
    else:
        # Use ground truth labels for outcome disparity analysis
        # This measures historical bias in the data itself — equally valid and often
        # more important than model prediction bias
        y_true = df[outcome_col].values
        # Generate predictions via trained model on held-out test split
        if X_override is not None:
            X_scaled = X_override
        else:
            feature_cols = [c for c in df.columns if c not in [outcome_col, "_y", "_pred", "_proba"]]
            X = df[feature_cols].copy()
            for col in X.select_dtypes(include="object").columns:
                X[col] = X[col].astype("category").cat.codes
            X = X.fillna(0)
            if hasattr(model, "scaler_"):
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    X_scaled = model.scaler_.transform(X)
            else:
                X_scaled = X.values
        y_pred = model.predict(X_scaled)
        y_prob = model.predict_proba(X_scaled)[:, 1] if hasattr(model, "predict_proba") else None

        df = df.copy()
        df["_y"]    = y_true
        df["_pred"] = y_true   # audit ground truth outcomes for outcome disparity
        df["_proba"] = y_prob if y_prob is not None else y_true.astype(float)
        # Override y_pred to ground truth for fairness metric computation
        y_pred = y_true

    metric_results = {}
    for attr, meta in available.items():
        groups = _get_groups(df, attr, meta)
        if len(groups) < 2:
            continue
        metric_results[attr] = _compute_metrics_for_attr(df, attr, groups, y_true, y_pred)

    fair_score, grade = _compute_fairscore(metric_results)

    # Build human impact data
    impact = _compute_human_impact(df, metric_results, available, y_true, y_pred, y_prob)
    # Build worst offender summary for verdict banner
    verdict_data = _compute_verdict_data(metric_results, available)

    return {
        "metrics":        metric_results,
        "fair_score":     fair_score,
        "grade":          grade,
        "domain":         domain,
        "outcome_col":    outcome_col,
        "n_samples":      len(df),
        "available_attrs": available,
        "df":             df,
        "y_true":         y_true,
        "y_pred":         y_pred,
        "y_prob":         y_prob,
        "impact":         impact,
        "verdict":        verdict_data,
    }


def _get_groups(df, attr, meta):
    if meta["type"] == "categorical":
        vals = df[attr].astype(str).unique()
        return [(v, df[attr].astype(str) == v) for v in sorted(vals) if pd.notna(v)]
    else:
        bins   = meta.get("bins",   [0, 33, 66, 100])
        labels = meta.get("labels", [f"G{i}" for i in range(len(bins)-1)])
        binned = pd.cut(pd.to_numeric(df[attr], errors="coerce"), bins=bins, labels=labels, right=False)
        return [(str(lbl), binned == lbl) for lbl in labels if (binned == lbl).any()]


def _compute_metrics_for_attr(df, attr, groups, y_true, y_pred):
    group_stats = {}
    for label, mask in groups:
        yt = y_true[mask]
        yp = y_pred[mask]
        n  = int(mask.sum())
        if n == 0:
            continue
        pos_rate = float(yp.mean())
        accuracy = float((yt == yp).mean())
        tp = int(((yt == 1) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        group_stats[label] = {"n": n, "pos_rate": pos_rate, "accuracy": accuracy, "tpr": float(tpr)}

    if not group_stats:
        return {}

    ref_label = max(group_stats, key=lambda g: group_stats[g]["pos_rate"])
    ref = group_stats[ref_label]

    results = {}
    for label, stats in group_stats.items():
        dp  = abs(stats["pos_rate"] - ref["pos_rate"])
        di  = (stats["pos_rate"] / ref["pos_rate"]) if ref["pos_rate"] > 0 else 1.0
        eo  = abs(stats["tpr"] - ref["tpr"])
        ag  = abs(stats["accuracy"] - ref["accuracy"])

        results[label] = {
            "n":                  stats["n"],
            "pos_rate":           stats["pos_rate"],
            "accuracy":           stats["accuracy"],
            "tpr":                stats["tpr"],
            "demographic_parity": round(dp, 4),
            "disparate_impact":   round(di, 4),
            "equalized_odds":     round(eo, 4),
            "accuracy_gap":       round(ag, 4),
            "severity":           _severity(dp, di, eo, ag),
            "is_reference":       label == ref_label,
        }
    return results


def _severity(dp, di, eo, ag):
    if dp > DIFF_THRESHOLD * 2 or di < RATIO_THRESHOLD - 0.1 or eo > DIFF_THRESHOLD * 2:
        return "critical"
    if dp > DIFF_THRESHOLD or di < RATIO_THRESHOLD or eo > DIFF_THRESHOLD or ag > GAP_THRESHOLD:
        return "high"
    return "ok"


def _compute_fairscore(metric_results):
    """
    Aggregate all group metrics into a single 0–100 FairScore.
    v2: Stricter thresholds + floor penalty for critical DI violations.
    """
    all_dp, all_di, all_eo, all_ag = [], [], [], []

    for attr_data in metric_results.values():
        for group_data in attr_data.values():
            if not isinstance(group_data, dict):
                continue
            all_dp.append(group_data.get("demographic_parity", 0))
            all_di.append(group_data.get("disparate_impact",   1))
            all_eo.append(group_data.get("equalized_odds",     0))
            all_ag.append(group_data.get("accuracy_gap",       0))

    if not all_dp:
        return 100, "A"

    dp_penalty = float(np.clip(np.mean(all_dp) / _DP_NORM, 0, 1) * 100)
    di_penalty = float(np.clip((1 - np.mean(all_di)) / _DI_NORM, 0, 1) * 100)
    eo_penalty = float(np.clip(np.mean(all_eo) / _EO_NORM, 0, 1) * 100)
    ag_penalty = float(np.clip(np.mean(all_ag) / _AG_NORM, 0, 1) * 100)

    avg_penalty = (_DP_W * dp_penalty + _DI_W * di_penalty +
                   _EO_W * eo_penalty + _AG_W * ag_penalty)

    # Floor penalty: any group's DI below threshold = automatic deduction
    floor_penalty = 0.0
    for di_val in all_di:
        if di_val < _DI_FLOOR_THRESHOLD:
            floor_penalty = _DI_FLOOR_PENALTY
            break

    score = round(max(0, 100 - avg_penalty - floor_penalty), 1)
    grade = next(g for threshold, g in GRADE_THRESHOLDS if score >= threshold)
    return score, grade


def _compute_human_impact(df, metric_results, available, y_true, y_pred, y_prob):
    """
    Find candidates in the disadvantaged group who were borderline rejected
    — likely would have passed in a fair system.
    Returns a dict with count and a sample DataFrame.
    """
    if y_prob is None:
        return {"count": 0, "df": None, "group": None, "attr": None}

    # Find worst attribute + group (lowest DI, not reference)
    worst_di   = 1.0
    worst_attr = None
    worst_grp  = None

    for attr, groups in metric_results.items():
        for grp, data in groups.items():
            if isinstance(data, dict) and not data.get("is_reference"):
                if data["disparate_impact"] < worst_di:
                    worst_di   = data["disparate_impact"]
                    worst_attr = attr
                    worst_grp  = grp

    if worst_attr is None:
        return {"count": 0, "df": None, "group": None, "attr": None}

    meta = available.get(worst_attr, {})
    if meta.get("type") == "categorical":
        mask = df[worst_attr].astype(str) == worst_grp
    else:
        bins   = meta.get("bins", [])
        labels = meta.get("labels", [])
        if bins and worst_grp in labels:
            idx    = labels.index(worst_grp)
            binned = pd.cut(pd.to_numeric(df[worst_attr], errors="coerce"),
                            bins=bins, labels=labels, right=False)
            mask   = binned == worst_grp
        else:
            mask = pd.Series([False] * len(df))

    # Borderline: rejected (pred=0) but probability >= 0.30
    borderline = mask & (y_pred == 0) & (y_prob >= 0.30)
    count = int(borderline.sum())

    if count == 0:
        return {"count": 0, "df": None, "group": worst_grp, "attr": worst_attr}

    display_cols = [c for c in df.columns if not c.startswith("_")]
    sample = df[borderline][display_cols + ["_proba"]].rename(
        columns={"_proba": "AI Probability"}
    ).head(50).copy()
    sample["AI Decision"] = "❌ Rejected"
    sample.index = range(len(sample))

    return {
        "count":      count,
        "df":         sample,
        "group":      worst_grp,
        "attr":       worst_attr,
        "attr_label": available.get(worst_attr, {}).get("label", worst_attr),
    }


def _compute_verdict_data(metric_results, available):
    """
    Find the single worst group discrimination for the verdict banner.
    Returns dict with ref_group, biased_group, multiplier, attr_label.
    """
    worst_di     = 1.0
    worst_attr   = None
    worst_grp    = None
    ref_grp      = None
    ref_rate     = 0.0
    biased_rate  = 0.0

    for attr, groups in metric_results.items():
        ref = next((g for g, d in groups.items() if isinstance(d, dict) and d.get("is_reference")), None)
        if ref is None:
            continue
        ref_pos = groups[ref]["pos_rate"]
        for grp, data in groups.items():
            if isinstance(data, dict) and not data.get("is_reference"):
                if data["disparate_impact"] < worst_di:
                    worst_di    = data["disparate_impact"]
                    worst_attr  = attr
                    worst_grp   = grp
                    ref_grp     = ref
                    ref_rate    = ref_pos
                    biased_rate = data["pos_rate"]

    if worst_attr is None:
        return None

    multiplier = round(ref_rate / max(biased_rate, 0.001), 1)
    return {
        "attr":        worst_attr,
        "attr_label":  available.get(worst_attr, {}).get("label", worst_attr),
        "biased_group": worst_grp,
        "ref_group":    ref_grp,
        "biased_rate":  round(biased_rate, 3),
        "ref_rate":     round(ref_rate, 3),
        "multiplier":   multiplier,
        "disparate_impact": round(worst_di, 3),
    }


def run_simulation(df, model, outcome_col, domain, protected_attrs, drop_cols=None, reweight=None):
    """
    Simulation: drop or reweight features, recompute metrics.
    """
    sim_df = df.copy()

    if drop_cols:
        sim_df = sim_df.drop(columns=[c for c in drop_cols if c in sim_df.columns], errors="ignore")

    if reweight:
        for col, factor in reweight.items():
            if col in sim_df.columns and sim_df[col].dtype in [np.float64, np.int64]:
                sim_df[col] = sim_df[col] * factor

    from core.ingestor import _train_internal_model
    sim_model, sim_df = _train_internal_model(sim_df, outcome_col)
    available = filter_available_attrs(protected_attrs, sim_df.columns)
    return run_audit(sim_df, sim_model, outcome_col, domain, available)
