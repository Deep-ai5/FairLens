"""
live_probe.py — C6: Counterfactual fairness testing against live prediction APIs.

Generates synthetic candidate pairs that are identical except for one protected
attribute, sends them to a live endpoint, and measures outcome disparity.
This is what real-world AI auditors do in production.
"""

import time
import numpy as np
import pandas as pd
import requests
import streamlit as st


# ── Probe pair generators per domain ─────────────────────────────────────────

PROBE_TEMPLATES = {
    "hr": {
        "base": {
            "experience_yrs": 5,
            "skills_score":   75,
            "college_tier":   "Tier 2",
            "graduation_year": 2019,
            "age": 28,
        },
        "protected_pairs": [
            ("gender", "Male", "Female"),
            ("region", "Metro", "Rural"),
            ("college_tier", "Tier 1", "Tier 3"),
        ],
        "outcome_field": "shortlisted",
    },
    "banking": {
        "base": {
            "credit_score":    700,
            "loan_amount":     500000,
            "employment_type": "Salaried",
            "income_bracket":  "Middle",
            "age": 35,
        },
        "protected_pairs": [
            ("gender",    "Male",   "Female"),
            ("area_type", "Urban",  "Rural"),
            ("employment_type", "Salaried", "Self-Employed"),
        ],
        "outcome_field": "approved",
    },
    "education": {
        "base": {
            "marks_pct":       72.0,
            "extracurricular": 5,
            "region":          "Metro",
            "age": 18,
        },
        "protected_pairs": [
            ("gender",      "Male",    "Female"),
            ("school_type", "Private", "Government"),
            ("medium",      "English", "Regional Language"),
        ],
        "outcome_field": "admitted",
    },
}


def generate_probe_pairs(domain: str, n_pairs: int = 50) -> list[dict]:
    """
    Generate n_pairs of candidate pairs per protected attribute.
    Each pair = two candidates identical except one protected attribute.
    Returns list of dicts: {pair_id, attr, group_a, group_b, payload_a, payload_b}
    """
    template = PROBE_TEMPLATES.get(domain, PROBE_TEMPLATES["hr"])
    base     = template["base"]
    pairs    = template["protected_pairs"]

    RNG      = np.random.default_rng(42)
    results  = []
    pair_id  = 0

    pairs_per_attr = max(1, n_pairs // len(pairs))

    for attr, val_a, val_b in pairs:
        for _ in range(pairs_per_attr):
            # Add small random noise to numeric fields for diversity
            noisy_base = {}
            for k, v in base.items():
                if isinstance(v, (int, float)) and k != "age":
                    noise = float(RNG.normal(0, v * 0.05))
                    noisy_base[k] = round(v + noise, 1) if isinstance(v, float) else int(v + noise)
                else:
                    noisy_base[k] = v

            payload_a = {**noisy_base, attr: val_a}
            payload_b = {**noisy_base, attr: val_b}

            results.append({
                "pair_id":   pair_id,
                "attr":      attr,
                "group_a":   val_a,
                "group_b":   val_b,
                "payload_a": payload_a,
                "payload_b": payload_b,
            })
            pair_id += 1

    return results


def send_probe(
    url: str,
    payload: dict,
    response_field: str = "prediction",
    auth_token: str = "",
    timeout: int = 5,
) -> tuple[float | None, str]:
    """
    Send a single probe payload to the API.
    Returns (prediction_value, error_message).
    prediction_value: float 0–1 (probability) or int 0/1, or None on failure.
    """
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        # Try to extract prediction from response
        if response_field in data:
            val = data[response_field]
        elif "prediction" in data:
            val = data["prediction"]
        elif "result" in data:
            val = data["result"]
        elif "score" in data:
            val = data["score"]
        elif "probability" in data:
            val = data["probability"]
        else:
            # Try first numeric value in response
            for v in data.values():
                if isinstance(v, (int, float, bool)):
                    val = v
                    break
            else:
                return None, f"Field '{response_field}' not found in response: {list(data.keys())}"

        # Normalize to float
        if isinstance(val, bool):
            return float(val), ""
        if isinstance(val, (int, float)):
            return float(val), ""
        if isinstance(val, str):
            if val.lower() in ("true", "yes", "1", "approved", "shortlisted", "admitted"):
                return 1.0, ""
            if val.lower() in ("false", "no", "0", "rejected", "denied"):
                return 0.0, ""
        return None, f"Could not parse prediction value: {val}"

    except requests.exceptions.Timeout:
        return None, "Request timed out (5s)"
    except requests.exceptions.ConnectionError:
        return None, "Connection refused — is the API running?"
    except requests.exceptions.HTTPError as e:
        return None, f"HTTP {e.response.status_code}: {e.response.text[:100]}"
    except Exception as e:
        return None, str(e)


def run_live_audit(
    url: str,
    domain: str,
    n_pairs: int = 50,
    response_field: str = "prediction",
    auth_token: str = "",
    progress_callback=None,
) -> dict:
    """
    Full live audit: generate pairs → send to API → compute disparity metrics.
    Returns a result dict compatible with the main audit result format.
    """
    probe_pairs = generate_probe_pairs(domain, n_pairs)
    total       = len(probe_pairs) * 2
    done        = 0
    errors      = 0
    records     = []

    for pair in probe_pairs:
        for side, grp in [("a", pair["group_a"]), ("b", pair["group_b"])]:
            payload = pair[f"payload_{side}"]
            val, err = send_probe(url, payload, response_field, auth_token)

            if val is None:
                errors += 1
            else:
                # Threshold at 0.5 for binary outcome
                outcome = int(val >= 0.5)
                records.append({
                    "pair_id":   pair["pair_id"],
                    "attr":      pair["attr"],
                    "group":     grp,
                    "side":      side,
                    "raw_score": val,
                    "outcome":   outcome,
                })

            done += 1
            if progress_callback:
                progress_callback(done / total)
            time.sleep(0.05)   # be polite to the API

    if not records:
        return {
            "success": False,
            "error":   "All requests failed. Check the URL and API format.",
            "errors":  errors,
        }

    df = pd.DataFrame(records)

    # Compute disparity per attribute
    attr_results = {}
    for attr in df["attr"].unique():
        attr_df = df[df["attr"] == attr]
        group_stats = {}
        for grp in attr_df["group"].unique():
            grp_df    = attr_df[attr_df["group"] == grp]
            pos_rate  = grp_df["outcome"].mean()
            n         = len(grp_df)
            group_stats[grp] = {"pos_rate": float(pos_rate), "n": int(n)}

        if len(group_stats) < 2:
            continue

        ref_grp  = max(group_stats, key=lambda g: group_stats[g]["pos_rate"])
        ref_rate = group_stats[ref_grp]["pos_rate"]

        per_group = {}
        for grp, stats in group_stats.items():
            dp = abs(stats["pos_rate"] - ref_rate)
            di = (stats["pos_rate"] / ref_rate) if ref_rate > 0 else 1.0
            per_group[grp] = {
                "n":                  stats["n"],
                "pos_rate":           stats["pos_rate"],
                "demographic_parity": round(dp, 4),
                "disparate_impact":   round(di, 4),
                "equalized_odds":     0.0,   # not computable without ground truth
                "accuracy_gap":       0.0,
                "severity":           _live_severity(dp, di),
                "is_reference":       grp == ref_grp,
            }
        attr_results[attr] = per_group

    # Build verdict
    worst_di   = 1.0
    verdict    = None
    for attr, groups in attr_results.items():
        ref = next((g for g,d in groups.items() if d.get("is_reference")), None)
        if not ref: continue
        for grp, data in groups.items():
            if not data.get("is_reference") and data["disparate_impact"] < worst_di:
                worst_di = data["disparate_impact"]
                verdict  = {
                    "attr":          attr,
                    "attr_label":    attr.replace("_", " ").title(),
                    "biased_group":  grp,
                    "ref_group":     ref,
                    "biased_rate":   data["pos_rate"],
                    "ref_rate":      groups[ref]["pos_rate"],
                    "multiplier":    round(groups[ref]["pos_rate"] / max(data["pos_rate"], 0.001), 1),
                    "disparate_impact": round(worst_di, 3),
                }

    # Compute a live FairScore (simplified — no ground truth)
    all_di = [
        d["disparate_impact"]
        for groups in attr_results.values()
        for d in groups.values()
        if isinstance(d, dict) and not d.get("is_reference")
    ]
    if all_di:
        avg_di    = np.mean(all_di)
        di_penalty = np.clip((1 - avg_di) / 0.20, 0, 1) * 100
        floor     = 20.0 if any(v < 0.70 for v in all_di) else 0.0
        score     = round(max(0, 100 - di_penalty - floor), 1)
    else:
        score = 100.0

    grade = next(g for t, g in [(90,"A"),(75,"B"),(60,"C"),(45,"D"),(0,"F")] if score >= t)

    return {
        "success":     True,
        "metrics":     attr_results,
        "fair_score":  score,
        "grade":       grade,
        "verdict":     verdict,
        "n_pairs":     len(probe_pairs),
        "n_requests":  total,
        "errors":      errors,
        "success_rate": round((total - errors) / total * 100, 1),
        "df":          df,
        "domain":      domain,
        "url":         url,
    }


def _live_severity(dp, di):
    if dp > 0.20 or di < 0.70: return "critical"
    if dp > 0.10 or di < 0.80: return "high"
    return "ok"
