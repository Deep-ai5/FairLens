"""
gemini_client.py
All Gemini API interactions:
  - Context-aware chat
  - Fix suggestions generation
  - Text description → synthetic dataset
"""

import os
import json
import textwrap
import pandas as pd
import numpy as np
import streamlit as st
import google.generativeai as genai


def _get_client():
    api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error("Gemini API key not found. Add GEMINI_API_KEY to .streamlit/secrets.toml")
        st.stop()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


def build_audit_context(audit_result):
    """Summarize audit result as a compact JSON string to inject into Gemini prompts."""
    ctx = {
        "domain":     audit_result["domain"],
        "fair_score": audit_result["fair_score"],
        "grade":      audit_result["grade"],
        "n_samples":  audit_result["n_samples"],
        "metrics_summary": {}
    }
    for attr, groups in audit_result["metrics"].items():
        ctx["metrics_summary"][attr] = {
            grp: {
                "pos_rate":           round(data["pos_rate"], 3),
                "demographic_parity": data["demographic_parity"],
                "disparate_impact":   data["disparate_impact"],
                "equalized_odds":     data["equalized_odds"],
                "severity":           data["severity"],
            }
            for grp, data in groups.items()
            if isinstance(data, dict)
        }
    return json.dumps(ctx, indent=2)


SYSTEM_PROMPT = """
You are FairLens, an AI bias auditor assistant. You have access to the full audit results
of an AI decision-making system. Your job is to explain bias findings clearly to
non-technical users (HR managers, compliance officers, university administrators).

Rules:
- Speak in plain language. No jargon unless you define it.
- Be specific: reference actual metric values from the audit context.
- Be direct and actionable. Every response should end with at least one concrete next step.
- If asked about legal implications, reference Indian IT Act, PDPB 2023, or global standards (GDPR, EEOC 80% rule) where relevant.
- Provide comprehensive, detail-rich answers that thoroughly explain the context and implications.
- Never make up metric values. Only use those in the audit context below.

AUDIT CONTEXT:
{audit_context}
"""


def chat(user_message, chat_history, audit_result):
    """
    Send a user message with full audit context.
    Returns assistant reply string.
    """
    model = _get_client()
    ctx   = build_audit_context(audit_result)
    system = SYSTEM_PROMPT.format(audit_context=ctx)

    # Build message history for Gemini
    history = []
    for turn in chat_history:
        role = "model" if turn["role"] == "assistant" else turn["role"]
        history.append({"role": role, "parts": [turn["content"]]})

    history.append({"role": "user", "parts": [user_message]})

    response = model.generate_content(
        [{"role": "user", "parts": [system]}, *history],
        generation_config={"temperature": 0.4, "max_output_tokens": 2048}
    )
    return response.text.strip()


def get_fix_suggestions(audit_result, shap_importance=None):
    """
    Generate ranked fix suggestions using Gemini.
    Returns a list of dicts: [{rank, fix, reason, estimated_improvement}]
    """
    model   = _get_client()
    ctx     = build_audit_context(audit_result)
    shap_str = ""
    if shap_importance is not None:
        top5 = shap_importance.head(5)
        shap_str = "Top 5 features by SHAP importance:\n" + top5.to_string(index=False)

    prompt = f"""
You are a fairness auditor. Based on the audit results below, generate exactly 5 ranked
fix suggestions to reduce algorithmic bias. Return ONLY a valid JSON array, no markdown.

Format: [{{"rank": 1, "fix": "...", "reason": "...", "estimated_improvement": "..."}}]

AUDIT CONTEXT:
{ctx}

{shap_str}

Rules:
- Rank from highest to lowest impact
- Be specific about which feature or process to change
- estimated_improvement should be a realistic percentage range like "15–25% reduction in demographic parity gap"
- Keep each fix under 30 words
"""
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.3, "max_output_tokens": 800}
    )
    raw = response.text.strip()
    # Strip markdown fences if present
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception:
        # Fallback: return generic suggestions
        return [
            {"rank": 1, "fix": "Remove or anonymize protected attributes (gender, age) from model input.", "reason": "Direct use of protected attributes violates fairness principles.", "estimated_improvement": "20–35% reduction in disparate impact gap"},
            {"rank": 2, "fix": "Apply re-sampling: oversample underrepresented groups in training data.", "reason": "Imbalanced training data is the most common root cause of demographic parity violations.", "estimated_improvement": "15–25% improvement in demographic parity"},
            {"rank": 3, "fix": "Add fairness constraints to model training (e.g., equalized odds post-processing).", "reason": "Algorithmic constraints directly optimize for fairness metrics during inference.", "estimated_improvement": "10–20% improvement in equalized odds"},
            {"rank": 4, "fix": "Audit proxy features — college tier, region, postal code — for indirect discrimination.", "reason": "These features often serve as proxies for protected attributes.", "estimated_improvement": "10–15% reduction in disparate impact"},
            {"rank": 5, "fix": "Introduce a human review step for borderline cases flagged by the audit.", "reason": "Human oversight on edge cases prevents systematic exclusion of borderline candidates.", "estimated_improvement": "5–10% improvement in overall accuracy gap"},
        ]


def describe_to_dataset(text_description):
    """
    Takes a plain-text description of an AI decision system and returns
    a synthetic pandas DataFrame and detected domain.
    Used when user has no CSV to upload.
    """
    model = _get_client()
    prompt = f"""
The user described their AI decision system as:
"{text_description}"

Generate a realistic synthetic CSV dataset (100 rows) that represents what their
data might look like. Return ONLY valid JSON with two keys:
  "domain": one of "hr", "banking", "education"
  "data": array of row objects (column names as keys)

Include realistic protected attribute columns (gender, age, region, etc.)
and an outcome column (hired/approved/admitted = 0 or 1).
No markdown, no explanation, just the JSON.
"""
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.5, "max_output_tokens": 2000}
    )
    raw = response.text.strip().replace("```json", "").replace("```", "").strip()
    parsed = json.loads(raw)
    df = pd.DataFrame(parsed["data"])
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df, parsed.get("domain", "hr")
