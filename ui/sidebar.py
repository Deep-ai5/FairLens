"""
sidebar.py — Left nav + file upload panel (v2 — includes Live Audit nav item)
"""

import streamlit as st
from core.ingestor import ingest_input


DOMAIN_ICONS  = {"hr": "👔", "banking": "🏦", "education": "🎓"}
DOMAIN_LABELS = {"hr": "HR / Hiring", "banking": "Banking / Loans", "education": "Education / Admissions"}

NAV_ITEMS = [
    ("dashboard",   "📊", "Dashboard"),
    ("simulation",  "🔬", "Simulation"),
    ("chat",        "💬", "Ask FairLens"),
    ("live_audit",  "🔌", "Live API Probe"),
    ("report",      "📄", "PDF Report"),
]


def render_sidebar():
    with st.sidebar:
        # Logo
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;padding:4px 0 20px;">
            <div style="width:32px;height:32px;background:#0F6E56;border-radius:8px;
                        display:flex;align-items:center;justify-content:center;font-size:18px;">⚖️</div>
            <div>
                <div style="font-size:17px;font-weight:700;color:#111;line-height:1.1;">FairLens</div>
                <div style="font-size:11px;color:#888;">AI Bias Auditor</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Upload section
        st.markdown('<div style="font-size:11px;font-weight:600;color:#888;letter-spacing:0.06em;'
                    'text-transform:uppercase;margin-bottom:8px;">Upload</div>', unsafe_allow_html=True)

        uploaded_csv   = st.file_uploader("Dataset (CSV)", type=["csv"],
                                          label_visibility="collapsed", key="csv_upload")
        uploaded_model = st.file_uploader("Model (.pkl/.joblib) — optional",
                                          type=["pkl","joblib","jbl"],
                                          label_visibility="collapsed", key="model_upload")

        st.markdown('<div style="font-size:11px;color:#888;margin:6px 0 4px;">'
                    'Or describe your AI system:</div>', unsafe_allow_html=True)
        text_desc = st.text_area("", placeholder="e.g. We use an AI tool to shortlist resumes…",
                                 height=70, label_visibility="collapsed", key="text_desc")

        # Demo buttons
        st.markdown('<div style="font-size:11px;color:#888;margin:8px 0 4px;">Try a demo:</div>',
                    unsafe_allow_html=True)
        demo_cols = st.columns(3)
        with demo_cols[0]:
            if st.button("👔 HR",   use_container_width=True): st.session_state["load_demo"] = "hr"
        with demo_cols[1]:
            if st.button("🏦 Bank", use_container_width=True): st.session_state["load_demo"] = "banking"
        with demo_cols[2]:
            if st.button("🎓 Edu",  use_container_width=True): st.session_state["load_demo"] = "education"

        run_btn = st.button("⚡ Run Audit", type="primary", use_container_width=True)

        if run_btn:
            csv_to_use = uploaded_csv
            if (not csv_to_use and not text_desc and
                    st.session_state.get("load_demo")):
                from data.demo_loader import get_demo_csv
                csv_to_use = get_demo_csv(st.session_state["load_demo"])

            with st.spinner("Running audit…"):
                result, domain, df, model = ingest_input(csv_to_use, uploaded_model, text_desc)

            if result:
                st.session_state.update({
                    "audit_result": result,
                    "domain":       domain,
                    "df":           df,
                    "model":        model,
                    "chat_history": [],
                    "sim_result":   None,
                    "live_result":  None,
                    "page":         "dashboard",
                    "shap_vals":    None,
                    "feat_names":   None,
                    "X_sample":     None,
                    "importance":   None,
                })
                st.rerun()

        # Domain badge
        if st.session_state.get("audit_result"):
            domain  = st.session_state["domain"]
            score   = st.session_state["audit_result"]["fair_score"]
            grade   = st.session_state["audit_result"]["grade"]
            gcol    = {"A":"#065F46","B":"#1D9E75","C":"#854F0B","D":"#991B1B","F":"#7F1D1D"}.get(grade,"#333")
            st.markdown(f"""
            <div style="background:#F7F9FB;border:1px solid #EBEBEB;border-radius:10px;
                        padding:12px 14px;margin:14px 0 6px;">
                <div style="font-size:11px;color:#888;margin-bottom:4px;">Current Audit</div>
                <div style="font-size:13px;font-weight:600;color:#111;">
                    {DOMAIN_ICONS.get(domain,'')} {DOMAIN_LABELS.get(domain, domain)}
                </div>
                <div style="margin-top:6px;display:flex;align-items:baseline;gap:6px;">
                    <span style="font-size:28px;font-weight:700;color:{gcol};">{score}</span>
                    <span style="font-size:13px;color:{gcol};font-weight:600;">Grade {grade}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Navigation
        st.markdown('<div style="font-size:11px;font-weight:600;color:#888;letter-spacing:0.06em;'
                    'text-transform:uppercase;margin-bottom:8px;">Navigate</div>', unsafe_allow_html=True)

        current = st.session_state.get("page", "dashboard")
        for page_id, icon, label in NAV_ITEMS:
            is_active = current == page_id
            if st.button(f"{icon}  {label}", key=f"nav_{page_id}", use_container_width=True,
                         type="primary" if is_active else "secondary"):
                st.session_state["page"] = page_id
                st.rerun()

    return st.session_state.get("page", "dashboard")
