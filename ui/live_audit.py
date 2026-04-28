"""
live_audit.py — C6: Live API bias probe UI page.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from core.live_probe import run_live_audit, PROBE_TEMPLATES


DOMAIN_LABELS = {
    "hr":        "HR / Hiring",
    "banking":   "Banking / Loans",
    "education": "Education / Admissions",
}

SEVERITY_COLORS = {"critical": "#E24B4A", "high": "#EF9F27", "ok": "#1D9E75"}
GRADE_COLORS    = {"A": "#065F46", "B": "#1D9E75", "C": "#854F0B", "D": "#991B1B", "F": "#7F1D1D"}


def render_live_audit():
    st.markdown("## 🔌 Live API Bias Probe")
    st.markdown(
        "Paste any live prediction endpoint URL. FairLens generates **counterfactual probe pairs** — "
        "candidates identical in every way except one protected attribute — sends them to your API, "
        "and measures whether the model treats them differently."
    )

    st.info(
        "💡 **What this tests:** If your API returns a higher acceptance rate for Male vs Female "
        "candidates with identical qualifications, that's counterfactual unfairness — "
        "the model has learned to discriminate.", icon="ℹ️"
    )

    st.markdown("---")

    # ── Configuration ─────────────────────────────────────────────────────────
    st.markdown("### Configuration")

    col_url, col_domain = st.columns([3, 1])
    with col_url:
        url = st.text_input(
            "Prediction API URL",
            placeholder="https://your-model-api.com/predict",
            help="Must accept POST requests with JSON body and return a prediction field.",
        )
    with col_domain:
        domain = st.selectbox(
            "Domain",
            options=["hr", "banking", "education"],
            format_func=lambda x: DOMAIN_LABELS[x],
        )

    col_field, col_token, col_pairs = st.columns([2, 2, 1])
    with col_field:
        response_field = st.text_input(
            "Response field name",
            value="prediction",
            help="The JSON key in the API response that contains the prediction (e.g. 'prediction', 'result', 'score').",
        )
    with col_token:
        auth_token = st.text_input(
            "Bearer token (optional)",
            type="password",
            placeholder="Leave blank if no auth required",
        )
    with col_pairs:
        n_pairs = st.slider("Probe pairs", min_value=10, max_value=100, value=30, step=10)

    # Show what will be sent
    with st.expander("📋 Preview probe payload format"):
        template = PROBE_TEMPLATES.get(domain, PROBE_TEMPLATES["hr"])
        st.markdown(f"**Base candidate** (fields that stay constant across pairs):")
        st.json(template["base"])
        st.markdown(f"**Protected attribute pairs** that will be swapped:")
        for attr, val_a, val_b in template["protected_pairs"]:
            st.markdown(f"- `{attr}`: **{val_a}** vs **{val_b}**")
        st.markdown(f"**Expected response format:** `{{ \"{response_field}\": 1 }}` or `{{ \"{response_field}\": 0.73 }}`")

    st.markdown("---")

    # ── Run ───────────────────────────────────────────────────────────────────
    if st.button("🚀 Run Live Audit", type="primary", disabled=not url):
        if not url.startswith("http"):
            st.error("URL must start with http:// or https://")
            st.stop()

        progress_bar = st.progress(0, text="Sending probe pairs…")
        status_text  = st.empty()

        def update_progress(frac):
            progress_bar.progress(frac, text=f"Sending probes… {int(frac*100)}%")

        with st.spinner(""):
            result = run_live_audit(
                url=url,
                domain=domain,
                n_pairs=n_pairs,
                response_field=response_field,
                auth_token=auth_token,
                progress_callback=update_progress,
            )

        progress_bar.empty()

        if not result.get("success"):
            st.error(f"❌ Audit failed: {result.get('error', 'Unknown error')}")
            st.stop()

        st.session_state["live_result"] = result
        st.rerun()

    # ── Results ───────────────────────────────────────────────────────────────
    result = st.session_state.get("live_result")
    if not result or not result.get("success"):
        return

    _render_live_results(result)


def _render_live_results(result):
    score   = result["fair_score"]
    grade   = result["grade"]
    verdict = result.get("verdict")
    errors  = result["errors"]
    total   = result["n_requests"]

    # ── Stats bar ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("FairScore",     f"{score} / 100")
    with c2: st.metric("Grade",         grade)
    with c3: st.metric("Pairs Sent",    f"{result['n_pairs']}")
    with c4: st.metric("Success Rate",  f"{result['success_rate']}%",
                       delta=f"{errors} failed" if errors else None,
                       delta_color="inverse")

    # ── Verdict banner ────────────────────────────────────────────────────────
    if score >= 75:
        bg, border, icon = "#D1FAE5", "#059669", "✅"
        msg = "No significant counterfactual bias detected. The model treats probe pairs similarly across protected attributes."
    elif score >= 60:
        bg, border, icon = "#FEF3C7", "#D97706", "⚠️"
        if verdict:
            msg = (f"<b>{verdict['biased_group']}</b> candidates are accepted at "
                   f"<b>{verdict['biased_rate']:.1%}</b> vs <b>{verdict['ref_rate']:.1%}</b> "
                   f"for <b>{verdict['ref_group']}</b> — a <b>{verdict['multiplier']}x</b> gap "
                   f"on otherwise identical probes.")
        else:
            msg = "Moderate counterfactual bias detected."
    else:
        bg, border, icon = "#FEE2E2", "#DC2626", "🚨"
        if verdict:
            msg = (f"Critical counterfactual bias. Your API accepts <b>{verdict['ref_group']}</b> "
                   f"candidates at <b>{verdict['multiplier']}x</b> the rate of <b>{verdict['biased_group']}</b> "
                   f"candidates with <b>identical qualifications</b>. "
                   f"The model has learned to discriminate on {verdict['attr_label']}.")
        else:
            msg = "Critical counterfactual bias detected."

    st.markdown(f"""
    <div style="background:{bg};border-left:5px solid {border};border-radius:8px;
                padding:16px 20px;margin:16px 0;">
        <div style="font-size:15px;font-weight:700;color:#111;margin-bottom:4px;">
            {icon} Live Audit — FairScore {score} / 100 (Grade {grade})
        </div>
        <div style="font-size:14px;color:#333;line-height:1.6;">{msg}</div>
        <div style="font-size:12px;color:#888;margin-top:8px;">
            Endpoint: {result['url']} &nbsp;·&nbsp; Domain: {DOMAIN_LABELS.get(result['domain'],'?')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Per-attribute results ─────────────────────────────────────────────────
    st.markdown("### Results by Protected Attribute")

    for attr, groups in result["metrics"].items():
        attr_label = attr.replace("_", " ").title()
        with st.expander(f"**{attr_label}**", expanded=True):
            cols = st.columns(len(groups))
            for i, (grp, data) in enumerate(groups.items()):
                sev    = data.get("severity", "ok")
                color  = SEVERITY_COLORS.get(sev, "#333")
                sev_bg = {"critical":"#FEE2E2","high":"#FEF3C7","ok":"#D1FAE5"}.get(sev,"#F0F0F0")
                ref_tag = " ★" if data.get("is_reference") else ""
                with cols[i % len(cols)]:
                    st.markdown(f"""
                    <div style="background:#fff;border:1px solid #eee;border-radius:10px;
                                padding:14px;margin:4px 0;">
                        <div style="font-size:12px;font-weight:600;color:#555;margin-bottom:6px;">
                            {grp}{ref_tag}
                        </div>
                        <div style="font-size:11px;color:#888;">Accept rate</div>
                        <div style="font-size:22px;font-weight:700;color:{color};">
                            {data['pos_rate']:.1%}
                        </div>
                        <div style="font-size:10px;color:#999;margin-top:6px;">
                            n={data['n']} probes<br>
                            Dem. Parity: <b>{data['demographic_parity']:.3f}</b><br>
                            Disp. Impact: <b>{data['disparate_impact']:.3f}</b>
                        </div>
                        <div style="margin-top:8px;">
                            <span style="background:{sev_bg};color:{color};border-radius:20px;
                                         padding:2px 8px;font-size:10px;font-weight:600;">
                                {sev.upper()}
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    st.markdown("### Acceptance Rate by Group (Live Probes)")
    rows = []
    for attr, groups in result["metrics"].items():
        for grp, data in groups.items():
            rows.append({
                "Attribute": attr.replace("_"," ").title(),
                "Group":     grp,
                "Rate":      data["pos_rate"],
                "Severity":  data.get("severity","ok"),
            })

    if rows:
        df_plot   = pd.DataFrame(rows)
        color_map = {"critical":"#E24B4A","high":"#EF9F27","ok":"#1D9E75"}
        fig = px.bar(df_plot, x="Group", y="Rate", color="Severity",
                     color_discrete_map=color_map, facet_col="Attribute",
                     facet_col_wrap=3, height=300,
                     labels={"Rate":"Accept Rate","Group":""})
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white",
                          font={"size":11}, margin=dict(l=10,r=10,t=30,b=10))
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Sample probe pairs ────────────────────────────────────────────────────
    with st.expander("🔍 Show sample probe pairs sent"):
        df_raw = result.get("df")
        if df_raw is not None and len(df_raw) > 0:
            st.dataframe(
                df_raw.head(20).style.format({"raw_score": "{:.3f}"}),
                use_container_width=True, height=300
            )
            st.caption(
                "Each row is one probe request. Pairs with the same pair_id are "
                "identical except for the protected attribute value."
            )
