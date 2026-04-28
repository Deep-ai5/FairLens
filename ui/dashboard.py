"""
dashboard.py — Main audit dashboard
v2: Verdict banner · Human impact statement · Candidate comparison · SHAP · Radar · Drill-down
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


SEVERITY_COLORS = {"critical": "#E24B4A", "high": "#EF9F27", "ok": "#1D9E75"}
GRADE_COLORS    = {"A": "#065F46", "B": "#1D9E75", "C": "#854F0B", "D": "#991B1B", "F": "#7F1D1D"}
DOMAIN_LABELS   = {"hr": "HR / Hiring", "banking": "Banking / Loans", "education": "Education / Admissions"}


def render_dashboard():
    result = st.session_state.get("audit_result")
    if not result:
        _render_empty_state()
        return

    _render_verdict_banner(result)          # C3
    _render_human_impact(result)            # C4
    st.markdown("---")
    _render_header(result)
    st.markdown("---")
    _render_metric_cards(result)
    st.markdown("---")
    col_l, col_r = st.columns([1, 1], gap="large")
    with col_l:
        _render_group_chart(result)
    with col_r:
        _render_radar_chart(result)         # C7
    st.markdown("---")
    _render_candidate_comparison(result)    # C5
    st.markdown("---")
    _render_heatmap(result)
    st.markdown("---")
    _render_shap_section(result)
    st.markdown("---")
    _render_candidate_drilldown(result)


# ── C3: VERDICT BANNER ───────────────────────────────────────────────────────
def _render_verdict_banner(result):
    score   = result["fair_score"]
    grade   = result["grade"]
    verdict = result.get("verdict")

    if score >= 75:
        bg, border, icon = "#D1FAE5", "#059669", "✅"
        if verdict:
            msg = f"No critical bias detected. Minor disparities noted below for review."
        else:
            msg = "Fairness metrics are within acceptable thresholds."
    elif score >= 60:
        bg, border, icon = "#FEF3C7", "#D97706", "⚠️"
        if verdict:
            msg = (f"<b>{verdict['biased_group']}</b> candidates are selected at "
                   f"<b>{verdict['biased_rate']:.1%}</b> vs <b>{verdict['ref_rate']:.1%}</b> "
                   f"for {verdict['ref_group']} candidates "
                   f"({verdict['multiplier']}x gap). Remediation recommended before deployment.")
        else:
            msg = "Significant bias detected. Remediation recommended before deployment."
    else:
        bg, border, icon = "#FEE2E2", "#DC2626", "🚨"
        if verdict:
            msg = (f"Your AI is selecting <b>{verdict['ref_group']}</b> candidates at "
                   f"<b>{verdict['multiplier']}x</b> the rate of <b>{verdict['biased_group']}</b> "
                   f"candidates ({verdict['ref_rate']:.1%} vs {verdict['biased_rate']:.1%}). "
                   f"This likely violates EEOC / fair lending standards. Immediate action required.")
        else:
            msg = "Critical bias detected. Immediate action required."

    st.markdown(f"""
    <div style="background:{bg};border-left:5px solid {border};border-radius:8px;
                padding:16px 20px;margin-bottom:8px;">
        <div style="font-size:15px;font-weight:700;color:#111;margin-bottom:4px;">
            {icon} FairScore {score} / 100 — Grade {grade}
        </div>
        <div style="font-size:14px;color:#333;line-height:1.5;">{msg}</div>
    </div>
    """, unsafe_allow_html=True)


# ── C4: HUMAN IMPACT STATEMENT ───────────────────────────────────────────────
def _render_human_impact(result):
    impact = result.get("impact", {})
    count  = impact.get("count", 0)
    if count == 0:
        return

    grp        = impact.get("group", "")
    attr_label = impact.get("attr_label", "")
    df_sample  = impact.get("df")

    col_stat, col_desc = st.columns([1, 3], gap="large")
    with col_stat:
        st.markdown(f"""
        <div style="background:#fff;border:1px solid #eee;border-radius:12px;
                    padding:20px;text-align:center;">
            <div style="font-size:52px;font-weight:800;color:#0F6E56;line-height:1;">
                {count}
            </div>
            <div style="font-size:13px;color:#555;margin-top:4px;line-height:1.4;">
                candidates likely<br>disadvantaged
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_desc:
        st.markdown(f"""
        <div style="padding:8px 0;">
            <div style="font-size:15px;font-weight:700;color:#111;margin-bottom:8px;">
                👤 What this means for real people
            </div>
            <div style="font-size:14px;color:#444;line-height:1.7;">
                <b>{count} {grp}</b> candidates in this dataset were rejected by the AI system
                despite having a meaningful probability of being qualified.
                In a fair system — one without {attr_label.lower()} bias —
                many of these candidates would likely have received a different outcome.
                <br><br>
                These are not statistics. These are real applications that were filtered out
                before a human ever reviewed them.
            </div>
        </div>
        """, unsafe_allow_html=True)

    if df_sample is not None and len(df_sample) > 0:
        with st.expander(f"👁 Show the {min(count, 50)} affected candidates"):
            st.dataframe(df_sample.head(50), use_container_width=True, height=280)


# ── FAIRSCORE HEADER ─────────────────────────────────────────────────────────
def _render_header(result):
    score  = result["fair_score"]
    grade  = result["grade"]
    n      = result["n_samples"]
    grade_c = GRADE_COLORS.get(grade, "#333")

    col_score, col_info = st.columns([1, 3], gap="large")
    with col_score:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            number={"font": {"size": 36, "color": grade_c}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#ccc"},
                "bar":  {"color": grade_c},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  45], "color": "#FEE2E2"},
                    {"range": [45, 60], "color": "#FEF3C7"},
                    {"range": [60, 75], "color": "#ECFDF5"},
                    {"range": [75, 100],"color": "#D1FAE5"},
                ],
            }
        ))
        fig.update_layout(height=200, margin=dict(l=10,r=10,t=20,b=10),
                          paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown(f'<div style="text-align:center;margin-top:-12px;font-size:18px;font-weight:700;color:{grade_c};">Grade {grade}</div>',
                    unsafe_allow_html=True)

    with col_info:
        critical = sum(1 for ad in result["metrics"].values()
                       for d in ad.values() if isinstance(d,dict) and d.get("severity")=="critical")
        high     = sum(1 for ad in result["metrics"].values()
                       for d in ad.values() if isinstance(d,dict) and d.get("severity")=="high")
        st.markdown(f"""
        <div style="padding:8px 0 16px;">
            <div style="font-size:20px;font-weight:700;color:#1A1A2E;margin-bottom:6px;">
                {DOMAIN_LABELS.get(result['domain'], result['domain'])} — Outcome Disparity Audit
            </div>
            <div style="font-size:13px;color:#666;">
                {n:,} records · {len(result['available_attrs'])} protected attributes audited
            </div>
        </div>
        """, unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("🔴 Critical Issues", critical)
        with c2: st.metric("🟡 High Issues", high)
        with c3: st.metric("Attributes", len(result["available_attrs"]))


# ── C7: BIAS FINGERPRINT RADAR ───────────────────────────────────────────────
def _render_radar_chart(result):
    st.markdown('<div class="fl-section">Bias Fingerprint</div>', unsafe_allow_html=True)

    attrs  = list(result["metrics"].keys())
    labels = [result["available_attrs"].get(a, {}).get("label", a) for a in attrs]

    # Worst DI per attribute (lower = more biased)
    scores = []
    for attr in attrs:
        groups = result["metrics"][attr]
        dis = [d["disparate_impact"] for d in groups.values()
               if isinstance(d, dict) and not d.get("is_reference")]
        scores.append(min(dis) if dis else 1.0)

    # Close the polygon
    labels_closed = labels + [labels[0]]
    scores_closed = scores + [scores[0]]
    perfect       = [1.0] * len(labels_closed)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=perfect, theta=labels_closed,
        fill=None, line=dict(color="#CCCCCC", dash="dash", width=1.5),
        name="Perfect Fairness"
    ))
    score_color = "#E24B4A" if result["fair_score"] < 45 else "#EF9F27" if result["fair_score"] < 75 else "#1D9E75"
    fill_rgba_str = f"rgba({int(score_color[1:3], 16)}, {int(score_color[3:5], 16)}, {int(score_color[5:7], 16)}, 0.2)"
    fig.add_trace(go.Scatterpolar(
        r=scores_closed, theta=labels_closed,
        fill="toself",
        fillcolor=fill_rgba_str,
        line=dict(color=score_color, width=2.5),
        name="Your Model",
        hovertemplate="<b>%{theta}</b><br>Disparate Impact: %{r:.3f}<extra></extra>"
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.1], tickvals=[0.25,0.5,0.75,1.0],
                            tickfont=dict(size=9), gridcolor="#EBEBEB"),
            angularaxis=dict(tickfont=dict(size=11))
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        height=320,
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("A perfect circle = zero bias. Dents show which attributes are most discriminatory.")


# ── C5: KILLER CANDIDATE COMPARISON ──────────────────────────────────────────
def _render_candidate_comparison(result):
    st.markdown('<div class="fl-section">Candidate Comparison — Same Merit, Different Outcome</div>',
                unsafe_allow_html=True)

    df      = result.get("df")
    verdict = result.get("verdict")
    if df is None or verdict is None:
        return

    attr       = verdict["attr"]
    ref_grp    = verdict["ref_group"]
    biased_grp = verdict["biased_group"]

    # Find skill/merit columns
    merit_cols = [c for c in df.columns
                  if any(k in c for k in ["skill","score","experience","marks","grade","gpa","credit"])
                  and c not in [result["outcome_col"],"_y","_pred","_proba"]]

    # Reference (accepted) candidate from favoured group
    out_col = result["outcome_col"]
    if result["available_attrs"].get(attr, {}).get("type") == "categorical":
        ref_mask    = (df[attr].astype(str) == ref_grp)    & (df["_pred"] == 1)
        biased_mask = (df[attr].astype(str) == biased_grp) & (df["_pred"] == 0)
    else:
        ref_mask    = (df["_pred"] == 1)
        biased_mask = (df["_pred"] == 0)

    ref_pool    = df[ref_mask]
    biased_pool = df[biased_mask]

    if len(ref_pool) == 0 or len(biased_pool) == 0:
        st.info("Not enough candidates to build comparison. Try a larger dataset.")
        return

    # Find the closest pair by merit columns
    if merit_cols:
        ref_med    = ref_pool[merit_cols].median()
        biased_med = biased_pool[merit_cols].median()
        # Pick ref candidate closest to biased median
        ref_dists    = ((ref_pool[merit_cols] - biased_med) ** 2).sum(axis=1)
        biased_dists = ((biased_pool[merit_cols] - ref_med) ** 2).sum(axis=1)
        cand_a = ref_pool.iloc[ref_dists.argmin()]
        cand_b = biased_pool.iloc[biased_dists.argmin()]
    else:
        cand_a = ref_pool.iloc[0]
        cand_b = biased_pool.iloc[0]

    display_cols = [c for c in df.columns
                    if c not in ["_y","_pred","_proba", out_col]
                    and not c.startswith("_")]

    col_a, col_sep, col_b = st.columns([5, 1, 5])

    def _fmt(val):
        if isinstance(val, float): return f"{val:.1f}"
        return str(val)

    with col_a:
        st.markdown("""
        <div style="background:#D1FAE5;border-radius:8px;padding:8px 14px;margin-bottom:10px;
                    font-weight:700;color:#065F46;font-size:14px;text-align:center;">
            ✅ SHORTLISTED — Candidate A
        </div>""", unsafe_allow_html=True)
        rows_a = ""
        for col in display_cols:
            val = _fmt(cand_a.get(col, "—"))
            highlight = " font-weight:600;color:#0F6E56;" if col == attr else ""
            rows_a += f'<tr><td style="color:#888;font-size:12px;padding:4px 8px;">{col}</td><td style="font-size:13px;{highlight}padding:4px 8px;">{val}</td></tr>'
        prob_a = cand_a.get("_proba", None)
        if prob_a is not None:
            rows_a += f'<tr><td style="color:#888;font-size:12px;padding:4px 8px;">AI Probability</td><td style="font-size:13px;color:#065F46;font-weight:600;padding:4px 8px;">{prob_a:.1%}</td></tr>'
        st.markdown(f'<table style="width:100%;border-collapse:collapse;">{rows_a}</table>', unsafe_allow_html=True)

    with col_sep:
        st.markdown('<div style="height:100%;border-left:1px dashed #ddd;margin:0 auto;width:1px;"></div>', unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div style="background:#FEE2E2;border-radius:8px;padding:8px 14px;margin-bottom:10px;
                    font-weight:700;color:#991B1B;font-size:14px;text-align:center;">
            ❌ REJECTED — Candidate B
        </div>""", unsafe_allow_html=True)
        rows_b = ""
        for col in display_cols:
            val = _fmt(cand_b.get(col, "—"))
            highlight = " font-weight:600;color:#991B1B;" if col == attr else ""
            rows_b += f'<tr><td style="color:#888;font-size:12px;padding:4px 8px;">{col}</td><td style="font-size:13px;{highlight}padding:4px 8px;">{val}</td></tr>'
        prob_b = cand_b.get("_proba", None)
        if prob_b is not None:
            rows_b += f'<tr><td style="color:#888;font-size:12px;padding:4px 8px;">AI Probability</td><td style="font-size:13px;color:#991B1B;font-weight:600;padding:4px 8px;">{prob_b:.1%}</td></tr>'
        st.markdown(f'<table style="width:100%;border-collapse:collapse;">{rows_b}</table>', unsafe_allow_html=True)

    attr_label = result["available_attrs"].get(attr, {}).get("label", attr)
    st.markdown(f"""
    <div style="background:#F7F9FB;border-radius:8px;padding:12px 16px;margin-top:12px;
                border-left:4px solid #EF9F27;font-size:13px;color:#444;line-height:1.6;">
        <b>What happened here?</b> Candidate B was rejected despite comparable merit scores.
        The key difference is <b>{attr_label}</b> — a protected attribute that
        should not influence hiring decisions. The AI system learned to use it as a signal
        from historical data, encoding past discrimination into future decisions.
    </div>
    """, unsafe_allow_html=True)


# ── METRIC CARDS ─────────────────────────────────────────────────────────────
def _render_metric_cards(result):
    st.markdown('<div class="fl-section">Fairness Metrics by Attribute</div>', unsafe_allow_html=True)
    for attr, groups in result["metrics"].items():
        label = result["available_attrs"].get(attr, {}).get("label", attr)
        with st.expander(f"**{label}**", expanded=True):
            n_cols = min(len(groups), 5)
            cols   = st.columns(n_cols)
            for i, (grp, data) in enumerate(groups.items()):
                if not isinstance(data, dict): continue
                sev   = data["severity"]
                color = SEVERITY_COLORS.get(sev, "#333")
                ref_tag = " ★" if data.get("is_reference") else ""
                sev_bg = {"critical":"#FEE2E2","high":"#FEF3C7","ok":"#D1FAE5"}.get(sev,"#F0F0F0")
                with cols[i % n_cols]:
                    st.markdown(f"""
                    <div style="background:#fff;border:1px solid #eee;border-radius:10px;
                                padding:14px;margin:4px 0;">
                        <div style="font-size:12px;font-weight:600;color:#555;margin-bottom:8px;">
                            {grp}{ref_tag}
                        </div>
                        <div style="font-size:11px;color:#888;">Selection rate</div>
                        <div style="font-size:22px;font-weight:700;color:{color};">
                            {data['pos_rate']:.1%}
                        </div>
                        <div style="margin-top:8px;font-size:10px;color:#999;">
                            Dem. Parity: <b>{data['demographic_parity']:.3f}</b><br>
                            Disp. Impact: <b>{data['disparate_impact']:.3f}</b><br>
                            Eq. Odds: <b>{data['equalized_odds']:.3f}</b>
                        </div>
                        <div style="margin-top:8px;">
                            <span style="background:{sev_bg};color:{color};border-radius:20px;
                                         padding:2px 8px;font-size:10px;font-weight:600;">
                                {sev.upper()}
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)


# ── GROUP BAR CHART ───────────────────────────────────────────────────────────
def _render_group_chart(result):
    st.markdown('<div class="fl-section">Selection Rate by Group</div>', unsafe_allow_html=True)
    rows = []
    for attr, groups in result["metrics"].items():
        label = result["available_attrs"].get(attr, {}).get("label", attr)
        for grp, data in groups.items():
            if isinstance(data, dict):
                rows.append({"Attribute": label, "Group": grp,
                             "Rate": data["pos_rate"], "Severity": data["severity"]})
    if not rows:
        st.info("No group data available.")
        return
    df_plot   = pd.DataFrame(rows)
    color_map = {"critical": "#E24B4A", "high": "#EF9F27", "ok": "#1D9E75"}
    fig = px.bar(df_plot, x="Group", y="Rate", color="Severity",
                 color_discrete_map=color_map, facet_col="Attribute",
                 facet_col_wrap=2, height=320,
                 labels={"Rate": "Selection Rate", "Group": ""})
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white",
                      font={"size":11}, margin=dict(l=10,r=10,t=30,b=10))
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── HEATMAP ───────────────────────────────────────────────────────────────────
def _render_heatmap(result):
    st.markdown('<div class="fl-section">Bias Heatmap (Metric × Attribute)</div>',
                unsafe_allow_html=True)
    metrics_list  = ["demographic_parity","disparate_impact","equalized_odds","accuracy_gap"]
    metric_labels = ["Dem. Parity","Disp. Impact","Eq. Odds","Acc. Gap"]
    attrs       = list(result["metrics"].keys())
    attr_labels = [result["available_attrs"].get(a,{}).get("label",a) for a in attrs]
    matrix = np.zeros((len(metrics_list), len(attrs)))
    text   = np.empty((len(metrics_list), len(attrs)), dtype=object)
    for j, attr in enumerate(attrs):
        groups = result["metrics"][attr]
        for mi, metric in enumerate(metrics_list):
            vals = [d[metric] for d in groups.values()
                    if isinstance(d,dict) and not d.get("is_reference")]
            if vals:
                worst = max(vals) if metric != "disparate_impact" else min(vals)
                matrix[mi,j] = worst
                text[mi,j]   = f"{worst:.3f}"
            else:
                text[mi,j] = "—"
    color_matrix = matrix.copy()
    color_matrix[1] = 1 - color_matrix[1]
    fig = go.Figure(go.Heatmap(
        z=color_matrix, x=attr_labels, y=metric_labels,
        text=text, texttemplate="%{text}",
        colorscale=[[0,"#D1FAE5"],[0.4,"#FEF3C7"],[1.0,"#FEE2E2"]],
        showscale=True,
        colorbar={"title":"Severity","thickness":12,"len":0.8},
        hovertemplate="<b>%{y}</b> / <b>%{x}</b><br>Value: %{text}<extra></extra>",
    ))
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white",
                      height=260, margin=dict(l=10,r=10,t=10,b=10), font={"size":11})
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── SHAP ─────────────────────────────────────────────────────────────────────
def _render_shap_section(result):
    st.markdown('<div class="fl-section">Feature Impact on Bias (SHAP)</div>',
                unsafe_allow_html=True)
    model = st.session_state.get("model")
    df    = result.get("df")
    if model is None or df is None:
        st.info("SHAP analysis requires a trained model.")
        return
    with st.spinner("Computing SHAP values…"):
        try:
            from core.explainer import compute_shap, get_feature_importance
            shap_vals, feat_names, X_sample = compute_shap(model, df, result["outcome_col"])
            importance = get_feature_importance(shap_vals, feat_names)
            st.session_state["shap_vals"]  = shap_vals
            st.session_state["feat_names"] = feat_names
            st.session_state["X_sample"]   = X_sample
            st.session_state["importance"] = importance
        except Exception as e:
            st.warning(f"SHAP computation failed: {e}")
            return
    top_n   = min(12, len(importance))
    plot_df = importance.head(top_n).sort_values("importance")
    fig = go.Figure(go.Bar(
        x=plot_df["importance"], y=plot_df["feature"],
        orientation="h",
        marker=dict(color=plot_df["importance"],
                    colorscale=[[0,"#E1F5EE"],[1,"#0F6E56"]], showscale=False),
        hovertemplate="<b>%{y}</b><br>Mean |SHAP|: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(height=max(280, top_n*30), paper_bgcolor="white",
                      plot_bgcolor="white", margin=dict(l=10,r=20,t=10,b=10),
                      xaxis_title="Mean |SHAP value|", font={"size":11})
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption("Features with high SHAP values drive the model's decisions most. "
               "If these correlate with protected attributes, they are proxy bias drivers.")


# ── CANDIDATE DRILL-DOWN ──────────────────────────────────────────────────────
def _render_candidate_drilldown(result):
    st.markdown('<div class="fl-section">Candidate Drill-Down</div>', unsafe_allow_html=True)
    df = result.get("df")
    if df is None: return
    shap_vals  = st.session_state.get("shap_vals")
    feat_names = st.session_state.get("feat_names")
    X_sample   = st.session_state.get("X_sample")
    display_cols = [c for c in df.columns if not c.startswith("_")]
    show_df = df[display_cols + ["_pred"]].rename(columns={"_pred": "Outcome"}).copy()
    show_df["Outcome"] = show_df["Outcome"].map({1: "✅ Positive", 0: "❌ Rejected"})
    st.dataframe(show_df.head(200), use_container_width=True, height=240)
    if shap_vals is not None:
        st.markdown("**Explain a specific candidate's AI score:**")
        candidate_idx = st.number_input("Row index (0-based)", min_value=0,
                                        max_value=min(len(X_sample)-1,199), value=0, step=1)
        if st.button("🔍 Explain this decision"):
            from core.explainer import get_candidate_shap
            cshap = get_candidate_shap(shap_vals, feat_names, X_sample, int(candidate_idx))
            fig = go.Figure(go.Bar(
                x=cshap["shap"].head(10), y=cshap["feature"].head(10),
                orientation="h",
                marker=dict(color=["#E24B4A" if v < 0 else "#1D9E75"
                                   for v in cshap["shap"].head(10)]),
                hovertemplate="<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
            ))
            fig.update_layout(height=280, paper_bgcolor="white", plot_bgcolor="white",
                              margin=dict(l=10,r=10,t=30,b=10),
                              title=f"Decision Explanation — Row {candidate_idx}",
                              xaxis_title="SHAP value (impact on outcome)",
                              font={"size":11})
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.caption("🟢 Green = pushed toward positive outcome. 🔴 Red = pushed toward rejection.")


# ── EMPTY STATE ───────────────────────────────────────────────────────────────
def _render_empty_state():
    st.markdown("""
    <div style="text-align:center;padding:80px 20px;">
        <div style="font-size:52px;">⚖️</div>
        <div style="font-size:24px;font-weight:700;color:#1A1A2E;margin:12px 0 8px;">
            Welcome to FairLens
        </div>
        <div style="font-size:15px;color:#666;max-width:480px;margin:0 auto;line-height:1.6;">
            Upload a CSV dataset, a trained model file, or describe your AI system
            in the sidebar to start your bias audit.
        </div>
        <div style="margin-top:32px;display:flex;justify-content:center;gap:24px;flex-wrap:wrap;">
            <div style="background:#fff;border:1px solid #eee;border-radius:12px;
                        padding:20px 24px;max-width:180px;">
                <div style="font-size:28px;">📂</div>
                <div style="font-weight:600;margin:8px 0 4px;font-size:14px;">CSV Dataset</div>
                <div style="font-size:12px;color:#888;">Upload your candidate or transaction data</div>
            </div>
            <div style="background:#fff;border:1px solid #eee;border-radius:12px;
                        padding:20px 24px;max-width:180px;">
                <div style="font-size:28px;">🤖</div>
                <div style="font-weight:600;margin:8px 0 4px;font-size:14px;">Model File</div>
                <div style="font-size:12px;color:#888;">Add .pkl or .joblib for deeper analysis</div>
            </div>
            <div style="background:#fff;border:1px solid #eee;border-radius:12px;
                        padding:20px 24px;max-width:180px;">
                <div style="font-size:28px;">💬</div>
                <div style="font-weight:600;margin:8px 0 4px;font-size:14px;">Text Description</div>
                <div style="font-size:12px;color:#888;">Describe your AI system in plain language</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
