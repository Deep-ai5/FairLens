"""
simulation.py — Before vs. After bias simulation panel
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from core.engine import run_simulation


def render_simulation():
    st.markdown("## 🔬 Bias Simulation")
    st.markdown("Drop or reweight features to see how fairness metrics change — without retraining from scratch.")

    result = st.session_state.get("audit_result")
    if not result:
        st.info("Run an audit first from the sidebar.")
        return

    df             = result["df"]
    outcome_col    = result["outcome_col"]
    domain         = result["domain"]
    protected      = result["available_attrs"]
    model          = st.session_state.get("model")

    feature_cols = [c for c in df.columns if c not in [outcome_col, "_y", "_pred", "_proba"] and not c.startswith("_")]

    st.markdown("### Configure changes")

    col_drop, col_weight = st.columns(2, gap="large")
    with col_drop:
        st.markdown("**Drop features entirely:**")
        drop_cols = st.multiselect(
            "Select features to remove from the model",
            options=feature_cols,
            help="Removing protected attributes or their proxies is the most common first fix.",
            label_visibility="collapsed"
        )

    with col_weight:
        st.markdown("**Reweight a feature (0 = remove, 2 = double influence):**")
        reweight_col = st.selectbox("Feature to reweight", ["(none)"] + feature_cols, label_visibility="collapsed")
        reweight_val = st.slider("Multiplier", 0.0, 2.0, 1.0, 0.1)
        reweight = {}
        if reweight_col != "(none)":
            reweight[reweight_col] = reweight_val

    if st.button("▶️ Run Simulation", type="primary"):
        with st.spinner("Simulating…"):
            sim = run_simulation(df, model, outcome_col, domain, protected, drop_cols=drop_cols, reweight=reweight or None)
            st.session_state["sim_result"] = sim

    # ── Results ───────────────────────────────────────────────────────────────
    sim_result = st.session_state.get("sim_result")
    if sim_result is None:
        st.markdown("---")
        st.caption("Configure changes above and click Run Simulation to see the impact.")
        return

    st.markdown("---")
    st.markdown("### Before vs. After")

    before_score = result["fair_score"]
    after_score  = sim_result["fair_score"]
    delta        = after_score - before_score
    delta_str    = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
    delta_color  = "#1D9E75" if delta >= 0 else "#E24B4A"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Before FairScore", f"{before_score}", "")
    with c2:
        st.metric("After FairScore", f"{after_score}", delta_str)
    with c3:
        before_g = result["grade"]
        after_g  = sim_result["grade"]
        st.metric("Grade Change", after_g, f"was {before_g}")

    # Metric comparison chart per attribute
    attrs = list(result["metrics"].keys())
    metric_keys   = ["demographic_parity", "disparate_impact", "equalized_odds", "accuracy_gap"]
    metric_labels = ["Dem. Parity", "Disp. Impact", "Eq. Odds", "Acc. Gap"]

    for attr in attrs:
        if attr not in result["metrics"] or attr not in sim_result["metrics"]:
            continue

        label = result["available_attrs"].get(attr, {}).get("label", attr)
        st.markdown(f"#### {label}")

        before_groups = result["metrics"][attr]
        after_groups  = sim_result["metrics"].get(attr, {})

        grp_names = [g for g in before_groups if isinstance(before_groups[g], dict) and not before_groups[g].get("is_reference")]

        if not grp_names:
            continue

        rows = []
        for grp in grp_names:
            bd = before_groups.get(grp, {})
            ad = after_groups.get(grp, {})
            if not isinstance(bd, dict):
                continue
            rows.append({
                "Group":  grp,
                "Before — Select Rate": bd.get("pos_rate", 0),
                "After  — Select Rate": ad.get("pos_rate", bd.get("pos_rate", 0)),
                "Before — Dem. Parity": bd.get("demographic_parity", 0),
                "After  — Dem. Parity": ad.get("demographic_parity", bd.get("demographic_parity", 0)),
            })

        if not rows:
            continue

        df_cmp = pd.DataFrame(rows)
        fig = go.Figure()
        fig.add_bar(name="Before", x=df_cmp["Group"], y=df_cmp["Before — Select Rate"], marker_color="#CBD5E1")
        fig.add_bar(name="After",  x=df_cmp["Group"], y=df_cmp["After  — Select Rate"], marker_color="#1D9E75")
        fig.update_layout(
            barmode="group", height=240,
            paper_bgcolor="white", plot_bgcolor="white",
            margin=dict(l=10, r=10, t=20, b=10),
            yaxis_tickformat=".0%", font={"size": 11},
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
