"""
chat.py — Gemini-powered audit chat interface
"""

import streamlit as st
from core.gemini_client import chat, get_fix_suggestions


STARTER_QUESTIONS = [
    "Why is my model biased?",
    "Which protected group is most disadvantaged?",
    "What should I fix first?",
    "Is this level of bias legally risky in India?",
    "Generate a fix plan for my engineering team.",
]


def render_chat():
    st.markdown("## 💬 Ask FairLens")
    st.markdown("Ask anything about your audit results. FairLens knows your dataset, metrics, and flagged features.")

    result = st.session_state.get("audit_result")
    if not result:
        st.info("Run an audit first from the sidebar.")
        return

    # Fix suggestions panel
    with st.expander("⚡ Auto-generated Fix Suggestions", expanded=False):
        if st.button("Generate Fix Suggestions with Gemini"):
            importance = st.session_state.get("importance")
            with st.spinner("Asking Gemini for fix recommendations…"):
                fixes = get_fix_suggestions(result, shap_importance=importance)
                st.session_state["fix_suggestions"] = fixes

        fixes = st.session_state.get("fix_suggestions")
        if fixes:
            for f in fixes:
                col_rank, col_body = st.columns([1, 11])
                with col_rank:
                    st.markdown(f'<div style="font-size:20px;font-weight:700;color:#0F6E56;text-align:center;padding-top:6px;">#{f["rank"]}</div>', unsafe_allow_html=True)
                with col_body:
                    st.markdown(f'<div style="background:#fff;border:1px solid #eee;border-radius:8px;padding:12px 14px;margin-bottom:8px;"><div style="font-weight:600;font-size:14px;color:#111;margin-bottom:4px;">{f["fix"]}</div><div style="font-size:12px;color:#555;margin-bottom:4px;">{f["reason"]}</div><div style="font-size:11px;color:#0F6E56;font-weight:600;">Est. improvement: {f["estimated_improvement"]}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Starter prompts
    st.markdown("**Quick questions:**")
    cols = st.columns(len(STARTER_QUESTIONS))
    for i, q in enumerate(STARTER_QUESTIONS):
        with cols[i]:
            if st.button(q, key=f"starter_{i}", use_container_width=True):
                st.session_state["pending_message"] = q

    st.markdown("---")

    # Chat history
    history = st.session_state.get("chat_history", [])
    chat_container = st.container()
    with chat_container:
        for turn in history:
            if turn["role"] == "user":
                st.markdown(f'<div class="chat-user">🧑 {turn["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-ai">⚖️ {turn["content"]}</div>', unsafe_allow_html=True)

    # Input
    pending = st.session_state.pop("pending_message", None)
    user_input = st.chat_input("Ask about your audit…") or pending

    if user_input:
        history.append({"role": "user", "content": user_input})
        with st.spinner("FairLens is thinking…"):
            try:
                reply = chat(user_input, history[:-1], result)
            except Exception as e:
                reply = f"⚠️ Gemini API error: {e}. Check your API key in secrets.toml."
        history.append({"role": "assistant", "content": reply})
        st.session_state["chat_history"] = history
        st.rerun()
