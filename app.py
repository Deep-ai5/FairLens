import streamlit as st
st.set_page_config(
    page_title="FairLens — AI Bias Auditor",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from ui.sidebar    import render_sidebar
from ui.dashboard  import render_dashboard
from ui.chat       import render_chat
from ui.report     import render_report_page
from ui.simulation import render_simulation
from ui.live_audit import render_live_audit

# ── Session state defaults ────────────────────────────────────────────────────
DEFAULTS = {
    "audit_result": None,
    "domain":       None,
    "df":           None,
    "model":        None,
    "chat_history": [],
    "page":         "dashboard",
    "sim_result":   None,
    "live_result":  None,
    "shap_vals":    None,
    "feat_names":   None,
    "X_sample":     None,
    "importance":   None,
    "fix_suggestions": None,
    "load_demo":    None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background: #F7F9FB; }
    [data-testid="stSidebar"] { background: #FFFFFF; border-right: 1px solid #EBEBEB; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .fl-section {
        font-size: 12px; font-weight: 600; color: #888;
        letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 12px;
    }
    #MainMenu, footer, header { visibility: hidden; }
    .stButton > button {
        border-radius: 8px;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# ── Router ────────────────────────────────────────────────────────────────────
page = render_sidebar()

if   page == "dashboard":  render_dashboard()
elif page == "simulation":  render_simulation()
elif page == "chat":        render_chat()
elif page == "live_audit":  render_live_audit()
elif page == "report":      render_report_page()
