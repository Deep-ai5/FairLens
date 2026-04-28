"""
report.py — PDF report generation page
"""

import streamlit as st
from utils.report_generator import generate_pdf


def render_report_page():
    st.markdown("## 📄 PDF Audit Report")
    st.markdown("Generate a downloadable consultant-grade bias audit report from your current session.")

    result = st.session_state.get("audit_result")
    if not result:
        st.info("Run an audit first from the sidebar.")
        return

    fixes = st.session_state.get("fix_suggestions")
    score = result["fair_score"]
    grade = result["grade"]
    n     = result["n_samples"]
    domain_labels = {"hr": "HR / Hiring", "banking": "Banking / Loans", "education": "Education / Admissions"}

    # Preview card
    st.markdown(f"""
    <div style="background:#fff;border:1px solid #eee;border-radius:12px;padding:24px 28px;margin-bottom:20px;">
        <div style="font-size:16px;font-weight:700;color:#111;margin-bottom:12px;">Report Preview</div>
        <div style="display:flex;gap:40px;flex-wrap:wrap;">
            <div><div style="font-size:11px;color:#888;">Domain</div><div style="font-weight:600;font-size:14px;">{domain_labels.get(result['domain'], result['domain'])}</div></div>
            <div><div style="font-size:11px;color:#888;">FairScore</div><div style="font-weight:700;font-size:22px;color:#0F6E56;">{score} / 100</div></div>
            <div><div style="font-size:11px;color:#888;">Grade</div><div style="font-weight:700;font-size:22px;">{grade}</div></div>
            <div><div style="font-size:11px;color:#888;">Records</div><div style="font-weight:600;font-size:14px;">{n:,}</div></div>
            <div><div style="font-size:11px;color:#888;">Fix suggestions</div><div style="font-weight:600;font-size:14px;">{"Included" if fixes else "Not generated yet"}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Report includes:**")
    checklist = [
        "Executive summary with FairScore and grade",
        "Full fairness metric tables per protected attribute",
        "Severity flagging (Critical / High / OK)",
        "Fix recommendations" + (" (AI-generated)" if fixes else " — generate in the Chat tab first"),
        "FairLens branding + date",
    ]
    for item in checklist:
        st.markdown(f"✅ {item}")

    st.markdown("---")

    if not fixes:
        st.info("Tip: Go to Ask FairLens → Generate Fix Suggestions to include AI-powered remediation steps in your report.")

    if st.button("🖨️ Generate PDF Report", type="primary"):
        with st.spinner("Building report…"):
            try:
                pdf_buf = generate_pdf(result, fix_suggestions=fixes)
                st.download_button(
                    label="⬇️ Download FairLens Report.pdf",
                    data=pdf_buf,
                    file_name="fairlens_audit_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
                st.success("Report ready! Click the button above to download.")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
