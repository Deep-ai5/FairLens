"""
report_generator.py
Generates a consultant-grade PDF audit report using ReportLab.
"""

import io
import datetime
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


# ── Brand colours ─────────────────────────────────────────────────────────────
TEAL        = colors.HexColor("#0F6E56")
TEAL_LIGHT  = colors.HexColor("#E1F5EE")
DARK        = colors.HexColor("#1A1A2E")
GRAY_BG     = colors.HexColor("#F7F9FB")
GRAY_BORDER = colors.HexColor("#E0E0E0")
RED         = colors.HexColor("#A32D2D")
AMBER       = colors.HexColor("#854F0B")
GREEN       = colors.HexColor("#065F46")
WHITE       = colors.white
BLACK       = colors.HexColor("#1A1A1A")


def severity_color(sev):
    return {
        "critical": RED,
        "high":     AMBER,
        "ok":       GREEN,
    }.get(sev, BLACK)


def grade_color(grade):
    return {
        "A": GREEN, "B": GREEN,
        "C": AMBER,
        "D": RED, "F": RED,
    }.get(grade, BLACK)


def generate_pdf(audit_result, fix_suggestions=None):
    """
    Returns a BytesIO PDF buffer ready for st.download_button.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        leftMargin=0.85*inch, rightMargin=0.85*inch,
        topMargin=0.85*inch,  bottomMargin=0.85*inch,
    )

    styles = getSampleStyleSheet()
    story  = []

    # ── Helpers ───────────────────────────────────────────────────────────────
    def H1(text):
        return Paragraph(text, ParagraphStyle(
            "h1", fontSize=22, fontName="Helvetica-Bold",
            textColor=TEAL, spaceAfter=6, spaceBefore=18
        ))

    def H2(text):
        return Paragraph(text, ParagraphStyle(
            "h2", fontSize=14, fontName="Helvetica-Bold",
            textColor=DARK, spaceAfter=4, spaceBefore=14
        ))

    def H3(text):
        return Paragraph(text, ParagraphStyle(
            "h3", fontSize=11, fontName="Helvetica-Bold",
            textColor=TEAL, spaceAfter=3, spaceBefore=10
        ))

    def Body(text):
        return Paragraph(text, ParagraphStyle(
            "body", fontSize=10, fontName="Helvetica",
            textColor=BLACK, spaceAfter=4, leading=14
        ))

    def rule():
        return HRFlowable(width="100%", thickness=0.5, color=GRAY_BORDER, spaceAfter=8, spaceBefore=8)

    domain_label = {"hr": "HR / Hiring", "banking": "Banking / Loan Approvals", "education": "Education / Admissions"}.get(audit_result["domain"], audit_result["domain"].upper())
    now = datetime.datetime.now().strftime("%B %d, %Y")

    # ── COVER ─────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("⚖ FairLens", ParagraphStyle(
        "brand", fontSize=36, fontName="Helvetica-Bold", textColor=TEAL, spaceAfter=4
    )))
    story.append(Paragraph("AI Bias Audit Report", ParagraphStyle(
        "subtitle", fontSize=18, fontName="Helvetica", textColor=DARK, spaceAfter=2
    )))
    story.append(Paragraph(f"{domain_label}  ·  {now}", ParagraphStyle(
        "meta", fontSize=11, fontName="Helvetica", textColor=colors.HexColor("#999999"), spaceAfter=16
    )))
    story.append(rule())

    # ── EXECUTIVE SUMMARY ─────────────────────────────────────────────────────
    story.append(H1("Executive Summary"))

    score = audit_result["fair_score"]
    grade = audit_result["grade"]

    summary_data = [
        ["Metric",         "Value"],
        ["FairScore",      f"{score} / 100  (Grade {grade})"],
        ["Domain",         domain_label],
        ["Dataset Size",   f"{audit_result['n_samples']:,} records"],
        ["Audit Date",     now],
        ["Attributes Audited", Paragraph(", ".join(audit_result["available_attrs"].keys()), ParagraphStyle("summary_cell", fontSize=10, fontName="Helvetica", textColor=BLACK, leading=14))],
    ]
    t = Table(summary_data, colWidths=[2.5*inch, 4.5*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,0), DARK),
        ("TEXTCOLOR",    (0,0), (-1,0), WHITE),
        ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 10),
        ("BACKGROUND",   (0,1), (-1,-1), GRAY_BG),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, GRAY_BG]),
        ("GRID",         (0,0), (-1,-1), 0.5, GRAY_BORDER),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING",   (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
        ("TEXTCOLOR",    (1,1), (1,1), grade_color(grade)),
        ("FONTNAME",     (1,1), (1,1), "Helvetica-Bold"),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # Overall verdict
    if score >= 75:
        verdict = "The audited system shows <b>acceptable fairness</b> overall. Minor improvements are recommended."
    elif score >= 60:
        verdict = "The audited system shows <b>moderate bias</b>. Targeted remediation is recommended before production use."
    elif score >= 45:
        verdict = "The audited system shows <b>significant bias</b>. Deployment should be paused pending remediation."
    else:
        verdict = "The audited system shows <b>critical bias</b> that likely causes unlawful discrimination. Immediate action required."
    story.append(Body(verdict))
    story.append(rule())

    # ── METRIC RESULTS ────────────────────────────────────────────────────────
    story.append(H1("Fairness Metric Results"))
    story.append(Body(
        "The following tables show fairness metrics computed for each protected attribute. "
        "The 80% Rule (EEOC standard) flags any group whose selection rate falls below 80% "
        "of the most-favoured group's rate."
    ))
    story.append(Spacer(1, 8))

    for attr, groups in audit_result["metrics"].items():
        label = audit_result["available_attrs"].get(attr, {}).get("label", attr)
        story.append(H3(f"Attribute: {label}"))

        header = ["Group", "N", "Select Rate", "Dem. Parity", "Disp. Impact", "Eq. Odds", "Acc. Gap", "Severity"]
        rows   = [header]
        for grp, data in groups.items():
            if not isinstance(data, dict):
                continue
            rows.append([
                f"{grp} {'★' if data.get('is_reference') else ''}".strip(),
                str(data["n"]),
                f"{data['pos_rate']:.1%}",
                f"{data['demographic_parity']:.3f}",
                f"{data['disparate_impact']:.3f}",
                f"{data['equalized_odds']:.3f}",
                f"{data['accuracy_gap']:.3f}",
                data["severity"].upper(),
            ])

        col_widths = [1.3*inch, 0.5*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.8*inch, 0.8*inch, 0.85*inch]
        t = Table(rows, colWidths=col_widths)

        sev_styles = []
        for i, row in enumerate(rows[1:], start=1):
            sev = row[-1].lower()
            c = severity_color(sev)
            sev_styles.append(("TEXTCOLOR", (7, i), (7, i), c))
            sev_styles.append(("FONTNAME",  (7, i), (7, i), "Helvetica-Bold"))

        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), TEAL),
            ("TEXTCOLOR",     (0,0), (-1,0), WHITE),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 8.5),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, GRAY_BG]),
            ("GRID",          (0,0), (-1,-1), 0.4, GRAY_BORDER),
            ("LEFTPADDING",   (0,0), (-1,-1), 5),
            ("RIGHTPADDING",  (0,0), (-1,-1), 5),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            *sev_styles
        ]))
        story.append(KeepTogether([t, Spacer(1, 10)]))

    story.append(rule())

    # ── FIX SUGGESTIONS ───────────────────────────────────────────────────────
    if fix_suggestions:
        story.append(H1("Recommended Remediation Steps"))
        story.append(Body(
            "The following fixes are ranked by estimated impact. Implement in order for maximum fairness improvement."
        ))
        story.append(Spacer(1, 8))

        cell_style = ParagraphStyle("cell", fontSize=8.5, fontName="Helvetica", textColor=BLACK, leading=11)
        fix_data = [["#", "Action", "Reason", "Est. Improvement"]]
        for f in fix_suggestions:
            fix_data.append([
                str(f.get("rank", "")),
                Paragraph(str(f.get("fix", "")), cell_style),
                Paragraph(str(f.get("reason", "")), cell_style),
                Paragraph(str(f.get("estimated_improvement", "")), cell_style),
            ])

        ft = Table(fix_data, colWidths=[0.3*inch, 2.2*inch, 2.5*inch, 2.0*inch])
        ft.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), DARK),
            ("TEXTCOLOR",     (0,0), (-1,0), WHITE),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,-1), 8.5),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [WHITE, GRAY_BG]),
            ("GRID",          (0,0), (-1,-1), 0.4, GRAY_BORDER),
            ("VALIGN",        (0,0), (-1,-1), "TOP"),
            ("LEFTPADDING",   (0,0), (-1,-1), 5),
            ("RIGHTPADDING",  (0,0), (-1,-1), 5),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ]))
        story.append(ft)
        story.append(rule())

    # ── FOOTER ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(
        "Generated by FairLens — AI Bias Auditor  ·  Team Coderz  ·  Build with AI 2026",
        ParagraphStyle("footer", fontSize=8, textColor=colors.HexColor("#AAAAAA"), alignment=TA_CENTER)
    ))

    doc.build(story)
    buf.seek(0)
    return buf
