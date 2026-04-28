# ⚖️ FairLens — AI Bias Auditor

**FairLens** is an AI-powered bias auditor for high-stakes decision systems. Built for HR hiring, banking loan approvals, and educational admissions — it detects, explains, and helps fix algorithmic discrimination.

---

## Features

- **Flexible input** — CSV dataset, trained model (.pkl/.joblib), or plain-text description
- **Auto domain detection** — HR / Banking / Education detected from column names
- **4 fairness metrics** — Demographic Parity, Equalized Odds, Disparate Impact, Accuracy Gap
- **FairScore** — 0–100 composite fairness score with letter grade
- **SHAP explainability** — feature-level attribution showing bias drivers
- **Candidate drill-down** — explain any individual decision
- **Before vs. After simulation** — drop/reweight features, see instant metric changes
- **Gemini-powered chat** — ask questions about your audit in plain language
- **PDF audit report** — one-click downloadable consultant-grade report

---

## Quickstart

```bash
git clone https://github.com/teamcoderz/fairlens
cd fairlens
pip install -r requirements.txt
```

Add your Gemini API key to `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-key-here"
```

Run:
```bash
streamlit run app.py
```

---

## Project Structure

```
fairlens/
├── app.py                  # Streamlit entry point
├── requirements.txt
├── .streamlit/
│   └── secrets.toml        # API keys (never commit this)
├── core/
│   ├── ingestor.py         # Input handling (CSV / model / text)
│   ├── domain_detector.py  # Auto-detect domain + protected attrs
│   ├── engine.py           # Fairness metric computation + FairScore
│   ├── explainer.py        # SHAP explainability
│   └── gemini_client.py    # Gemini API (chat + fixes + text→dataset)
├── ui/
│   ├── sidebar.py          # Nav + file upload
│   ├── dashboard.py        # Main audit dashboard
│   ├── simulation.py       # Before vs. After simulation
│   ├── chat.py             # Gemini chat interface
│   └── report.py           # PDF report page
├── utils/
│   └── report_generator.py # ReportLab PDF builder
└── data/
    └── demo_loader.py      # Synthetic demo datasets
```

---

## Deployment (Streamlit Cloud)

1. Push this repo to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as entry point
4. Add `GEMINI_API_KEY` in Settings → Secrets
5. Deploy

---

## Team

**Team Deep** — Build with AI, Solution Challenge 2026
Deepchandra Maurya
