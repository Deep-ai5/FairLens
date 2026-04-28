"""
domain_detector.py
Detects domain from column names and defines protected attributes per domain.
"""

DOMAIN_SIGNALS = {
    "hr": [
        "gender", "sex", "age", "college", "university", "gpa", "experience",
        "department", "role", "salary", "shortlisted", "hired", "selected",
        "college_tier", "graduation_year", "region", "skills", "resume"
    ],
    "banking": [
        "loan", "credit", "income", "debt", "approved", "default", "balance",
        "account", "asset", "emi", "cibil", "score", "rural", "urban",
        "employment_type", "collateral", "interest", "repayment"
    ],
    "education": [
        "school", "marks", "grade", "admission", "board", "medium", "stream",
        "percentage", "entrance", "scholarship", "admitted", "institution",
        "attendance", "extracurricular", "rank", "merit"
    ],
}

PROTECTED_ATTRS = {
    "hr": {
        "gender":          {"label": "Gender",           "type": "categorical"},
        "age":             {"label": "Age",              "type": "numeric", "bins": [0, 25, 35, 45, 100], "labels": ["<25", "25–35", "35–45", "45+"]},
        "college_tier":    {"label": "College Tier",     "type": "categorical"},
        "region":          {"label": "Region",           "type": "categorical"},
        "graduation_year": {"label": "Graduation Year",  "type": "numeric", "bins": [0, 2015, 2020, 2023, 2030], "labels": ["Pre-2015", "2015–20", "2020–23", "2023+"]},
    },
    "banking": {
        "gender":          {"label": "Gender",           "type": "categorical"},
        "age":             {"label": "Age",              "type": "numeric", "bins": [0, 25, 35, 50, 100], "labels": ["<25", "25–35", "35–50", "50+"]},
        "income_bracket":  {"label": "Income Bracket",  "type": "categorical"},
        "area_type":       {"label": "Area (Rural/Urban)", "type": "categorical"},
        "employment_type": {"label": "Employment Type", "type": "categorical"},
    },
    "education": {
        "gender":          {"label": "Gender",           "type": "categorical"},
        "school_type":     {"label": "School Type",      "type": "categorical"},
        "region":          {"label": "Region",           "type": "categorical"},
        "medium":          {"label": "Medium of Instruction", "type": "categorical"},
        "family_income":   {"label": "Family Income",   "type": "numeric", "bins": [0, 200000, 600000, 1200000, 99999999], "labels": ["<2L", "2–6L", "6–12L", "12L+"]},
    },
}


def detect_domain(columns):
    """
    Score each domain by how many signal words appear in the column list.
    Returns the domain with the highest score. Defaults to 'hr'.
    """
    col_set = set(c.lower() for c in columns)
    scores = {}
    for domain, signals in DOMAIN_SIGNALS.items():
        scores[domain] = sum(1 for s in signals if any(s in col for col in col_set))

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "hr"


def get_protected_attrs(domain):
    """
    Returns the protected attribute config dict for the given domain.
    Only returns attrs that are actually available — caller filters by df columns.
    """
    return PROTECTED_ATTRS.get(domain, PROTECTED_ATTRS["hr"])


def filter_available_attrs(protected_attrs, df_columns):
    """
    Filter protected attrs to only those present in the dataframe.
    """
    col_set = set(c.lower() for c in df_columns)
    return {
        attr: meta
        for attr, meta in protected_attrs.items()
        if attr in col_set
    }
