"""
demo_loader.py — Synthetic demo datasets.
v2: Tuned for realistic bias — Grade D/F scores, human-interpretable multipliers.
Target rates:
  HR:         Male ~65%, Female ~40%, Rural ~37%  → ~1.6x gender gap
  Banking:    Male ~44%, Female ~30%, Rural ~23%  → ~1.5x gender gap
  Education:  Private ~55%, Government ~20%       → ~2.5x school gap
"""

import io
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def get_demo_csv(domain: str) -> io.BytesIO:
    builders = {"hr": _make_hr_dataset, "banking": _make_banking_dataset, "education": _make_education_dataset}
    df  = builders.get(domain, _make_hr_dataset)()
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    buf.name = f"demo_{domain}.csv"
    return buf


def _make_hr_dataset(n=1200):
    gender        = RNG.choice(["Male", "Female"], n, p=[0.57, 0.43])   # removed Other (too small, noisy)
    age           = RNG.integers(22, 55, n)
    college_tier  = RNG.choice(["Tier 1", "Tier 2", "Tier 3"], n, p=[0.20, 0.45, 0.35])
    region        = RNG.choice(["Metro", "Semi-Urban", "Rural"],  n, p=[0.45, 0.35, 0.20])
    experience    = np.clip(RNG.normal(5, 3, n).astype(int), 0, 25)
    skills_score  = RNG.integers(40, 100, n)
    graduation_year = RNG.integers(2010, 2024, n)

    logit = (
        0.020 * skills_score
        + 0.040 * experience
        - 1.000 * (gender == "Female").astype(float)      # ~1.6x gender gap
        - 0.900 * (region == "Rural").astype(float)       # strong region bias
        - 0.400 * (region == "Semi-Urban").astype(float)
        - 0.800 * (college_tier == "Tier 3").astype(float)
        - 0.250 * (college_tier == "Tier 2").astype(float)
        - 0.500 * ((gender == "Female") & (region == "Rural")).astype(float)  # compound
        - 0.200                                            # intercept → overall ~55%
    )
    prob        = 1 / (1 + np.exp(-logit))
    shortlisted = RNG.binomial(1, prob)

    return pd.DataFrame({
        "gender":          gender,
        "age":             age,
        "college_tier":    college_tier,
        "region":          region,
        "experience_yrs":  experience,
        "skills_score":    skills_score,
        "graduation_year": graduation_year,
        "shortlisted":     shortlisted,
    })


def _make_banking_dataset(n=1000):
    gender          = RNG.choice(["Male", "Female"],                     n, p=[0.58, 0.42])
    age             = RNG.integers(21, 65, n)
    income_bracket  = RNG.choice(["Low", "Middle", "High"],              n, p=[0.40, 0.40, 0.20])
    area_type       = RNG.choice(["Urban", "Semi-Urban", "Rural"],       n, p=[0.50, 0.30, 0.20])
    employment_type = RNG.choice(["Salaried", "Self-Employed", "Unemployed"], n, p=[0.55, 0.30, 0.15])
    loan_amount     = RNG.integers(50000, 2000000, n)
    credit_score    = np.clip(RNG.normal(680, 80, n).astype(int), 300, 900)

    logit = (
        0.005 * credit_score
        - 0.500 * (income_bracket  == "Low").astype(float)
        - 0.700 * (area_type       == "Rural").astype(float)
        - 0.300 * (area_type       == "Semi-Urban").astype(float)
        - 0.400 * (employment_type == "Self-Employed").astype(float)
        - 0.800 * (employment_type == "Unemployed").astype(float)
        - 0.600 * (gender          == "Female").astype(float)
        - 3.000                                            # intercept → male~44%, female~30%
    )
    prob     = 1 / (1 + np.exp(-logit))
    approved = RNG.binomial(1, prob)

    return pd.DataFrame({
        "gender":          gender,
        "age":             age,
        "income_bracket":  income_bracket,
        "area_type":       area_type,
        "employment_type": employment_type,
        "loan_amount":     loan_amount,
        "credit_score":    credit_score,
        "approved":        approved,
    })


def _make_education_dataset(n=900):
    gender          = RNG.choice(["Male", "Female"],                           n, p=[0.52, 0.48])
    school_type     = RNG.choice(["Private", "Government", "Semi-Government"], n, p=[0.35, 0.45, 0.20])
    region          = RNG.choice(["Metro", "Tier-2 City", "Rural"],            n, p=[0.40, 0.35, 0.25])
    medium          = RNG.choice(["English", "Regional Language"],             n, p=[0.55, 0.45])
    marks_pct       = np.clip(RNG.normal(72, 12, n), 40, 100)
    extracurricular = RNG.integers(0, 10, n)

    logit = (
        0.060 * marks_pct
        + 0.100 * extracurricular
        - 1.200 * (school_type == "Government").astype(float)     # ~2.5x school gap
        - 0.600 * (school_type == "Semi-Government").astype(float)
        - 0.800 * (medium      == "Regional Language").astype(float)
        - 0.500 * (region      == "Rural").astype(float)
        - 0.250 * (gender      == "Female").astype(float)
        - 5.200                                                     # intercept
    )
    prob     = 1 / (1 + np.exp(-logit))
    admitted = RNG.binomial(1, prob)

    return pd.DataFrame({
        "gender":          gender,
        "school_type":     school_type,
        "region":          region,
        "medium":          medium,
        "marks_pct":       marks_pct.round(1),
        "extracurricular": extracurricular,
        "admitted":        admitted,
    })
