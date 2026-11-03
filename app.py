"""
============================================================
                     RA-PREDICT
   Machine-learning-based Rheumatoid Arthritis Prediction Tool
============================================================

XGBoost Classifier

Run:
    streamlit run app.py

Compatible with:
✓ Streamlit
✓ GitHub
✓ Windows
✓ Ubuntu
✓ PyInstaller

============================================================
"""

import os
import sys
from pathlib import Path

import base64
import html
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

try:
    import shap

    SHAP_AVAILABLE = True
except Exception:  # shap not installed
    SHAP_AVAILABLE = False

try:
    from report_extraction import (
        FIELD_LABELS,
        FILE_TYPES,
        ocr_available,
        read_reports,
        summarise_for_form,
    )

    REPORT_READING = True
except Exception:  # module missing: the report-upload card is hidden
    REPORT_READING = False


# ============================================================
# RESOURCE PATH
# ============================================================

def resource_path(relative_path):
    """
    Returns correct path for development and PyInstaller.
    """
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path(resource_path("model/RA_Predict_XGB.json"))
BACKGROUND_PATH = Path(resource_path("model/background.csv"))
LOGO_PATH = resource_path("assets/logo_hero.png")
TEAM_DIR = Path(resource_path("assets/team"))


# ============================================================
# TEAM / AUTHORS  (as listed in the manuscript)
# "photo" is a file name inside assets/team/ (optional).
# "affiliation" may contain <br> for line breaks.
# ============================================================

AUTHORS = [
    {
        "name": "Ms. Maha Yousaf",
        "affiliation": (
            "Department of Precision Medicine<br>"
            "Sungkyunkwan University School of Medicine<br>"
            "Suwon, Republic of Korea"
        ),
        "email": "maha.yousaf@g.skku.edu",
        "photo": "female_icon.png",
    },
    {
        "name": "Ms. Dur E Nishwa",
        "affiliation": (
            "Department of Precision Medicine<br>"
            "Sungkyunkwan University School of Medicine<br>"
            "Suwon, Republic of Korea"
        ),
        "email": "dure512@skku.edu",
        "photo": "female_icon.png",
    },
    {
        "name": "Ms. Saadia Iftikhar",
        "affiliation": (
            "Genomics Research Lab<br>"
            "Department of Biological Sciences<br>"
            "International Islamic University Islamabad, Pakistan"
        ),
        "email": "ifti.saadia@gmail.com",
        "photo": "female_icon.png",
    },
    {
        "name": "Dr. Shaheen Shahzad",
        "affiliation": (
            "Genomics Research Lab<br>"
            "Department of Biological Sciences<br>"
            "International Islamic University Islamabad, Pakistan"
        ),
        "email": "drshaheen@iiu.edu.pk",
        "photo": "female_icon.png",
    },
    {
        "name": "Dr. Muhammad Tahir ul Qamar",
        "affiliation": (
            "Integrative Omics and Molecular Modeling Laboratory<br>"
            "Department of Bioinformatics and Biotechnology<br>"
            "Government College University Faisalabad (GCUF), Pakistan"
        ),
        "email": "tahirulqamar@gcuf.edu.pk",
        "photo": "male_icon.png",
    },
    {
        "name": "Mr. Tamim Alkhalifah",
        "affiliation": (
            "Department of Computer Engineering, College of Computer<br>"
            "Qassim University<br>"
            "Buraydah, Saudi Arabia"
        ),
        "email": "tkhliefh@qu.edu.sa",
        "photo": "male_icon.png",
    },
    {
        "name": "Dr. Seung Won Lee",
        "affiliation": (
            "Department of Precision Medicine<br>"
            "Sungkyunkwan University School of Medicine<br>"
            "Suwon, Republic of Korea"
        ),
        "email": "swleemd@g.skku.edu",
        "photo": "male_icon.png",
    },
]

# Shown on the Team page only when filled in
CITATION = ""


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RA-Predict",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# STYLING  (blue and red theme)
# ============================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600;8..60,700&family=Source+Sans+3:wght@400;500;600;700&display=swap');

:root {
    --ra-blue: #2F5597;
    --ra-blue-dark: #23427A;
    --ra-blue-tint: #EAF0F9;
    --ra-red: #C0272D;
    --ra-red-dark: #9E1F24;
    --ra-red-tint: #FDEEEE;
    --ra-ink: #1F2A33;
    --ra-muted: #5B6B77;
    --ra-line: #DCE2EC;
    --ra-surface: #F5F7FB;
}

html, body, .stApp, [class*="css"] {
    font-family: 'Source Sans 3', 'Source Sans Pro', -apple-system, sans-serif;
    color: var(--ra-ink);
}
.stApp { background: #FFFFFF; }

h1, h2, h3, h4 {
    font-family: 'Source Serif 4', Georgia, serif !important;
    color: var(--ra-ink);
}

/* ---------- Page frame ---------- */
header[data-testid="stHeader"] { display: none; }
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {
    display: none;
}
[data-testid="stElementContainer"]:has(style) { display: none; }
[data-testid="stMainBlockContainer"] { padding-top: 0 !important; }
.block-container {
    max-width: 78rem !important;
    margin: 0 auto;
    padding: 0 2rem 0 2rem;
}

/* ---------- Hero ---------- */
.ra-hero {
    display: flex;
    align-items: stretch;
    gap: 1.8rem;
    margin: 1.1rem 0 1.1rem 0;
    padding: 2rem 2.3rem;
    border-radius: 16px;
    border-bottom: 5px solid var(--ra-red);
    color: #FFFFFF;
    background:
        radial-gradient(circle at 93% 22%, rgba(192, 39, 45, 0.42) 0, rgba(192, 39, 45, 0) 36%),
        linear-gradient(135deg, #23427A 0%, #2F5597 60%, #3F6BB8 100%);
}
.ra-hero-logo {
    flex: 0 0 auto;
    width: 16rem;
    min-height: 10.5rem;
    border-radius: 18px;
    background: #FFFFFF;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.18);
}
.ra-hero-logo img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    padding: 0.7rem;
}
.ra-hero-body { align-self: center; }
.ra-hero-title {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 2.3rem;
    font-weight: 700;
    line-height: 1.1;
}
.ra-hero-tag { font-size: 1.12rem; opacity: 0.96; margin-top: 0.25rem; }
.ra-hero-text { font-size: 0.95rem; opacity: 0.85; margin-top: 0.55rem; max-width: 46rem; }
.ra-badges { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.95rem; }
.ra-badge {
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.38);
}
.ra-badge.red { background: var(--ra-red); border-color: var(--ra-red); }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.3rem;
    border-bottom: 1px solid var(--ra-line);
}
.stTabs [data-baseweb="tab"] {
    height: 3rem;
    padding: 0 1.15rem;
    font-weight: 600;
    color: var(--ra-muted);
}
.stTabs [aria-selected="true"] { color: var(--ra-blue) !important; }
.stTabs [data-baseweb="tab-highlight"] {
    background-color: var(--ra-red) !important;
    height: 3px;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.1rem; }

/* ---------- Section cards ---------- */
[class*="st-key-card_"] {
    background: #FFFFFF;
    border: 1px solid var(--ra-line);
    border-radius: 14px;
    padding: 1.05rem 1.3rem 0.9rem 1.3rem;
    box-shadow: 0 1px 3px rgba(31, 42, 51, 0.05);
    gap: 0.55rem;
}
.st-key-card_scan { background: var(--ra-blue-tint); border-color: #C9D6EC; }
.ra-sec { display: flex; flex-wrap: wrap; align-items: center; gap: 0.2rem 0.7rem; margin-bottom: 0.15rem; }
.ra-sec-num {
    width: 1.75rem;
    height: 1.75rem;
    border-radius: 50%;
    background: var(--ra-blue);
    color: #FFFFFF;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.92rem;
}
.ra-sec-title {
    font-family: 'Source Serif 4', Georgia, serif;
    font-weight: 600;
    font-size: 1.22rem;
}
.ra-sec-desc { font-size: 0.88rem; color: var(--ra-muted); }

/* ---------- Inputs ---------- */
label[data-testid="stWidgetLabel"] p {
    font-size: 0.84rem;
    font-weight: 600;
    color: var(--ra-muted);
}
[data-testid="stForm"] { border: none; padding: 0; }
[data-testid="stForm"] > div { gap: 0.9rem; }

/* ---------- Buttons ---------- */
[data-testid="stBaseButton-primaryFormSubmit"],
[data-testid="stBaseButton-primary"] {
    background: var(--ra-red);
    border: 1px solid var(--ra-red);
    color: #FFFFFF;
    border-radius: 10px;
    min-height: 3.1rem;
    font-weight: 700;
    font-size: 1.02rem;
}
[data-testid="stBaseButton-primaryFormSubmit"]:hover,
[data-testid="stBaseButton-primary"]:hover {
    background: var(--ra-red-dark);
    border-color: var(--ra-red-dark);
    color: #FFFFFF;
}
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-secondaryFormSubmit"] {
    border: 1px solid var(--ra-blue);
    color: var(--ra-blue);
    border-radius: 10px;
    font-weight: 600;
}
[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stBaseButton-secondaryFormSubmit"]:hover {
    background: var(--ra-blue-tint);
    color: var(--ra-blue-dark);
    border-color: var(--ra-blue-dark);
}

/* ---------- Result ---------- */
.ra-result {
    --acc: var(--ra-blue);
    --acc-tint: var(--ra-blue-tint);
    border: 1px solid var(--ra-line);
    border-left: 6px solid var(--acc);
    border-radius: 14px;
    background: var(--acc-tint);
    padding: 1.15rem 1.4rem 1.2rem 1.4rem;
}
.ra-result.positive { --acc: var(--ra-red); --acc-tint: var(--ra-red-tint); }
.ra-result-label {
    font-size: 0.76rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--ra-muted);
}
.ra-result-value {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.65rem;
    font-weight: 700;
    color: var(--acc);
    margin: 0.15rem 0 0.7rem 0;
}
.ra-score { font-size: 2.5rem; font-weight: 700; line-height: 1; color: var(--ra-ink); }
.ra-score-cap { font-size: 0.86rem; color: var(--ra-muted); margin-left: 0.55rem; }
.ra-bar {
    position: relative;
    height: 12px;
    margin-top: 0.9rem;
    border-radius: 999px;
    background: #FFFFFF;
    border: 1px solid var(--ra-line);
}
.ra-bar-fill { height: 100%; border-radius: 999px; background: var(--acc); }
.ra-bar-mark { position: absolute; top: -5px; bottom: -5px; width: 2px; background: var(--ra-ink); }
.ra-bar-scale {
    display: flex;
    justify-content: space-between;
    margin-top: 0.3rem;
    font-size: 0.74rem;
    color: var(--ra-muted);
}
.ra-note { font-size: 0.82rem; color: var(--ra-muted); margin-top: 0.7rem; line-height: 1.5; }

/* ---------- Team ---------- */
.ra-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(270px, 1fr));
    gap: 18px;
    margin: 0.6rem 0 1rem 0;
}
.ra-card {
    border: 1px solid var(--ra-line);
    border-radius: 14px;
    background: #FFFFFF;
    padding: 1.4rem 1rem 1.2rem 1rem;
    text-align: center;
    box-shadow: 0 1px 3px rgba(31, 42, 51, 0.05);
}
.ra-card:hover { border-color: #B9C8E4; box-shadow: 0 3px 10px rgba(47, 85, 151, 0.12); }
.ra-avatar {
    width: 104px;
    height: 104px;
    margin: 0 auto 0.7rem auto;
    border-radius: 50%;
    background: var(--ra-blue);
    color: #FFFFFF;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.9rem;
    font-weight: 600;
    overflow: hidden;
}
.ra-avatar.photo { background: transparent; }
.ra-avatar img { width: 100%; height: 100%; object-fit: cover; }
.ra-name {
    font-family: 'Source Serif 4', Georgia, serif;
    font-weight: 700;
    font-size: 1.1rem;
    line-height: 1.3;
    margin: 0;
}
.ra-role {
    display: inline-block;
    margin-top: 0.35rem;
    padding: 0.1rem 0.6rem;
    border-radius: 999px;
    font-size: 0.74rem;
    font-weight: 600;
    color: var(--ra-red);
    background: var(--ra-red-tint);
}
.ra-aff { font-size: 0.86rem; line-height: 1.5; margin-top: 0.7rem; color: var(--ra-ink); }
.ra-mail { font-size: 0.86rem; margin-top: 0.7rem; word-break: break-all; }
.ra-mail a { color: var(--ra-blue); }

/* ---------- Footer ---------- */
.ra-footer {
    margin-top: 2.2rem;
    padding: 1.9rem 2.2rem 1.1rem 2.2rem;
    border-radius: 16px 16px 0 0;
    border-top: 5px solid var(--ra-red);
    background: var(--ra-blue-dark);
    color: #E6EDF8;
}
.ra-footer-grid { display: grid; grid-template-columns: 1.4fr 1fr 1fr; gap: 2rem; }
.ra-footer h5 {
    margin: 0 0 0.5rem 0;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #FFFFFF;
    opacity: 0.8;
}
.ra-footer p { margin: 0.15rem 0; font-size: 0.9rem; line-height: 1.5; }
.ra-footer .ra-lab {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #FFFFFF;
}
.ra-footer-bottom {
    margin-top: 1.4rem;
    padding-top: 0.85rem;
    border-top: 1px solid rgba(255, 255, 255, 0.2);
    text-align: center;
    font-size: 0.8rem;
    opacity: 0.85;
}
@media (max-width: 760px) {
    .ra-hero { flex-direction: column; align-items: flex-start; }
    .ra-footer-grid { grid-template-columns: 1fr; }
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)



# ============================================================
# FEATURE ORDER
# ============================================================

FEATURE_ORDER = [

    "Gender",
    "Family History",
    "Age",
    "Weight",
    "Height",
    "Exposed to Smoking",
    "Pollution",
    "HB Level",
    "ESR",
    "Creatinine",
    "CRP",
    "Anti-CCP",
    "Uric Acid Level",
    "Hypertension",
    "Insomnia",
    "Diabetes",
    "Fatigue",

]

BINARY_FEATURES = [
    "Family History",
    "Exposed to Smoking",
    "Pollution",
    "Hypertension",
    "Insomnia",
    "Diabetes",
    "Fatigue",
]


# ============================================================
# VARIABLE MAPPINGS
# ============================================================

YES_NO = {
    1: "Yes",
    0: "No"
}

GENDER = {
    1: "Male",
    0: "Female"
}


# ============================================================
# MODEL SETTINGS AND INPUT VALIDATION
# ============================================================

# Class label = model.predict (probability of RA above 0.50)
DECISION_THRESHOLD = 0.50

# Range of each continuous variable in the training data (n = 400).
# Inputs outside these ranges trigger a warning.
TRAIN_RANGES = {
    "Age": (15.00, 70.00),
    "Weight": (40.00, 120.00),
    "Height": (4.10, 6.70),
    "HB Level": (6.20, 14.10),
    "ESR": (4.00, 140.00),
    "Creatinine": (46.00, 112.50),
    "CRP": (0.00, 98.00),
    "Anti-CCP": (3.29, 98.00),
    "Uric Acid Level": (0.40, 20.00),
}


def range_issues(row):
    """Return (feature, message) pairs for values outside the training range."""
    issues = []

    for feature, (low, high) in TRAIN_RANGES.items():

        value = row.get(feature)

        if value is None or pd.isna(value):
            continue

        if value < low or value > high:
            issues.append((
                feature,
                f"{feature} = {value:g} is outside the range seen in "
                f"the training data ({low:g} to {high:g}); the prediction "
                f"may be unreliable."
            ))

    creatinine = row.get("Creatinine")

    if creatinine is not None and not pd.isna(creatinine) and creatinine < 20:
        issues.append((
            "Creatinine",
            "Creatinine is below 20. This tool expects creatinine in "
            "µmol/L (typical values here are 46 to 112). If your value "
            "is in mg/dL, multiply it by 88.4 before entering it."
        ))

    return issues


def range_warnings(row):
    """Warning messages only."""
    return [message for _, message in range_issues(row)]


def display_value(feature, value):
    """Readable value for display tables."""
    if feature == "Gender":
        return GENDER.get(int(value), str(value))

    if feature in BINARY_FEATURES:
        return YES_NO.get(int(value), str(value))

    return f"{value:g}"


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    model = XGBClassifier()

    model.load_model(str(MODEL_PATH))

    return model


if not MODEL_PATH.exists():

    st.error(
        "RA_Predict_XGB.json was not found inside the model folder."
    )

    st.stop()


try:

    model = load_model()

except Exception as e:

    st.error(str(e))
    st.stop()


# The model keeps its own column names (two of them end with a space).
# Patients are passed to it in this order with these names.
MODEL_COLUMNS = list(model.get_booster().feature_names or [])

if [c.strip() for c in MODEL_COLUMNS] != FEATURE_ORDER:

    st.error(
        "The feature names stored in the model do not match the "
        "variables of this application."
    )

    st.stop()


def to_model_frame(df):
    """Variables in the model's order and with the model's own column names."""
    frame = df[FEATURE_ORDER].copy()
    frame.columns = MODEL_COLUMNS
    return frame


# ============================================================
# SHAP EXPLANATION HELPERS
# ============================================================

@st.cache_resource
def get_explainer(_model):
    """
    Interventional TreeSHAP on the probability scale. The background is a
    sample of 100 training patients (model/background.csv) with the model's
    own column names.
    """
    background = pd.read_csv(BACKGROUND_PATH)[MODEL_COLUMNS]

    return shap.TreeExplainer(
        _model,
        data=background,
        model_output="probability",
        feature_perturbation="interventional",
    )


def ra_shap_values(explainer, X):
    """SHAP values for the RA class, shape (n_samples, n_features)."""
    values = explainer.shap_values(X)

    if isinstance(values, list):
        # older shap versions: one array per class
        return np.asarray(values[1])

    values = np.asarray(values)

    if values.ndim == 3:
        # newer shap versions: (samples, features, classes)
        return values[:, :, 1]

    return values


def explain_patient(X):
    """
    Returns (contributions DataFrame, baseline, error message).
    Contributions are in probability units for the RA class.
    """
    if not SHAP_AVAILABLE:
        return None, None, "The shap package is not installed."

    if not BACKGROUND_PATH.exists():
        return None, None, "the background file model/background.csv was not found"

    try:
        explainer = get_explainer(model)

        values = ra_shap_values(explainer, X)

        expected = np.asarray(explainer.expected_value).ravel()
        baseline = float(expected[1] if expected.size > 1 else expected[0])

        contributions = pd.DataFrame({
            "Feature": FEATURE_ORDER,
            "Value": [display_value(f, X.iloc[0, j]) for j, f in enumerate(FEATURE_ORDER)],
            "Contribution": values[0],
        })

        return contributions, baseline, None

    except Exception as e:
        return None, None, str(e)


def plot_contributions(contributions, top_n=8):

    order = contributions["Contribution"].abs().sort_values(
        ascending=False
    ).index

    top = contributions.loc[order].head(top_n).iloc[::-1]

    fig, ax = plt.subplots(figsize=(7, 0.45 * len(top) + 1.4))

    colors = [
        "#C0272D" if v > 0 else "#2F5597"
        for v in top["Contribution"]
    ]

    labels = [
        f"{f} = {v}" for f, v in zip(top["Feature"], top["Value"])
    ]

    ax.barh(labels, top["Contribution"], color=colors)

    ax.axvline(0, color="black", linewidth=0.8)

    ax.set_xlabel(
        "Contribution to model score (probability units)\n"
        "← toward Non-RA        toward RA →"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    return fig


# ============================================================
# SINGLE PATIENT RESULT
# ============================================================

def readable_patient(patient):
    """Patient values with coded variables converted to labels."""
    result = patient.copy()

    result["Gender"] = result["Gender"].map(GENDER)

    for col in BINARY_FEATURES:
        result[col] = result[col].map(YES_NO)

    return result


def render_single_result(patient):

    X = to_model_frame(patient)

    prediction = int(model.predict(X)[0])

    probability = float(model.predict_proba(X)[0, 1])

    st.markdown(
        '<div class="ra-sec" style="margin-top:0.6rem">'
        '<span class="ra-sec-title">Prediction Result</span></div>',
        unsafe_allow_html=True,
    )

    for message in range_warnings(patient.iloc[0].to_dict()):
        st.warning(message)

    contributions, baseline, error = explain_patient(X)

    top_text = ""

    if contributions is not None:

        ranked = contributions.reindex(
            contributions["Contribution"].abs().sort_values(
                ascending=False
            ).index
        ).head(5)

        top_text = "; ".join(
            f"{r.Feature}={r.Value} ({r.Contribution:+.3f})"
            for r in ranked.itertuples()
        )

    col1, col2 = st.columns([5, 7])

    # --------------------------------------------------------
    # RESULT CARD
    # --------------------------------------------------------

    with col1:

        positive = prediction == 1

        label = (
            "Rheumatoid Arthritis" if positive
            else "Non-Rheumatoid Arthritis"
        )

        percent = probability * 100

        st.markdown(
            f'<div class="ra-result {"positive" if positive else "negative"}">'
            '<div class="ra-result-label">Predicted class</div>'
            f'<div class="ra-result-value">{label}</div>'
            f'<div><span class="ra-score">{percent:.1f}%</span>'
            '<span class="ra-score-cap">model score: probability of RA'
            "</span></div>"
            '<div class="ra-bar">'
            f'<div class="ra-bar-fill" style="width:{percent:.1f}%"></div>'
            '<div class="ra-bar-mark" style="left:50%"></div></div>'
            '<div class="ra-bar-scale"><span>0%</span>'
            "<span>50% decision threshold</span><span>100%</span></div>"
            '<div class="ra-note">The class label is Rheumatoid Arthritis '
            "when the score is above 50%. The score is the predicted "
            "probability of an XGBoost model trained on a 1:1 case-control "
            "sample; it is not a population risk estimate. Research "
            "use only; it does not replace clinical diagnosis.</div>"
            "</div>",
            unsafe_allow_html=True,
        )

        report = readable_patient(patient)

        report["Prediction"] = label

        report["RA Probability (model score)"] = round(probability, 4)

        report["Top contributing variables (SHAP)"] = top_text

        st.download_button(
            "📥 Download Prediction Report",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name="RA_Prediction_Report.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # --------------------------------------------------------
    # PER-PATIENT EXPLANATION
    # --------------------------------------------------------

    with col2:

        st.markdown(
            '<div class="ra-sec"><span class="ra-sec-title">'
            "Why this prediction?</span>"
            '<span class="ra-sec-desc">SHAP explanation</span></div>',
            unsafe_allow_html=True,
        )

        if contributions is None:

            st.info(
                "Per-patient explanation is not available "
                f"({error})."
            )

        else:

            st.pyplot(plot_contributions(contributions))

            st.caption(
                f"Red bars push the score toward RA, blue bars toward "
                f"Non-RA. Starting from the average model output "
                f"({baseline * 100:.1f}%), the contributions of all 17 "
                f"variables add up to this patient's score "
                f"({probability * 100:.1f}%). SHAP describes how this "
                f"model arrived at the score; it does not show causal "
                f"effects."
            )

    # --------------------------------------------------------
    # DETAILS
    # --------------------------------------------------------

    with st.expander("Submitted values"):

        show = readable_patient(patient).T.reset_index()

        show.columns = ["Variable", "Value"]

        show["Value"] = show["Value"].astype(str)

        st.dataframe(
            show,
            hide_index=True,
            use_container_width=True
        )

    if contributions is not None:

        with st.expander("All variable contributions"):

            table = contributions.copy()

            table["Contribution"] = table["Contribution"].round(4)

            st.dataframe(
                table.sort_values(
                    "Contribution",
                    key=lambda s: s.abs(),
                    ascending=False
                ),
                hide_index=True,
                use_container_width=True
            )


# ============================================================
# REPORT UPLOAD (optional pre-fill of the laboratory values)
# ============================================================

def scan_state():
    ss = st.session_state
    ss.setdefault("prefill", {})
    ss.setdefault("prefill_version", 0)
    ss.setdefault("scan_upload_version", 0)
    ss.setdefault("scan_summary", None)
    return ss


def widget_key(name):
    """Keys change after each report is read so the form is re-filled."""
    return f"in_{name}_{st.session_state.prefill_version}"


def prefilled(name):
    return st.session_state.prefill.get(name)


@st.cache_resource
def ocr_ready():
    return ocr_available() if REPORT_READING else False


def apply_extraction(extraction):
    summary = summarise_for_form(extraction)
    ss = st.session_state
    ss.prefill = summary.pop("prefill")
    ss.prefill_version += 1
    ss.scan_summary = summary


def render_scan_card():
    ss = scan_state()

    with st.container(key="card_scan"):

        st.markdown("#### 📄 Fill laboratory values from a report (optional)")

        st.caption(
            "Upload a laboratory report as a PDF or as a photo/scan "
            "(PNG, JPG). The tool reads HB, ESR, creatinine, CRP, "
            "anti-CCP and uric acid and fills them into the form below; "
            "units printed on the report are converted automatically "
            "(for example creatinine in mg/dL to µmol/L). Check every "
            "value against the original report. Files are read in "
            "memory and are not saved by this application"
        )

        if not ocr_ready():
            st.warning(
                "Photos and scanned PDFs cannot be read on this computer "
                "because Tesseract OCR is not installed. PDFs that contain "
                "selectable text can still be read."
            )

        files = st.file_uploader(
            "Upload report",
            type=list(FILE_TYPES),
            accept_multiple_files=True,
            key=f"scan_files_{ss.scan_upload_version}",
            label_visibility="collapsed",
        )

        b1, b2, _ = st.columns([1, 1, 2])

        with b1:
            read = st.button(
                "Read report",
                disabled=not files,
                use_container_width=True,
                key="scan_read",
            )

        with b2:
            clear = st.button(
                "Clear filled values",
                disabled=not ss.prefill and ss.scan_summary is None,
                use_container_width=True,
                key="scan_clear",
            )

        if clear:
            ss.prefill = {}
            ss.scan_summary = None
            ss.prefill_version += 1
            ss.scan_upload_version += 1
            st.rerun()

        if read and files:
            with st.spinner("Reading report..."):
                extraction = read_reports(
                    [(f.name, f.getvalue()) for f in files]
                )
            apply_extraction(extraction)

        summary = ss.scan_summary

        if summary is None:
            return

        n_filled = len(summary["rows"])

        if n_filled:

            st.info(
                f"{n_filled} of {len(FIELD_LABELS)} laboratory values were "
                "filled into the form. Check each value against the "
                "original report before running the prediction."
            )

            st.dataframe(
                pd.DataFrame(summary["rows"]),
                hide_index=True,
                use_container_width=True,
            )

        else:

            st.warning(
                "No laboratory values could be read from this file. "
                "Please enter them in the form below."
            )

        if summary["attention"]:
            st.warning(
                "Needs attention:\n\n"
                + "\n".join(f"- {a}" for a in summary["attention"])
            )

        if summary["not_found"] and n_filled:
            st.caption(
                "Not filled, enter by hand: "
                + ", ".join(summary["not_found"]) + "."
            )

        with st.expander("Text read from the report"):
            for name, method, text in summary["files"]:
                st.markdown(f"**{name}** ({method or 'not read'})")
                st.text(text.strip() or "(no text)")


# ============================================================
# TEAM PAGE
# ============================================================

@st.cache_data
def image_data_uri(path_text):
    path = Path(path_text)

    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"

    data = base64.b64encode(path.read_bytes()).decode()

    return f"data:{mime};base64,{data}"


def avatar_html(author):
    photo = author.get("photo")

    if photo:
        path = TEAM_DIR / photo

        if path.exists():
            return (
                '<div class="ra-avatar photo">'
                f'<img src="{image_data_uri(str(path))}" alt="">'
                "</div>"
            )

    initials = "".join(
        part[0] for part in author["name"].split()
        if part.rstrip(".").lower() not in ("ms", "mr", "dr", "prof")
    )[:2].upper()

    return f'<div class="ra-avatar">{html.escape(initials)}</div>'


def render_team():
    st.markdown(
        '<div class="ra-sec"><span class="ra-sec-title">Team</span>'
        '<span class="ra-sec-desc">The research team behind RA-Predict'
        "</span></div>",
        unsafe_allow_html=True,
    )

    cards = []

    for author in AUTHORS:

        email = html.escape(author.get("email", ""))

        email_html = (
            f'<div class="ra-mail"><a href="mailto:{email}">{email}</a></div>'
            if email else ""
        )

        role = author.get("role", "")

        role_html = (
            f'<div class="ra-role">{html.escape(role)}</div>' if role else ""
        )

        cards.append(
            '<div class="ra-card">'
            f"{avatar_html(author)}"
            f'<div class="ra-name">{html.escape(author["name"])}</div>'
            f"{role_html}"
            f'<div class="ra-aff">{author.get("affiliation", "")}</div>'
            f"{email_html}"
            "</div>"
        )

    st.markdown(
        f'<div class="ra-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )

    if CITATION:
        st.markdown("#### How to cite")
        st.write(CITATION)


# ============================================================
# HEADER
# ============================================================

def logo_data_uri():
    if os.path.exists(LOGO_PATH):
        try:
            return image_data_uri(str(LOGO_PATH))
        except Exception:
            return ""

    return ""


def render_hero():

    logo = logo_data_uri()

    logo_html = (
        f'<div class="ra-hero-logo"><img src="{logo}" alt=""></div>'
        if logo else ""
    )

    st.markdown(
        '<div class="ra-hero">'
        f"{logo_html}"
        '<div class="ra-hero-body">'
        '<div class="ra-hero-title">RA-Predict</div>'
        '<div class="ra-hero-tag">Machine-learning-based Rheumatoid '
        "Arthritis Prediction Tool</div>"
        '<div class="ra-hero-text">Estimate the likelihood of rheumatoid '
        "arthritis from clinical, laboratory and lifestyle variables, with "
        "a per-patient explanation of the result.</div>"
        '<div class="ra-badges">'
        '<span class="ra-badge">17 variables</span>'
        '<span class="ra-badge">SHAP explanations</span>'
        '<span class="ra-badge red">Research use only</span>'
        "</div></div></div>",
        unsafe_allow_html=True,
    )


def section_header(number, title, description):
    st.markdown(
        '<div class="ra-sec">'
        f'<span class="ra-sec-num">{number}</span>'
        f'<span class="ra-sec-title">{title}</span>'
        f'<span class="ra-sec-desc">{description}</span>'
        "</div>",
        unsafe_allow_html=True,
    )


render_hero()


def render_coding_reference():

    with st.expander("📋 Variable Coding Reference"):

        coding = pd.DataFrame({

            "Variable": FEATURE_ORDER,

            "Coding": [

                "Male = 1, Female = 0",

                "Yes = 1, No = 0",

                "Age in Years",

                "Weight (kg)",

                "Height (ft), same format as the training data",

                "Yes = 1, No = 0",

                "Yes = 1, No = 0",

                "g/dL",

                "mm/hr",

                "µmol/L",

                "mg/L",

                "U/mL",

                "mg/dL",

                "Yes = 1, No = 0",

                "Yes = 1, No = 0",

                "Yes = 1, No = 0",

                "Yes = 1, No = 0"

            ]

        })

        st.table(coding)


# ============================================================
# TABS
# ============================================================

tab_single, tab_batch, tab_about, tab_team = st.tabs(
    [
        "Single Prediction",
        "Batch Prediction",
        "About",
        "Team"
    ]
)


# ============================================================
# TAB 1 : SINGLE PATIENT PREDICTION
# ============================================================

with tab_single:

    scan_state()

    if REPORT_READING:
        render_scan_card()

    with st.form("prediction_form", border=False):

        # =====================================================
        # 1. DEMOGRAPHICS
        # =====================================================

        with st.container(key="card_demo"):

            section_header(1, "Demographics", "Basic patient details")

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                gender = st.selectbox(
                    "Gender",
                    [1, 0],
                    format_func=lambda x: GENDER[x]
                )

            with c2:
                age = st.number_input(
                    "Age (Years)",
                    min_value=1,
                    max_value=120,
                    value=None,
                    placeholder="Enter age",
                    step=1
                )

            with c3:
                weight = st.number_input(
                    "Weight (kg)",
                    min_value=10.0,
                    max_value=250.0,
                    value=None,
                    placeholder="Enter weight"
                )

            with c4:
                height = st.number_input(
                    "Height (ft)",
                    min_value=3.0,
                    max_value=8.0,
                    value=None,
                    placeholder="e.g. 5.7",
                    format="%.1f",
                    help="Enter in the same format as the training data."
                )

        # =====================================================
        # 2. CLINICAL CONDITIONS
        # =====================================================

        with st.container(key="card_clinical"):

            section_header(
                2, "Clinical Conditions",
                "Family history and existing conditions"
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                family_history = st.selectbox(
                    "Family History",
                    [1, 0],
                    format_func=lambda x: YES_NO[x]
                )

            with c2:
                hypertension = st.selectbox(
                    "Hypertension",
                    [1, 0],
                    format_func=lambda x: YES_NO[x]
                )

            with c3:
                diabetes = st.selectbox(
                    "Diabetes",
                    [1, 0],
                    format_func=lambda x: YES_NO[x]
                )

        # =====================================================
        # 3. ENVIRONMENTAL AND LIFESTYLE FACTORS
        # =====================================================

        with st.container(key="card_lifestyle"):

            section_header(
                3, "Environmental and Lifestyle Factors",
                "Exposures and everyday symptoms"
            )

            c1, c2, c3, c4 = st.columns(4)

            with c1:
                smoking = st.selectbox(
                    "Exposed to Smoking",
                    [1, 0],
                    format_func=lambda x: YES_NO[x]
                )

            with c2:
                pollution = st.selectbox(
                    "Pollution Exposure",
                    [1, 0],
                    format_func=lambda x: YES_NO[x]
                )

            with c3:
                fatigue = st.selectbox(
                    "Fatigue",
                    [1, 0],
                    format_func=lambda x: YES_NO[x]
                )

            with c4:
                insomnia = st.selectbox(
                    "Insomnia",
                    [1, 0],
                    format_func=lambda x: YES_NO[x]
                )

        # =====================================================
        # 4. LABORATORY INVESTIGATIONS
        # =====================================================

        with st.container(key="card_lab"):

            section_header(
                4, "Laboratory Investigations",
                "Enter the values in the units shown"
            )

            r1c1, r1c2, r1c3 = st.columns(3)

            with r1c1:
                hb = st.number_input(
                    "HB Level (g/dL)",
                    value=prefilled("HB Level"),
                    placeholder="e.g. 13.5",
                    format="%.2f",
                    key=widget_key("HB Level")
                )

            with r1c2:
                esr = st.number_input(
                    "ESR (mm/hr)",
                    value=prefilled("ESR"),
                    placeholder="e.g. 35",
                    format="%.2f",
                    key=widget_key("ESR")
                )

            with r1c3:
                creatinine = st.number_input(
                    "Creatinine (µmol/L)",
                    value=prefilled("Creatinine"),
                    placeholder="e.g. 60",
                    format="%.2f",
                    key=widget_key("Creatinine")
                )

            r2c1, r2c2, r2c3 = st.columns(3)

            with r2c1:
                crp = st.number_input(
                    "CRP (mg/L)",
                    value=prefilled("CRP"),
                    placeholder="e.g. 8.5",
                    format="%.2f",
                    key=widget_key("CRP")
                )

            with r2c2:
                anti_ccp = st.number_input(
                    "Anti-CCP (U/mL)",
                    value=prefilled("Anti-CCP"),
                    placeholder="e.g. 45",
                    format="%.2f",
                    key=widget_key("Anti-CCP")
                )

            with r2c3:
                uric = st.number_input(
                    "Uric Acid Level (mg/dL)",
                    value=prefilled("Uric Acid Level"),
                    placeholder="e.g. 5.8",
                    format="%.2f",
                    key=widget_key("Uric Acid Level")
                )

        predict = st.form_submit_button(
            "🔬 Predict Rheumatoid Arthritis",
            use_container_width=True,
            type="primary"
        )

    # =========================================================
    # PREDICTION
    # =========================================================

    if predict:

        required = {
            "Age": age,
            "Weight": weight,
            "Height": height,
            "HB Level": hb,
            "ESR": esr,
            "Creatinine": creatinine,
            "CRP": crp,
            "Anti-CCP": anti_ccp,
            "Uric Acid Level": uric,
        }

        missing = [name for name, v in required.items() if v is None]

        if missing:

            st.error(
                "Please enter a value for: " + ", ".join(missing)
            )

        else:

            patient = pd.DataFrame([{

                "Gender": gender,
                "Family History": family_history,
                "Age": age,
                "Weight": weight,
                "Height": height,
                "Exposed to Smoking": smoking,
                "Pollution": pollution,
                "HB Level": hb,
                "ESR": esr,
                "Creatinine": creatinine,
                "CRP": crp,
                "Anti-CCP": anti_ccp,
                "Uric Acid Level": uric,
                "Hypertension": hypertension,
                "Insomnia": insomnia,
                "Diabetes": diabetes,
                "Fatigue": fatigue,

            }])

            render_single_result(patient)


    render_coding_reference()


# ============================================================
# TAB 2 : BATCH PREDICTION
# ============================================================

with tab_batch:

    st.subheader("Batch Prediction")

    st.caption(
        "Upload a CSV file containing patient information. "
        "Column names must exactly match the model features. "
        "Use the units listed in the Variable Coding Reference."
    )

    # Download template
    template = pd.DataFrame(columns=FEATURE_ORDER)

    st.download_button(
        label="📥 Download Blank Template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="RA_Template.csv",
        mime="text/csv"
    )

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded is not None:

        try:

            data = pd.read_csv(uploaded)

            # Remove accidental spaces from column names
            data.columns = data.columns.str.strip()

            missing = [
                col for col in FEATURE_ORDER
                if col not in data.columns
            ]

            if missing:

                st.error(
                    "Missing columns:\n\n" +
                    ", ".join(missing)
                )

            elif data[FEATURE_ORDER].isna().any().any():

                bad_rows = (
                    data[FEATURE_ORDER].isna().any(axis=1)
                )

                st.error(
                    "Missing values found in row(s): " +
                    ", ".join(str(i + 1) for i in data.index[bad_rows]) +
                    ". Please complete or remove these rows."
                )

            else:

                X = to_model_frame(data)

                pred = model.predict(X)

                proba = model.predict_proba(X)[:, 1]

                data["Prediction"] = [
                    "Rheumatoid Arthritis"
                    if x == 1
                    else "Non-Rheumatoid Arthritis"
                    for x in pred
                ]

                data["RA Probability (model score)"] = np.round(proba, 4)

                flags = []

                for _, row in data[FEATURE_ORDER].iterrows():

                    names = {f for f, _ in range_issues(row.to_dict())}

                    flags.append(", ".join(sorted(names)))

                data["Input range check"] = flags

                st.success(
                    f"Successfully analyzed {len(data)} patient(s)."
                )

                n_flagged = sum(1 for f in flags if f)

                if n_flagged:

                    st.warning(
                        f"{n_flagged} row(s) have values outside the "
                        f"training range or a suspected unit problem "
                        f"(see the 'Input range check' column)."
                    )

                c1, c2 = st.columns(2)

                with c1:
                    st.metric(
                        "Predicted RA",
                        int((pred == 1).sum())
                    )

                with c2:
                    st.metric(
                        "Predicted Non-RA",
                        int((pred == 0).sum())
                    )

                st.dataframe(
                    data,
                    use_container_width=True
                )

                st.download_button(
                    "📥 Download Results",
                    data.to_csv(index=False).encode("utf-8"),
                    file_name="RA_Predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

        except Exception as e:

            st.error(str(e))


    render_coding_reference()


# ============================================================
# TAB 3 : ABOUT
# ============================================================

with tab_about:

    st.header("About RA-Predict")

    st.markdown("""
### RA-Predict

RA-Predict is a machine-learning-based research prototype
that estimates the likelihood of Rheumatoid Arthritis from
17 clinical, laboratory and demographic variables using an
XGBoost classifier.

---

### Model

- XGBoost Classifier (gradient-boosted decision trees)
- Class label: Rheumatoid Arthritis when the model score
  (probability of RA) is above 0.50

---

### Input Variables

The model uses **17 variables** including:

- Demographics
- Clinical Conditions
- Environmental and Lifestyle Factors
- Laboratory Investigations

---

### Output

For each patient the application reports:

- The predicted class (Rheumatoid Arthritis or Non-Rheumatoid Arthritis)
- The model score (probability of RA)
- A per-patient SHAP explanation showing which variables
  raised or lowered the score

---

### Reading laboratory reports (optional)

- On the Single Prediction tab a laboratory report can be uploaded
  as a PDF or as a photo or scan (PNG, JPG). The tool reads the six
  laboratory values (HB, ESR, creatinine, CRP, anti-CCP, uric acid)
  and fills them into the form.
- Units printed on the report are converted to the units the model
  uses.
- Text recognition (OCR) can misread digits. Every value must be
  checked against the original report before the prediction is run.
- Files are read in memory and are not saved by this application.

---

### Intended use and limitations
- Trained on 400 participants (200 RA, 200 controls) from a single
  hospital in Pakistan, as a 1:1 case-control sample of established
  RA. It has not been validated in an external population or in
  patients with early or undifferentiated arthritis.
- The model score is the predicted probability within this
  case-control sample. It is not a population risk estimate, and
  scores are frequently close to 0% or 100%.
- The tool needs laboratory inputs (ESR, CRP, anti-CCP,
  haemoglobin, creatinine, uric acid), so it does not remove the
  cost of laboratory testing.
- SHAP explains how the model reached its score; it does not
  show causal effects.
- Inputs outside the training range produce a warning, and the
  prediction should then be treated as unreliable.

---

### Disclaimer

This software is intended for **Research Use Only**
""")


# ============================================================
# TAB 4 : TEAM
# ============================================================

with tab_team:

    render_team()


# ============================================================
# FOOTER
# ============================================================

LAB_NAME = "Medical Big Data Lab"
LAB_DEPARTMENT = "Department of Precision Medicine"
LAB_INSTITUTION = "Sungkyunkwan University, Republic of Korea"


def render_footer():

    citation_text = CITATION or (
        f"If RA-Predict supports your research, please acknowledge the "
        f"{LAB_NAME}, {LAB_DEPARTMENT}, Sungkyunkwan University."
    )

    st.markdown(
        '<div class="ra-footer"><div class="ra-footer-grid">'
        # Developer
        "<div><h5>Developed by</h5>"
        f'<p class="ra-lab">{LAB_NAME}</p>'
        f"<p>{LAB_DEPARTMENT}</p>"
        f"<p>{LAB_INSTITUTION}</p></div>"
        # Citation
        f"<div><h5>Citation</h5><p>{citation_text}</p></div>"
        # Disclaimer
        "<div><h5>Disclaimer</h5>"
        "<p>For research and educational use only. Not a substitute for "
        "professional clinical diagnosis.</p></div>"
        "</div>"
        # Bottom line
        '<div class="ra-footer-bottom">'
        f"&copy; 2026 {LAB_NAME}, {LAB_DEPARTMENT}, Sungkyunkwan "
        "University. All rights reserved.</div></div>",
        unsafe_allow_html=True,
    )


render_footer()
