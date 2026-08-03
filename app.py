"""
============================================================
                     RA-PREDICT
     AI-based Rheumatoid Arthritis Prediction Tool
============================================================

Random Forest Classifier

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

import joblib
import pandas as pd
import streamlit as st


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

MODEL_PATH = Path(resource_path("model/RA_Predict.pkl"))
LOGO_PATH = resource_path("assets/logo.png")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RA-Predict",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)


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
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    try:
        model.feature_names_in_ = [
            x.strip() for x in model.feature_names_in_
        ]
    except Exception:
        pass

    return model


if not MODEL_PATH.exists():

    st.error(
        "RA_Predict.pkl was not found inside the model folder."
    )

    st.stop()


try:

    model = load_model()

except Exception as e:

    st.error(str(e))
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About")

    st.info(
        """
RA-Predict is an AI-based clinical decision support
tool developed using a Random Forest classifier for
prediction of Rheumatoid Arthritis.

Research Use Only.
        """
    )

    st.divider()

    st.markdown("### Model")

    st.success("Random Forest Classifier")

    st.divider()

    st.markdown("### Coding")

    st.write("Male = 1")
    st.write("Female = 0")
    st.write("Yes = 1")
    st.write("No = 0")


# ============================================================
# HEADER
# ============================================================

c1, c2, c3 = st.columns([1,2,1])

with c2:

    st.image(LOGO_PATH, use_container_width=True)

st.title("RA-Predict")

st.subheader(
    "AI-based Rheumatoid Arthritis Prediction Tool"
)

st.caption(
    "Machine Learning-based Clinical Decision Support System"
)

st.divider()


# ============================================================
# TABS
# ============================================================

tab_single, tab_batch, tab_about = st.tabs(
    [
        "Single Prediction",
        "Batch Prediction",
        "About"
    ]
)
# ============================================================
# TAB 1 : SINGLE PATIENT PREDICTION
# ============================================================

with tab_single:

    st.subheader("Patient Clinical Information")

    with st.form("prediction_form"):

        left, right = st.columns(2)

        # =====================================================
        # LEFT COLUMN
        # =====================================================

        with left:

            st.markdown("## 👤 Demographics")

            gender = st.selectbox(
                "Gender",
                [1, 0],
                format_func=lambda x: GENDER[x]
            )

            age = st.number_input(
                "Age (Years)",
                min_value=1,
                max_value=120,
                value=None,
                placeholder="Enter age",
                step=1
                
            )

            weight = st.number_input(
                "Weight (kg)",
                min_value=10.0,
                max_value=250.0,
                value=None,
                placeholder="Enter weight"
            )

            height = st.number_input(
                "Height (ft)",
                min_value=3.0,
                max_value=8.0,
                value=None,
                placeholder="e.g. 5.7",
                format="%.1f"
            )

            st.divider()

            st.markdown("## 🩺 Clinical Conditions")

            family_history = st.selectbox(
                "Family History",
                [1, 0],
                format_func=lambda x: YES_NO[x]
            )

            hypertension = st.selectbox(
                "Hypertension",
                [1, 0],
                format_func=lambda x: YES_NO[x]
            )

            diabetes = st.selectbox(
                "Diabetes",
                [1, 0],
                format_func=lambda x: YES_NO[x]
            )

            fatigue = st.selectbox(
                "Fatigue",
                [1, 0],
                format_func=lambda x: YES_NO[x]
            )

            insomnia = st.selectbox(
                "Insomnia",
                [1, 0],
                format_func=lambda x: YES_NO[x]
            )

        # =====================================================
        # RIGHT COLUMN
        # =====================================================

        with right:

            st.markdown("## 🌍 Environmental Risk Factors")

            smoking = st.selectbox(
                "Exposed to Smoking",
                [1, 0],
                format_func=lambda x: YES_NO[x]
            )

            pollution = st.selectbox(
                "Pollution Exposure",
                [1, 0],
                format_func=lambda x: YES_NO[x]
            )

            st.divider()

            st.markdown("## 🧪 Laboratory Investigations")

            hb = st.number_input(
                "HB Level (g/dL)",
                value=None,
                placeholder="e.g. 13.5",
                format="%.2f"
            )

            esr = st.number_input(
                "ESR (mm/hr)",
                value=None,
                placeholder="e.g. 35",
                format="%.2f"
            )

            creatinine = st.number_input(
                "Creatinine (µmol/L)",
                value=None,
                placeholder="e.g. 1.2",
                format="%.2f"
            )

            crp = st.number_input(
                "CRP (mg/L)",
                value=None,
                placeholder="e.g. 8.5",
                format="%.2f"
            )

            anti_ccp = st.number_input(
                "Anti-CCP (U/mL)",
                value=None,
                placeholder="e.g. 45",
                format="%.2f"
            )

            uric = st.number_input(
                "Uric Acid Level (mg/dL)",
                value=None,
                placeholder="e.g. 5.8",
                format="%.2f"
            )

        st.divider()

        predict = st.form_submit_button(
            "🔬 Predict Rheumatoid Arthritis",
            use_container_width=True,
            type="primary"
        )

    # =========================================================
    # PREDICTION
    # =========================================================

    if predict:

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

        prediction = int(model.predict(patient[FEATURE_ORDER])[0])

        st.divider()

        st.subheader("Prediction Result")

        col1, col2 = st.columns([1, 1])

        with col1:

            if prediction == 1:

                st.error("🔴 Prediction: Rheumatoid Arthritis")

                st.markdown(
                    """
The entered demographic, environmental,
clinical and laboratory profile is
**predicted to be consistent with
Rheumatoid Arthritis.**
                    """
                )

            else:

                st.success("🟢 Prediction: Non-Rheumatoid Arthritis")

                st.markdown(
                    """
The entered demographic, environmental,
clinical and laboratory profile is
**predicted to be Non-Rheumatoid Arthritis.**
                    """
                )

            st.info(
                "Research Use Only. This tool should not replace clinical diagnosis."
            )

        with col2:

            st.markdown("### Submitted Values")

            show = patient.T.reset_index()

            show.columns = ["Variable", "Value"]

            st.dataframe(
                show,
                hide_index=True,
                use_container_width=True
            )
# ============================================================
# DOWNLOAD SINGLE PATIENT RESULT
# ============================================================

        result = patient.copy()

        # Convert coded values to readable values
        result["Gender"] = result["Gender"].map({1: "Male", 0: "Female"})

        binary_cols = [
            "Family History",
            "Exposed to Smoking",
            "Pollution",
            "Hypertension",
            "Insomnia",
            "Diabetes",
            "Fatigue",
        ]

        for col in binary_cols:
            result[col] = result[col].map({1: "Yes", 0: "No"})

        result["Prediction"] = (
            "Rheumatoid Arthritis"
            if prediction == 1
            else "Non-Rheumatoid Arthritis"
        )

        csv = result.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download Prediction Report",
            data=csv,
            file_name="RA_Prediction_Report.csv",
            mime="text/csv",
            use_container_width=True,
        )
# ============================================================
# TAB 2 : BATCH PREDICTION
# ============================================================

with tab_batch:

    st.subheader("Batch Prediction")

    st.caption(
        "Upload a CSV file containing patient information. "
        "Column names must exactly match the model features."
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

            else:

                pred = model.predict(
                    data[FEATURE_ORDER]
                )

                data["Prediction"] = [
                    "Rheumatoid Arthritis"
                    if x == 1
                    else "Non-Rheumatoid Arthritis"
                    for x in pred
                ]

                st.success(
                    f"Successfully analyzed {len(data)} patient(s)."
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


# ============================================================
# TAB 3 : ABOUT
# ============================================================

with tab_about:

    st.header("About RA-Predict")

    st.markdown("""
### RA-Predict

RA-Predict is an Artificial Intelligence-based
clinical decision support tool developed for
prediction of Rheumatoid Arthritis using a
Random Forest machine learning algorithm.

The application is intended to assist
researchers in evaluating clinical,
environmental and laboratory parameters.

---

### Model

- Random Forest Classifier

---

### Input Variables

The model uses **17 variables** including:

- Demographics
- Clinical Conditions
- Environmental Risk Factors
- Laboratory Investigations

---

### Output

The application predicts:

- Rheumatoid Arthritis
- Non-Rheumatoid Arthritis

---

### Disclaimer

This software is intended for

**Research Use Only**

It should not be used as a substitute
for professional medical diagnosis or
clinical decision making.
""")


# ============================================================
# VARIABLE CODING
# ============================================================

st.divider()

with st.expander("📋 Variable Coding Reference"):

    coding = pd.DataFrame({

        "Variable": FEATURE_ORDER,

        "Coding": [

            "Male = 1, Female = 0",

            "Yes = 1, No = 0",

            "Age in Years",

            "Weight (kg)",

            "Height (ft)",

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
# FOOTER
# ============================================================

st.divider()

st.caption(
    "© 2026 RA-Predict | AI-based Rheumatoid Arthritis Prediction Tool"
)

st.caption(
    "Developed for research purposes using a Random Forest classifier."
)
