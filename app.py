
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_PATH = "/content/drive/MyDrive/EEDC_Outage_Predictor"

MODEL_PATH = os.path.join(
    PROJECT_PATH,
    "models",
    "eedc_outage_model.pkl"
)

THRESHOLD_PATH = os.path.join(
    PROJECT_PATH,
    "models",
    "prediction_threshold.pkl"
)

DATA_PATH = os.path.join(
    PROJECT_PATH,
    "data",
    "outage_data.csv"
)

EVALUATION_PATH = os.path.join(
    PROJECT_PATH,
    "outputs",
    "model_evaluation.json"
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EEDC Outage Predictor",
    page_icon="⚡",
    layout="wide"
)

# ============================================================
# LOAD MODEL AND DATA
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    data["date"] = pd.to_datetime(data["date"])
    return data


@st.cache_data
def load_evaluation():
    with open(EVALUATION_PATH, "r") as f:
        return json.load(f)


model = load_model()
threshold = joblib.load(THRESHOLD_PATH)
df = load_data()
evaluation = load_evaluation()

# ============================================================
# BACKEND LOOKUP
# ============================================================

def get_backend_inputs(state, area, hour):

    matches = df[
        (df["state"] == state) &
        (df["area"] == area) &
        (df["hour"] == hour)
    ]

    if len(matches) > 0:
        row = matches.iloc[0]

    else:

        location_data = df[
            (df["state"] == state) &
            (df["area"] == area)
        ].copy()

        if len(location_data) == 0:
            return None

        location_data["hour_difference"] = (
            abs(location_data["hour"] - hour)
        )

        row = location_data.sort_values(
            "hour_difference"
        ).iloc[0]

    return {
        "rainfall_mm": float(row["rainfall_mm"]),
        "estimated_load_mw": float(
            row["estimated_load_mw"]
        ),
        "high_load": int(row["high_load"]),
        "previous_outage_count_24h": int(
            row["previous_outage_count_24h"]
        ),
        "previous_outage_duration_min": float(
            row["previous_outage_duration_min"]
        ),
        "heavy_rain": int(row["heavy_rain"])
    }


# ============================================================
# PREDICTION
# ============================================================

def predict_outage(state, area, hour):

    backend = get_backend_inputs(
        state,
        area,
        hour
    )

    if backend is None:
        return None

    current_date = pd.Timestamp.now()

    model_input = pd.DataFrame([{
        "state": state,
        "area": area,
        "hour": hour,
        "rainfall_mm": backend["rainfall_mm"],
        "estimated_load_mw": backend["estimated_load_mw"],
        "high_load": backend["high_load"],
        "previous_outage_count_24h":
            backend["previous_outage_count_24h"],
        "previous_outage_duration_min":
            backend["previous_outage_duration_min"],
        "heavy_rain": backend["heavy_rain"],
        "month": current_date.month,
        "day_of_week": current_date.dayofweek
    }])

    probability = model.predict_proba(
        model_input
    )[0][1]

    prediction = int(
        probability >= threshold
    )

    if probability >= 0.70:
        risk = "VERY HIGH"
    elif probability >= 0.50:
        risk = "HIGH"
    elif probability >= 0.30:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "probability": probability * 100,
        "prediction": prediction,
        "risk": risk
    }


# ============================================================
# HEADER
# ============================================================

st.title("⚡ EEDC Outage Predictor")

st.markdown(
    """
    ### Predict the likelihood of an electricity outage
    within the next 6 hours.

    Select your location and expected time. The system
    automatically processes backend environmental and
    load information.
    """
)

st.divider()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("📍 Prediction Input")

states = sorted(df["state"].unique())

selected_state = st.sidebar.selectbox(
    "State",
    states
)

areas = sorted(
    df[df["state"] == selected_state]["area"].unique()
)

selected_area = st.sidebar.selectbox(
    "Area",
    areas
)

selected_hour = st.sidebar.slider(
    "Hour",
    min_value=0,
    max_value=23,
    value=19
)

predict_button = st.sidebar.button(
    "🔮 Predict Outage",
    use_container_width=True
)

# ============================================================
# MAIN PREDICTION
# ============================================================

if predict_button:

    result = predict_outage(
        selected_state,
        selected_area,
        selected_hour
    )

    if result is None:

        st.error(
            "No backend information is available "
            "for this location."
        )

    else:

        st.subheader("Prediction Result")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Outage Probability",
                f"{result['probability']:.1f}%"
            )

        with col2:
            st.metric(
                "Prediction",
                "OUTAGE LIKELY"
                if result["prediction"] == 1
                else "OUTAGE UNLIKELY"
            )

        with col3:
            st.metric(
                "Risk Level",
                result["risk"]
            )

        if result["prediction"] == 1:

            st.warning(
                f"⚠️ OUTAGE LIKELY — "
                f"{result['probability']:.1f}% probability"
            )

        else:

            st.success(
                f"✅ OUTAGE UNLIKELY — "
                f"{result['probability']:.1f}% probability"
            )


# ============================================================
# TREND VIEW
# ============================================================

st.divider()

st.header("📈 Outage Trend")

trend = (
    df.groupby("date")["outage_next_6h"]
    .mean()
    .reset_index()
)

trend["outage_probability"] = (
    trend["outage_next_6h"] * 100
)

st.line_chart(
    trend.set_index("date")[
        "outage_probability"
    ]
)

st.caption(
    "Historical outage probability in the dataset."
)


# ============================================================
# MODEL EVALUATION
# ============================================================

st.divider()

st.header("📊 Model Evaluation")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Accuracy",
        f"{evaluation['accuracy'] * 100:.1f}%"
    )

with col2:
    st.metric(
        "Precision",
        f"{evaluation['precision'] * 100:.1f}%"
    )

with col3:
    st.metric(
        "Recall",
        f"{evaluation['recall'] * 100:.1f}%"
    )

with col4:
    st.metric(
        "F1 Score",
        f"{evaluation['f1_score'] * 100:.1f}%"
    )

with col5:
    st.metric(
        "ROC-AUC",
        f"{evaluation['roc_auc']:.2f}"
    )

st.info(
    f"Prediction threshold used: {threshold:.2f}"
)

st.subheader("Confusion Matrix")

cm = np.array(
    evaluation["confusion_matrix"]
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual: No Outage", "Actual: Outage"],
    columns=["Predicted: No Outage", "Predicted: Outage"]
)

st.dataframe(
    cm_df,
    use_container_width=True
)

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "EEDC Outage Predictor — 3MTT MVP"
)
