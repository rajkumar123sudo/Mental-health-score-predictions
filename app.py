import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Mental Health Score Predictor",
    page_icon="🧠",
    layout="centered"
)

# --------------------------------------------------
# Load model
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "mental_health_model.pkl"


@st.cache_resource
def load_model():
    model_data = joblib.load(MODEL_PATH)
    return model_data["model"], model_data["top_countries"]


model, top_countries = load_model()


# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("🧠 Mental Health Score Predictor")
st.write("Enter the student's information to predict the mental health score.")


# --------------------------------------------------
# Input fields
# --------------------------------------------------
age = st.number_input(
    "Age",
    min_value=10,
    max_value=100,
    value=18,
    step=1
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

academic_level = st.selectbox(
    "Academic Level",
    ["Undergraduate", "Graduate", "High School"]
)

country = st.text_input(
    "Country",
    value="India"
)

most_used_platform = st.selectbox(
    "Most Used Platform",
    [
        "Facebook",
        "LinkedIn",
        "Instagram",
        "Snapchat",
        "Twitter",
        "YouTube",
        "TikTok",
        "LINE",
        "KakaoTalk",
        "VKontakte",
        "WhatsApp",
        "WeChat"
    ]
)

purpose_of_use = st.selectbox(
    "Purpose of Use",
    [
        "Networking",
        "Education",
        "Entertainment",
        "News"
    ]
)

avg_daily_usage_hours = st.number_input(
    "Average Daily Usage Hours",
    min_value=0.0,
    max_value=24.0,
    value=4.0,
    step=0.1
)

daily_unlocks = st.number_input(
    "Daily Unlocks",
    min_value=0,
    value=50,
    step=1
)

study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=4.0,
    step=0.1
)

physical_activity_hours = st.number_input(
    "Physical Activity Hours",
    min_value=0.0,
    max_value=24.0,
    value=1.0,
    step=0.1
)

sleep_hours_per_night = st.number_input(
    "Sleep Hours Per Night",
    min_value=0.0,
    max_value=24.0,
    value=7.0,
    step=0.1
)

stress_level = st.selectbox(
    "Stress Level",
    ["Low", "Medium", "High", "Very High"]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("🔮 Predict Mental Health Score", type="primary"):

    country_group = (
        country if country in top_countries else "Other"
    )

    input_row = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Country": country,
        "Academic_Level": academic_level,
        "Most_Used_Platform": most_used_platform,
        "Purpose_Of_Use": purpose_of_use,
        "Avg_Daily_Usage_Hours": avg_daily_usage_hours,
        "Daily_Unlocks": daily_unlocks,
        "Study_Hours": study_hours,
        "Physical_Activity_Hours": physical_activity_hours,
        "Sleep_Hours_Per_Night": sleep_hours_per_night,
        "Stress_Level": stress_level,
        "grouped_country": country_group
    }])

    try:
        prediction = model.predict(input_row)[0]

        st.success(
            f"Predicted Mental Health Score: **{float(prediction):.2f}**"
        )

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)