import streamlit as st
import pandas as pd
import numpy as np
import joblib

# LOAD ARTIFACTS
artifacts = joblib.load("models/aussie_rain.joblib")

model = artifacts["model"]
imputer = artifacts["imputer"]
scaler = artifacts["scaler"]
encoder = artifacts["encoder"]

numeric_cols = artifacts["numeric_cols"]
categorical_cols = artifacts["categorical_cols"]
encoded_cols = artifacts["encoded_cols"]

feature_columns = numeric_cols + encoded_cols

# PREDICTION FUNCTION
def predict_rain(user_data):
    input_df = pd.DataFrame([user_data])

    # numeric
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # categorical
    encoded = encoder.transform(input_df[categorical_cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoded_cols,
        index=input_df.index
    )

    final_df = pd.concat([input_df[numeric_cols], encoded_df], axis=1)

    final_df = final_df.reindex(columns=feature_columns, fill_value=0)

    pred = model.predict(final_df)[0]
    proba = model.predict_proba(final_df)[0]

    rain_proba = proba[list(model.classes_).index("Yes")]

    return pred, rain_proba



# LOAD DATA FOR RANGES

df = pd.read_csv("data/weatherAUS.csv")

numeric_features = [
    "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
    "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
    "Humidity9am", "Humidity3pm",
    "Pressure9am", "Pressure3pm",
    "Cloud9am", "Cloud3pm",
    "Temp9am", "Temp3pm"
]

RANGES = {
    col: (
        float(df[col].min()),
        float(df[col].max()),
        float(df[col].median())
    )
    for col in numeric_features
}

categorical_features = [
    "Location",
    "WindGustDir",
    "WindDir9am",
    "WindDir3pm",
    "RainToday"
]

OPTIONS = {
    col: sorted(df[col].dropna().unique().tolist())
    for col in categorical_features
}

# UI
st.set_page_config(
    page_title="Прогноз дощу",
    page_icon="🌦",
    layout="wide"
)

st.title("Прогноз дощу 🌧")
st.markdown("""
Введіть поточні погодні умови та отримайте прогноз моделі машинного навчання
щодо ймовірності дощу завтра.
""")
with st.expander("⚙️ Ввести погодні параметри"):

    st.header("📍 Локація та погода")
    location = st.selectbox("Локація", OPTIONS["Location"])
    rain_today = st.selectbox("Чи був дощ сьогодні", OPTIONS["RainToday"])

    col1, col2 = st.columns(2)

    with col1:
        cloud_9am = st.slider("Хмарність 9:00", *RANGES["Cloud9am"])
        sunshine = st.slider("Сонячні години", *RANGES["Sunshine"])

    with col2:
        cloud_3pm = st.slider("Хмарність 15:00", *RANGES["Cloud3pm"])
        rainfall = st.slider("Опади", *RANGES["Rainfall"])

    
    st.header("🌡 Температура")

    col1, col2 = st.columns(2)

    with col1:
        min_temp = st.slider("Мін температура", *RANGES["MinTemp"])
        temp_9am = st.slider("Температура 9:00", *RANGES["Temp9am"])

    with col2:
        max_temp = st.slider("Макс температура", *RANGES["MaxTemp"])
        temp_3pm = st.slider("Температура 15:00", *RANGES["Temp3pm"])


    st.header("💧 Вологість і тиск")

    col1, col2 = st.columns(2)

    with col1:
        humidity_9am = st.slider("Вологість 9:00", *RANGES["Humidity9am"])
        pressure_9am = st.slider("Тиск 9:00", *RANGES["Pressure9am"])
        evaporation = st.slider("Випаровування", *RANGES["Evaporation"])

    with col2:
        humidity_3pm = st.slider("Вологість 15:00", *RANGES["Humidity3pm"])
        pressure_3pm = st.slider("Тиск 15:00", *RANGES["Pressure3pm"])


    st.header("🌬 Вітер")

    wind_gust_dir = st.selectbox("Порив вітру", OPTIONS["WindGustDir"])
    wind_gust_speed = st.slider("Пориви вітру", *RANGES["WindGustSpeed"])

    col1, col2 = st.columns(2)

    with col1:
        wind_dir_9am = st.selectbox("Вітер 9:00", OPTIONS["WindDir9am"])
        wind_speed_9am = st.slider("Швидкість 9:00", *RANGES["WindSpeed9am"])
        
    with col2:
        wind_dir_3pm = st.selectbox("Вітер 15:00", OPTIONS["WindDir3pm"])
        wind_speed_3pm = st.slider("Швидкість 15:00", *RANGES["WindSpeed3pm"])


# USER DATA 
user_data = {
    "Location": location,
    "WindGustDir": wind_gust_dir,
    "WindDir9am": wind_dir_9am,
    "WindDir3pm": wind_dir_3pm,
    "RainToday": rain_today,

    "MinTemp": min_temp,
    "MaxTemp": max_temp,
    "Rainfall": rainfall,
    "Evaporation": evaporation,
    "Sunshine": sunshine,

    "WindGustSpeed": wind_gust_speed,
    "WindSpeed9am": wind_speed_9am,
    "WindSpeed3pm": wind_speed_3pm,

    "Humidity9am": humidity_9am,
    "Humidity3pm": humidity_3pm,

    "Pressure9am": pressure_9am,
    "Pressure3pm": pressure_3pm,

    "Cloud9am": cloud_9am,
    "Cloud3pm": cloud_3pm,

    "Temp9am": temp_9am,
    "Temp3pm": temp_3pm
}



# BUTTON
if st.button("🔮 Передбачити дощ"):
    pred, prob = predict_rain(user_data)

    st.subheader("📊 Результат")

    col1, col2 = st.columns(2)

    with col1:
        if pred == "Yes":
            st.error("🌧 Очікується дощ")
        else:
            st.success("☀️ Дощу не буде")

    with col2:
        st.metric("Ймовірність", f"{prob:.2%}")