import streamlit as st
import pandas as pd
import numpy as np
import datetime
import altair as alt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ⚠️ Optional XGBoost (safe import)
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

# --- SETTINGS ---
st.set_page_config(page_title="Urban Traffic Flow Ops", layout="wide", page_icon="🚦")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz"
        df = pd.read_csv(url, compression='gzip')
    except:
        st.error("❌ Failed to load dataset. Upload locally for reliability.")
        return None

    df['date_time'] = pd.to_datetime(df['date_time'])
    df['hour'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month

    df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if (8 <= x <= 11) or (17 <= x <= 21) else 0)

    severity = {'Clear': 0, 'Clouds': 1, 'Rain': 2, 'Drizzle': 2,
                'Mist': 2, 'Haze': 2, 'Fog': 3, 'Snow': 4,
                'Thunderstorm': 5, 'Squall': 5, 'Smoke': 4}

    df['weather_severity'] = df['weather_main'].map(severity).fillna(1)

    df = df.sort_values('date_time')
    df['traffic_lag_1h'] = df['traffic_volume'].shift(1).bfill()

    return df

# --- MODEL TRAINING (LIGHTWEIGHT) ---
@st.cache_resource
def train_models(df):
    features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all',
                'hour', 'day_of_week', 'month',
                'is_peak_hour', 'weather_severity', 'traffic_lag_1h']

    X = df[features]
    y = df['traffic_volume']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=30, n_jobs=1, random_state=42)
    }

    if XGB_AVAILABLE:
        models["XGBoost"] = xgb.XGBRegressor(n_estimators=30, n_jobs=1, random_state=42)

    trained_models = {}
    metrics = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        metrics[name] = {
            "R2": r2_score(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds))
        }

        trained_models[name] = model

    return trained_models, metrics, scaler, features


# --- APP START ---
st.title("🚦 Urban Traffic Flow Optimization System")

df = load_data()

if df is None:
    st.stop()

models, metrics, scaler, feature_cols = train_models(df)

# --- SIDEBAR ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["EDA", "Prediction"])

st.sidebar.markdown("### Model R² Scores")
for m, v in metrics.items():
    st.sidebar.write(f"{m}: {v['R2']:.3f}")

# --- EDA PAGE ---
if page == "EDA":
    st.header("📊 Dataset Overview")
    st.write(df.head())

    hourly = df.groupby('hour')['traffic_volume'].median().reset_index()
    st.line_chart(hourly, x='hour', y='traffic_volume')

    weather = df.groupby('weather_main')['traffic_volume'].mean().reset_index()
    st.bar_chart(weather, x='weather_main', y='traffic_volume')

# --- PREDICTION PAGE ---
elif page == "Prediction":
    st.header("🤖 Traffic Prediction")

    col1, col2 = st.columns(2)

    with col1:
        time_input = st.time_input("Time", datetime.time(8, 0))
        day = st.selectbox("Day", list(range(7)))

    with col2:
        temp = st.slider("Temperature", 240.0, 310.0, 290.0)
        lag = st.number_input("Previous Traffic", value=3000)

    model_choice = st.selectbox("Model", list(models.keys()))

    if st.button("Predict"):
        hour = time_input.hour
        is_peak = 1 if (8 <= hour <= 11) or (17 <= hour <= 21) else 0

        input_data = pd.DataFrame([[
            temp, 0, 0, 50,
            hour, day, 6,
            is_peak, 1, lag
        ]], columns=feature_cols)

        scaled = scaler.transform(input_data)
        pred = models[model_choice].predict(scaled)[0]

        st.success(f"🚗 Predicted Traffic: {int(pred)} vehicles/hour")
