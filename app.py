import streamlit as st
import pandas as pd
import numpy as np
import datetime
import altair as alt

# --- SAFE IMPORTS ---
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
except:
    st.error("❌ scikit-learn is not installed. Add it to requirements.txt")
    st.stop()

# Optional XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

# --- SETTINGS ---
st.set_page_config(page_title="Urban Traffic Flow Ops", layout="wide", page_icon="🚦")

# --- DATA LOADING ---
@st.cache_data
def load_and_preprocess_data():
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz"
        df = pd.read_csv(url, compression='gzip')
    except:
        st.error("❌ Failed to load dataset (network issue).")
        st.stop()

    df['date_time'] = pd.to_datetime(df['date_time'])
    df['date'] = df['date_time'].dt.date
    df['hour'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month

    df['is_peak_hour'] = df['hour'].apply(
        lambda x: 1 if (8 <= x <= 11) or (17 <= x <= 21) else 0
    )

    df['day_type'] = df['day_of_week'].apply(
        lambda x: 'Weekend' if x >= 5 else 'Working'
    )

    severity = {
        'Clear': 0, 'Clouds': 1, 'Rain': 2, 'Drizzle': 2,
        'Mist': 2, 'Haze': 2, 'Fog': 3, 'Snow': 4,
        'Thunderstorm': 5, 'Squall': 5, 'Smoke': 4
    }

    df['weather_severity'] = df['weather_main'].map(severity).fillna(1)

    df = df.sort_values('date_time')
    df['traffic_lag_1h'] = df['traffic_volume'].shift(1).bfill()

    return df

# --- MODEL TRAINING ---
@st.cache_resource
def train_models(df):
    features = [
        'temp', 'rain_1h', 'snow_1h', 'clouds_all',
        'hour', 'day_of_week', 'month',
        'is_peak_hour', 'weather_severity', 'traffic_lag_1h'
    ]

    X = df[features]
    y = df['traffic_volume']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=50, random_state=42, n_jobs=1  # safer
        )
    }

    if XGB_AVAILABLE:
        models['XGBoost'] = xgb.XGBRegressor(
            n_estimators=50, random_state=42, n_jobs=1
        )

    trained_models = {}
    metrics = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        metrics[name] = {
            'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
            'MAE': mean_absolute_error(y_test, preds),
            'R2': r2_score(y_test, preds)
        }

        trained_models[name] = model

    return trained_models, metrics, scaler, features

# --- RUN APP ---
st.title("🚦 Urban Traffic Flow Optimization System")

df = load_and_preprocess_data()
models, metrics, scaler, feature_cols = train_models(df)

st.success("✅ App loaded successfully!")
