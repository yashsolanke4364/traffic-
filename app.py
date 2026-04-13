import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import datetime
import altair as alt

# --- SETTINGS ---
st.set_page_config(page_title="Urban Traffic Flow Ops", layout="wide", page_icon="🚦")

# --- 1. DATA LOADING & PREPROCESSING (Cached) ---
@st.cache_data
def load_and_preprocess_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz"
    df = pd.read_csv(url, compression='gzip')
    
    # Handle dates
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['date'] = df['date_time'].dt.date
    df['hour'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month
    
    # Feature Engineering (Step 3)
    # Peak hour: 8-11 AM, 5-9 PM (17-21)
    df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if (8 <= x <= 11) or (17 <= x <= 21) else 0)
    
    # Day type: 0=Working, 1=Weekend
    df['day_type'] = df['day_of_week'].apply(lambda x: 'Weekend' if x >= 5 else 'Working')
    
    # Weather severity index (simple mapping)
    severity = {'Clear': 0, 'Clouds': 1, 'Rain': 2, 'Drizzle': 2, 'Mist': 2, 'Haze': 2, 'Fog': 3, 'Snow': 4, 'Thunderstorm': 5, 'Squall': 5, 'Smoke': 4}
    df['weather_severity'] = df['weather_main'].map(severity).fillna(1)
    
    # For Lag (Simulating Lag by grouping)
    df = df.sort_values('date_time')
    df['traffic_lag_1h'] = df['traffic_volume'].shift(1).bfill()
    
    return df

@st.cache_resource
def train_models(df):
    features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'hour', 'day_of_week', 'month', 'is_peak_hour', 'weather_severity', 'traffic_lag_1h']
    target = 'traffic_volume'
    
    X = df[features]
    y = df[target]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numerical
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
        'XGBoost': xgb.XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    }
    
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

# --- RUN PIPELINE ---
st.title("🚦 Urban Traffic Flow Optimization System")
st.markdown("A complete data analytics platform to predict, detect, and optimize city traffic!")

data_load_state = st.text('Loading and preprocessing data...')
df = load_and_preprocess_data()
models, metrics, scaler, feature_cols = train_models(df)
data_load_state.empty() # clear loading text

# --- SIDEBAR NAV ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["1️⃣ Dataset & EDA", "2️⃣ ML Prediction Engine", "3️⃣ Unique Innovators (Dashboards)"])

st.sidebar.markdown("---")
st.sidebar.subheader("Model Validation (R²)")
for m, v in metrics.items():
    st.sidebar.write(f"**{m}**: {v['R2']:.3f}")

# ----------------- PAGE 1: EDA -----------------
if page == "1️⃣ Dataset & EDA":
    st.header("📊 Dataset Overview & Visualizations")
    st.write(df.head())
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Traffic vs Hour of Day")
        hourly_data = df.groupby('hour')['traffic_volume'].median().reset_index()
        st.line_chart(hourly_data, x='hour', y='traffic_volume')
        
    with col2:
        st.subheader("Weather vs Traffic Impact")
        weather_data = df.groupby('weather_main')['traffic_volume'].mean().reset_index()
        st.bar_chart(weather_data, x='weather_main', y='traffic_volume')

    st.subheader("Heatmap: Day vs Hour Traffic")
    day_hour_data = df.groupby(['day_of_week', 'hour'])['traffic_volume'].mean().reset_index()
    day_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
    day_hour_data['day_of_week'] = day_hour_data['day_of_week'].map(day_map)
    heatmap = alt.Chart(day_hour_data).mark_rect().encode(
        x='hour:O',
        y=alt.Y('day_of_week:N', sort=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']),
        color=alt.Color('traffic_volume:Q', scale=alt.Scale(scheme='inferno'))
    ).properties(height=400)
    st.altair_chart(heatmap, use_container_width=True)

# ----------------- PAGE 2: PREDICTION ENGINE -----------------
elif page == "2️⃣ ML Prediction Engine":
    st.header("🤖 Real-Time Traffic Prediction")
    st.write("Input current conditions to predict congestion volume in real-time.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        time_input = st.time_input("Time of Travel", datetime.time(8, 0))
        day_input = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        day_map = {"Monday":0, "Tuesday":1, "Wednesday":2, "Thursday":3, "Friday":4, "Saturday":5, "Sunday":6}
        day_num = day_map[day_input]
    
    with col2:
        temp_input = st.slider("Temperature (Kelvin)", 240.0, 310.0, 290.0)
        weather_input = st.selectbox("Weather Condition", ['Clear', 'Clouds', 'Rain', 'Drizzle', 'Mist', 'Haze', 'Fog', 'Snow', 'Thunderstorm', 'Squall', 'Smoke'])
        severity = {'Clear': 0, 'Clouds': 1, 'Rain': 2, 'Drizzle': 2, 'Mist': 2, 'Haze': 2, 'Fog': 3, 'Snow': 4, 'Thunderstorm': 5, 'Squall': 5, 'Smoke': 4}
        sev_val = severity[weather_input]
        
    with col3:
        month_num = st.slider("Month", 1, 12, 6)
        lag_input = st.number_input("Prior Hour Traffic Volume", min_value=0, value=3000)
        
    # Logic
    model_choice = st.selectbox("Choose Model Architecture", ["XGBoost", "Random Forest", "Linear Regression"])
    
    hour_val = time_input.hour
    is_peak = 1 if (8 <= hour_val <= 11) or (17 <= hour_val <= 21) else 0
    
    if st.button("Predict Traffic Volume"):
        input_data = pd.DataFrame([[temp_input, 0.0, 0.0, 50.0, hour_val, day_num, month_num, is_peak, sev_val, lag_input]], columns=feature_cols)
        input_scaled = scaler.transform(input_data)
        prediction = models[model_choice].predict(input_scaled)[0]
        
        # Output
        st.success(f"### 🚗 Predicted Traffic Volume: {int(prediction)} vehicles/hour")
        
        # Small breakdown
        st.write(f"**Model Used**: {model_choice} (Baseline R²: {metrics[model_choice]['R2']:.3f})")

# ----------------- PAGE 3: UNIQUE INNOVATORS -----------------
elif page == "3️⃣ Unique Innovators (Dashboards)":
    st.header("💡 Advanced Optimization Dashboard")
    st.write("Demonstrating the unique, high-level innovations from the architecture plan!")
    st.divider()
    
    # We will grab a random sample prediction to drive the demo automatically
    st.subheader("1. Demo Feed (Live Snapshot Generator)")
    if st.button("Generate Random Live Situation"):
        sample_data = df.sample(1)
        hr = int(sample_data['hour'].iloc[0])
        wx = sample_data['weather_main'].iloc[0]
        
        # Process it through XGBoost to get the "live" prediction
        samp_x = sample_data[feature_cols]
        pred_vol = int(models['XGBoost'].predict(scaler.transform(samp_x))[0])
        
        st.write(f"**Current Hour:** {hr}:00 | **Weather:** {wx} | **Baseline Volume Pred:** {pred_vol}")
        
        st.divider()
        col1, col2 = st.columns(2)
        
        # --- IDEA 2: Traffic Mood Index ---
        with col1:
            st.markdown("### 🌟 Traffic Mood Index")
            if pred_vol < 2000:
                st.markdown("<h1 style='text-align: center; font-size: 80px;'>😄</h1>", unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: center; color: green;'>Smooth Sailing (Score: 10/10)</h3>", unsafe_allow_html=True)
            elif pred_vol < 4500:
                st.markdown("<h1 style='text-align: center; font-size: 80px;'>😐</h1>", unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: center; color: orange;'>Moderate Congestion (Score: 6/10)</h3>", unsafe_allow_html=True)
            else:
                st.markdown("<h1 style='text-align: center; font-size: 80px;'>😡</h1>", unsafe_allow_html=True)
                st.markdown("<h3 style='text-align: center; color: red;'>Heavy Traffic! (Score: 2/10)</h3>", unsafe_allow_html=True)

        # --- IDEA 6: Weather + Traffic Fusion Model Alerts ---
        with col2:
            st.markdown("### 🌧️ Weather Fusion Alert")
            if wx in ['Rain', 'Snow', 'Thunderstorm', 'Squall', 'Fog'] and pred_vol > 3500:
                st.error(f"🚨 **CRITICAL ALERT**: Heavy congestion expected due to {wx} conditions at {hr}:00!")
            elif wx in ['Rain', 'Snow'] and pred_vol <= 3500:
                st.warning(f"⚠️ **BE ADVISED**: {wx} conditions present, but traffic is currently manageable.")
            else:
                st.success("✅ Weather is clear or not heavily impacting current flow limits.")

        st.divider()
        # --- IDEA 10: Carbon Emission Estimator ---
        st.markdown("### 🌱 Carbon Emission Estimator")
        st.write("Calculated based on traffic volume density and average idle times.")
        
        # Simple heuristic: heavily congested traffic increases emissions exponentially due to idling
        base_emissions = pred_vol * 0.15 # 0.15 kg CO2 per vehicle per hour
        congestion_multiplier = 1.0 + (pred_vol / 7000) # Scales up to ~1.8x at max capacity
        total_co2 = base_emissions * congestion_multiplier
        
        st.metric(label="Estimated CO₂ Emissions (per hour for this sector)", value=f"{total_co2:.2f} kg", delta=f"{(congestion_multiplier-1)*100:.1f}% Increase from Idling penalty!", delta_color="inverse")
        
        if total_co2 > 1000:
            st.info("💡 **AI Route Suggestion**: Rerouting heavy vehicles to peripheral loops is recommended to drop central city emissions to safe levels.")
    else:
        st.info("Click the button above to generate a live snapshot and view the Advanced Dashboards!")
