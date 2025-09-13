import streamlit as st
import pandas as pd
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px # Import Plotly Express for charts
from datetime import timedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="AQI Health AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Load Model and Artifacts ---
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load('health_risk_model.pkl')
        scaler = joblib.load('scaler.pkl')
        le = joblib.load('le.pkl')
        with open('columns.json', 'r') as f:
            columns = json.load(f)
        return model, scaler, le, columns
    except FileNotFoundError:
        return None, None, None, None

# --- NEW: Load Historical Data ---
@st.cache_data
def load_historical_data():
    try:
        df = pd.read_csv('city_day_filled.csv', parse_dates=['Date'])
        # Clean data: clip AQI at 0 and remove outliers for better visualization
        df['AQI'] = df['AQI'].clip(lower=0)
        df = df[df['AQI'] < 1000] # Remove extreme outliers
        return df
    except FileNotFoundError:
        return None

model, scaler, le, columns = load_artifacts()
historical_df = load_historical_data()

if not all([model, scaler, le, columns]):
    st.error("Fatal Error: Model or necessary artifact files not found. Please ensure 'health_risk_model.pkl', 'scaler.pkl', 'le.pkl', and 'columns.json' are in the root directory.")
    st.stop()
if historical_df is None:
    st.error("Fatal Error: 'city_day_filled.csv' not found. This file is required for historical trend analysis.")
    st.stop()


# --- Custom CSS (same as before) ---
css = """
/* General Styling */
.stApp {
    background-image: linear-gradient(to bottom right, #000428, #004e92);
    color: #FAFAFA;
}
.stButton>button {
    border: 2px solid #00C4FF;
    border-radius: 25px;
    padding: 12px 24px;
    background-image: linear-gradient(to right, #00C4FF, #4B8BBE);
    color: white;
    font-weight: bold;
    transition: all 0.3s ease-in-out;
    box-shadow: 0 4px 15px rgba(0, 196, 255, 0.2);
    width: 100%;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 196, 255, 0.4);
}
.stNumberInput input, .stSelectbox [data-baseweb="select"] {
    background-color: rgba(255, 255, 255, 0.08);
    color: #FAFAFA;
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.2);
}
h1, h2, h3, h4, h5, h6 {
    color: #E0E0E0;
    text-shadow: 0 0 5px rgba(0,0,0,0.5);
}
/* Glassmorphism Containers */
.glass-container {
    background: rgba(40, 60, 80, 0.5);
    border-radius: 16px;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    padding: 2rem;
    margin-top: 1rem;
}
/* Custom Metric Styling */
.metric-container {
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    text-align: center;
}
.metric-container-HighRisk { background-color: rgba(255, 75, 75, 0.2); border-left: 6px solid #FF4B4B; }
.metric-container-MediumRisk { background-color: rgba(255, 165, 0, 0.2); border-left: 6px solid #FFA500; }
.metric-container-LowRisk { background-color: rgba(75, 181, 67, 0.2); border-left: 6px solid #4BB543; }
.metric-label { font-size: 1.1rem; color: #b0b0b0; text-transform: uppercase; }
.metric-value { font-size: 3rem; font-weight: bold; color: #FAFAFA; }
.stProgress > div > div > div > div {
    background-image: linear-gradient(to right, #4B8BBE , #00C4FF);
}
/* Make tabs look better */
div[data-baseweb="tab-list"] {
		gap: 24px;
}
div[data-baseweb="tab-list"] button {
    background-color: transparent;
    border-radius: 8px;
}
div[data-baseweb="tab-list"] button[aria-selected="true"] {
    background-color: rgba(0, 196, 255, 0.2);
}
"""
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# --- Helper Functions (same as before) ---
def get_aqi_status(aqi_val):
    if aqi_val <= 50: return "Good", "lime"
    if aqi_val <= 100: return "Satisfactory", "green"
    if aqi_val <= 200: return "Moderate", "orange"
    if aqi_val <= 300: return "Poor", "red"
    if aqi_val <= 400: return "Very Poor", "violet"
    return "Severe", "maroon"

def get_advice_and_info(risk_level, inputs, is_sensitive):
    advice, high_pollutants = "", []
    if inputs['PM2.5'][0] > 50: high_pollutants.append(f"**High PM2.5 ({inputs['PM2.5'][0]} µg/m³):** These fine particles can penetrate deep into the lungs.")
    if inputs['O3'][0] > 100: high_pollutants.append(f"**High Ozone (O3) ({inputs['O3'][0]} µg/m³):** Can cause breathing problems and trigger asthma.")
    if inputs['NO2'][0] > 40: high_pollutants.append(f"**High NO2 ({inputs['NO2'][0]} µg/m³):** Can irritate airways and aggravate respiratory diseases.")
    if inputs['CO'][0] > 4: high_pollutants.append(f"**High CO ({inputs['CO'][0]} mg/m³):** A toxic gas that reduces oxygen delivery in the body.")
    sensitive_advice = " **Sensitive Groups (Children, Elderly, Respiratory/Heart Patients):**"
    if risk_level == 'High Risk':
        advice = "### 🚨 **Precautions: High Risk**\n- **General Population:** Avoid all outdoor physical activity. Keep windows and doors closed. Use air purifiers.\n"
        advice += f"{sensitive_advice} Remain indoors and keep activity levels minimal. Monitor symptoms closely."
    elif risk_level == 'Medium Risk':
        advice = "### ⚠️ **Precautions: Medium Risk**\n- **General Population:** Reduce prolonged or strenuous exertion outdoors.\n"
        advice += f"{sensitive_advice} Avoid strenuous outdoor activities. Wear an N95 mask if you must be outside for extended periods."
        if is_sensitive: advice += "\n\n- **Personalized Tip:** As you've identified as part of a sensitive group, consider rescheduling non-essential outdoor errands."
    else:
        advice = "### ✅ **Precautions: Low Risk**\n- **General Population:** Air quality is good. It's a great time for outdoor activities.\n"
        advice += f"{sensitive_advice} Enjoy the fresh air! No specific precautions are necessary."
    if high_pollutants:
        advice += f"\n\n**Key Contributors to Poor Air Quality:**\n"
        for pollutant in high_pollutants: advice += f"- {pollutant}\n"
    return advice

def create_pollutant_radar_chart(values):
    # (Function is the same)
    pollutants = ['PM2.5', 'PM10', 'NO2', 'O3', 'SO2', 'CO']
    safe_levels = [50, 100, 40, 100, 20, 4]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=[values.get(p, 0) for p in pollutants], theta=pollutants, fill='toself', name='Current Levels', line=dict(color='cyan')))
    fig.add_trace(go.Scatterpolar(r=safe_levels, theta=pollutants, fill='toself', name='Safe Levels', line=dict(color='rgba(0, 255, 0, 0.4)'), fillcolor='rgba(0, 255, 0, 0.1)'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, max(max(safe_levels), max([values.get(p, 0) for p in pollutants])) * 1.1])), showlegend=True, template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(x=0.5, y=-0.1, xanchor="center", orientation="h"), font=dict(color="white"))
    return fig

# --- UI Layout ---
st.title('Air Quality Health Risk Prediction System')
st.caption('Predictive insights into air quality health risks with interactive analysis.')

col1, col2 = st.columns([0.6, 0.4])

with col1:
    with st.container():
        cities = sorted(historical_df['City'].unique().tolist())
        main_c1, main_c2 = st.columns(2)
        city = main_c1.selectbox('**Select City**', cities, index=cities.index('Delhi'))
        aqi = main_c2.number_input('**Air Quality Index (AQI)**', min_value=0, max_value=500, value=155, step=1)
        aqi_status, aqi_color = get_aqi_status(aqi)
        st.info(f"AQI Status: **:{aqi_color}[{aqi_status}]**")
        st.write("---")
        st.subheader("Pollutant Concentrations")
        c1, c2, c3 = st.columns(3)
        pm25 = c1.number_input('PM2.5 (μg/m³)', 0.0, 1000.0, 55.5)
        pm10 = c2.number_input('PM10 (μg/m³)', 0.0, 1000.0, 110.0)
        nh3 = c3.number_input('NH3 (μg/m³)', 0.0, 400.0, 30.0)
        no = c1.number_input('NO (μg/m³)', 0.0, 400.0, 18.0)
        no2 = c2.number_input('NO2 (μg/m³)', 0.0, 400.0, 35.0)
        nox = c3.number_input('NOx (ppb)', 0.0, 500.0, 45.0)
        co = c1.number_input('CO (mg/m³)', 0.0, 200.0, 2.5)
        so2 = c2.number_input('SO2 (μg/m³)', 0.0, 200.0, 18.0)
        o3 = c3.number_input('O3 (μg/m³)', 0.0, 300.0, 40.0)
        benzene = c1.number_input('Benzene (μg/m³)', 0.0, 500.0, 3.5)
        toluene = c2.number_input('Toluene (μg/m³)', 0.0, 500.0, 9.0)
        xylene = c3.number_input('Xylene (μg/m³)', 0.0, 200.0, 3.5)

with col2:
    st.subheader("Personalize Your Advice")
    is_sensitive_group = st.toggle("I belong to a sensitive group", help="Enable for tailored advice for the elderly, children, or individuals with respiratory/heart conditions.")
    if is_sensitive_group:
        st.success("✅ Advice will be tailored for sensitive individuals.")
    st.write("")
    st.write("")
    predict_button = st.button('Forecast Health Risk', use_container_width=True)

# --- Main Panel for Output ---
if predict_button:
    input_dict = {'PM2.5': [pm25], 'PM10': [pm10], 'NO': [no], 'NO2': [no2], 'NOx': [nox], 'NH3': [nh3], 'CO': [co], 'SO2': [so2], 'O3': [o3], 'Benzene': [benzene], 'Toluene': [toluene], 'Xylene': [xylene], 'AQI': [float(aqi)], 'City': [city]}
    input_data = pd.DataFrame(input_dict)
    input_data_encoded = pd.get_dummies(input_data, columns=['City'])
    input_data_reindexed = input_data_encoded.reindex(columns=columns, fill_value=0)
    input_data_scaled = scaler.transform(input_data_reindexed)
    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)
    risk_level = le.inverse_transform(prediction)[0]
    
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    # --- UPDATED: Added new tab for historical data ---
    tab1, tab2, tab3, tab4 = st.tabs(["📊 **Dashboard**", "📉 **Historical Trends & Forecasting**", "🩺 **Health Advisory**", "🔬 **Scenario Planner**"])

    with tab1:
        st.header("Prediction Dashboard")
        risk_color_class = risk_level.replace(" ", "")
        st.markdown(f'<div class="metric-container metric-container-{risk_color_class}"><div class="metric-label">Predicted Health Risk</div><div class="metric-value">{risk_level}</div></div>', unsafe_allow_html=True)
        
        # --- UPDATED: Compact confidence scores ---
        st.write("#### Prediction Confidence")
        cols_confidence = st.columns(len(le.classes_))
        for i, class_label in enumerate(le.classes_):
            with cols_confidence[i]:
                prob = prediction_proba[0][i]
                st.metric(label=class_label, value=f"{prob:.1%}")
                st.progress(prob)
        
        st.write("---")

        # --- UPDATED: Gauge and Radar Chart in one row ---
        viz_col1, viz_col2 = st.columns(2)
        with viz_col1:
            st.write("#### AQI Gauge")
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = aqi,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': aqi_status, 'font': {'size': 20}},
                gauge = {
                    'axis': {'range': [None, 500], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': aqi_color},
                    'steps' : [
                        {'range': [0, 50], 'color': 'lime'},
                        {'range': [50, 100], 'color': 'green'},
                        {'range': [100, 200], 'color': 'yellow'},
                        {'range': [200, 300], 'color': 'orange'},
                        {'range': [300, 400], 'color': 'red'},
                        {'range': [400, 500], 'color': 'maroon'}]
                }))
            fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white", 'family': "Arial"}, height=280, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with viz_col2:
            st.write("#### Pollutant Profile")
            radar_values = {'PM2.5': pm25, 'PM10': pm10, 'NO2': no2, 'O3': o3, 'SO2': so2, 'CO': co}
            fig_radar = create_pollutant_radar_chart(radar_values)
            fig_radar.update_layout(height=280, margin=dict(l=40, r=40, t=50, b=10))
            st.plotly_chart(fig_radar, use_container_width=True)

    with tab2: # --- NEW FEATURE TAB ---
        st.header(f"Historical Trends & Forecasting for {city}")
        city_data = historical_df[historical_df['City'] == city].sort_values('Date')
        
        if not city_data.empty:
            # --- Time Period Selection ---
            days = st.radio("Select Time Period:", (7, 30), horizontal=True)
            
            # --- Historical Trend Chart ---
            st.subheader(f"AQI Trend for the Last {days} Days")
            end_date = city_data['Date'].max()
            start_date = end_date - timedelta(days=days-1)
            trend_data = city_data[(city_data['Date'] >= start_date) & (city_data['Date'] <= end_date)]
            
            fig_trend = px.line(trend_data, x='Date', y='AQI', title=f'Daily AQI for {city}', markers=True)
            fig_trend.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # --- Simple Forecasting ---
            st.subheader("Next-Day AQI Forecast")
            if len(trend_data) >= 3:
                # Use a 3-day moving average for a simple forecast
                last_3_days_avg = trend_data['AQI'].tail(3).mean()
                forecast_aqi = round(last_3_days_avg, 2)
                
                # Compare to the last known value
                last_known_aqi = trend_data['AQI'].iloc[-1]
                delta = round(forecast_aqi - last_known_aqi, 2)
                
                st.metric(label="Forecasted AQI for Tomorrow", value=forecast_aqi, delta=f"{delta} vs Last Day")
                status, _ = get_aqi_status(forecast_aqi)
                st.info(f"The forecasted air quality is expected to be **{status}**.")
            else:
                st.warning("Not enough historical data available for this city to generate a forecast.")
        else:
            st.warning(f"No historical data found for {city}.")

    with tab3:
        st.header("Personalized Health Advisory")
        advice_text = get_advice_and_info(risk_level, input_dict, is_sensitive_group)
        st.markdown(advice_text)

    with tab4:
        st.header("Interactive Scenario Planner")
        st.info("Adjust the sliders for key pollutants to see how the health risk might change.")
        what_if_pm25 = st.slider("What if PM2.5 was...", 0.0, 200.0, pm25)
        what_if_aqi = st.slider("What if AQI was...", 0, 500, aqi)
        what_if_dict = input_dict.copy(); what_if_dict['PM2.5'] = [what_if_pm25]; what_if_dict['AQI'] = [float(what_if_aqi)]; what_if_data = pd.DataFrame(what_if_dict)
        what_if_encoded = pd.get_dummies(what_if_data, columns=['City']); what_if_reindexed = what_if_encoded.reindex(columns=columns, fill_value=0); what_if_scaled = scaler.transform(what_if_reindexed)
        what_if_prediction = model.predict(what_if_reindexed); what_if_risk = le.inverse_transform(what_if_prediction)[0]
        st.write(""); st.metric(label="Potential Health Risk", value=what_if_risk, delta=f"From {risk_level}" if what_if_risk != risk_level else "No Change")
        if what_if_risk == 'Low Risk' and risk_level != 'Low Risk': st.success("This scenario significantly improves the health outlook!")
        elif what_if_risk == 'Medium Risk' and risk_level == 'High Risk': st.warning("This scenario shows a moderate improvement.")

    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Configure the parameters on the left and click 'Forecast Health Risk' to generate an analysis.")

