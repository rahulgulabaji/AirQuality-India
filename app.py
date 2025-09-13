import streamlit as st
import pandas as pd
import joblib
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="AQI Health Forecaster",
    page_icon="🌬️",
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

model, scaler, le, columns = load_artifacts()

if not all([model, scaler, le, columns]):
    st.error("Fatal Error: Model or necessary artifact files not found.")
    st.stop()

# --- Custom CSS for a Polished UI ---
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
    margin-bottom: 1rem;
}
/* Custom Metric Styling */
.metric-container {
    padding: 1.5rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    text-align: center;
}
.metric-container-High { background-color: rgba(255, 75, 75, 0.2); border-left: 6px solid #FF4B4B; }
.metric-container-Medium { background-color: rgba(255, 165, 0, 0.2); border-left: 6px solid #FFA500; }
.metric-container-Low { background-color: rgba(75, 181, 67, 0.2); border-left: 6px solid #4BB543; }

.metric-label { font-size: 1.1rem; color: #b0b0b0; text-transform: uppercase; }
.metric-value { font-size: 3rem; font-weight: bold; color: #FAFAFA; }
.stProgress > div > div > div > div {
    background-image: linear-gradient(to right, #4B8BBE , #00C4FF);
}
.stExpander {
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    background: rgba(40, 60, 80, 0.3);
}
"""
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

# --- Data & Helper Functions ---
city_averages = {
    'Delhi': {'PM2.5': 119.9, 'AQI': 260.1}, 'Ahmedabad': {'PM2.5': 68.6, 'AQI': 199.1},
    'Bengaluru': {'PM2.5': 42.2, 'AQI': 98.7}, 'Mumbai': {'PM2.5': 45.9, 'AQI': 105.3},
    'Kolkata': {'PM2.5': 67.5, 'AQI': 148.8}, 'Chennai': {'PM2.5': 39.8, 'AQI': 97.6},
}

def get_aqi_status(aqi_val):
    if aqi_val <= 50: return "Good", "lime"
    if aqi_val <= 100: return "Satisfactory", "green"
    if aqi_val <= 200: return "Moderate", "orange"
    if aqi_val <= 300: return "Poor", "red"
    if aqi_val <= 400: return "Very Poor", "violet"
    return "Severe", "maroon"

def get_advice_and_info(risk_level, inputs):
    advice, high_pollutants = "", []
    if inputs['PM2.5'][0] > 50: high_pollutants.append(f"**High PM2.5 ({inputs['PM2.5'][0]} µg/m³):** These fine particles can penetrate deep into the lungs and are a major health concern.")
    if inputs['O3'][0] > 100: high_pollutants.append(f"**High Ozone (O3) ({inputs['O3'][0]} µg/m³):** Ground-level ozone can cause breathing problems, trigger asthma, and harm lung tissue.")
    if inputs['NO2'][0] > 40: high_pollutants.append(f"**High NO2 ({inputs['NO2'][0]} µg/m³):** Nitrogen Dioxide can irritate airways and aggravate respiratory diseases.")
    if inputs['CO'][0] > 4: high_pollutants.append(f"**High CO ({inputs['CO'][0]} mg/m³):** Carbon Monoxide is a toxic gas that can reduce oxygen delivery to the body's organs.")

    if risk_level == 'High Risk': advice = "### 🚨 **Precautions: High Risk**\n- **General Population:** Significantly reduce or avoid outdoor physical activity. Keep windows and doors closed. Use air purifiers with HEPA filters if available.\n- **Sensitive Groups (Children, Elderly, Respiratory/Heart Patients):** Remain indoors and keep activity levels minimal.\n"
    elif risk_level == 'Medium Risk': advice = "### ⚠️ **Precautions: Medium Risk**\n- **General Population:** Reduce prolonged or strenuous exertion outdoors, especially during peak pollution hours.\n- **Sensitive Groups:** Avoid strenuous outdoor activities. Consider wearing a mask (N95) if you need to be outside for an extended period.\n"
    else: advice = "### ✅ **Precautions: Low Risk**\n- **General Population:** Air quality is good. It's an excellent time for all outdoor activities.\n- **Sensitive Groups:** Enjoy the fresh air! No specific precautions are necessary.\n"

    if high_pollutants:
        advice += f"\n**Key Contributors Identified:**\n"
        for pollutant in high_pollutants: advice += f"- {pollutant}\n"

    return advice

# --- UI Layout ---
st.title('AQI Health Forecaster 🌬️')
st.caption('An intelligent system to predict air quality health risks based on pollutant data.')

with st.expander("📍 Configure Input Parameters", expanded=True):
    with st.container():
        cities = sorted(list(city_averages.keys()) + ['Aizawl', 'Amaravati', 'Amritsar', 'Bhopal', 'Chandigarh', 'Coimbatore', 'Ernakulam', 'Gurugram', 'Guwahati', 'Hyderabad', 'Jaipur', 'Jorapokhar', 'Kochi', 'Lucknow', 'Patna', 'Shillong', 'Talcher', 'Thiruvananthapuram', 'Visakhapatnam'])
        
        main_c1, main_c2 = st.columns([1,2])
        with main_c1:
            city = st.selectbox('**City**', cities, index=cities.index('Delhi'))
            # --- UPDATED: Replaced slider with number_input for precision ---
            aqi = st.number_input('**Air Quality Index (AQI)**', min_value=0, max_value=500, value=155, step=1)
            aqi_status, aqi_color = get_aqi_status(aqi)
            st.info(f"AQI Status: **:{aqi_color}[{aqi_status}]**")
        
        with main_c2:
            c1, c2, c3 = st.columns(3)
            pm25 = c1.number_input('PM2.5', 0.0, 1000.0, 55.5, help="Particulate Matter 2.5 (μg/m³)")
            pm10 = c2.number_input('PM10', 0.0, 1000.0, 110.0, help="Particulate Matter 10 (μg/m³)")
            nh3 = c3.number_input('NH3', 0.0, 400.0, 30.0, help="Ammonia (μg/m³)")
            
            no = c1.number_input('NO', 0.0, 400.0, 18.0, help="Nitric Oxide (μg/m³)")
            no2 = c2.number_input('NO2', 0.0, 400.0, 35.0, help="Nitrogen Dioxide (μg/m³)")
            nox = c3.number_input('NOx', 0.0, 500.0, 45.0, help="Nitrogen Oxides (ppb)")
            
            co = c1.number_input('CO', 0.0, 200.0, 2.5, help="Carbon Monoxide (mg/m³)")
            so2 = c2.number_input('SO2', 0.0, 200.0, 18.0, help="Sulphur Dioxide (μg/m³)")
            o3 = c3.number_input('O3', 0.0, 300.0, 40.0, help="Ozone (μg/m³)")
            
            benzene = c1.number_input('Benzene', 0.0, 500.0, 3.5, help="μg/m³")
            toluene = c2.number_input('Toluene', 0.0, 500.0, 9.0, help="μg/m³")
            xylene = c3.number_input('Xylene', 0.0, 200.0, 3.5, help="μg/m³")

        st.write("") # Spacer
        predict_button = st.button('Forecast Health Risk')

# --- Main Panel for Output ---
if predict_button:
    input_dict = {
        'PM2.5': [pm25], 'PM10': [pm10], 'NO': [no], 'NO2': [no2],
        'NOx': [nox], 'NH3': [nh3], 'CO': [co], 'SO2': [so2],
        'O3': [o3], 'Benzene': [benzene], 'Toluene': [toluene],
        'Xylene': [xylene], 'AQI': [float(aqi)], 'City': [city]
    }
    input_data = pd.DataFrame(input_dict)
    input_data_encoded = pd.get_dummies(input_data, columns=['City'])
    input_data_reindexed = input_data_encoded.reindex(columns=columns, fill_value=0)
    input_data_scaled = scaler.transform(input_data_reindexed)

    prediction = model.predict(input_data_scaled)
    prediction_proba = model.predict_proba(input_data_scaled)

    risk_level = le.inverse_transform(prediction)[0]
    probabilities = prediction_proba

    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.header("📈 Prediction Analysis")
    
    risk_color_class = risk_level.replace(" ", "")
    st.markdown(f'<div class="metric-container metric-container-{risk_color_class}"><div class="metric-label">Predicted Health Risk</div><div class="metric-value">{risk_level}</div></div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.write("#### Prediction Confidence")
        for i, class_label in enumerate(le.classes_):
            prob = probabilities[0][i]
            st.write(f"{class_label}:")
            st.progress(prob)
    
    with c2:
        st.write("#### Input vs. Historical Average")
        if city in city_averages:
            avg_aqi = city_averages[city]['AQI']
            avg_pm25 = city_averages[city]['PM2.5']
            st.metric(label=f"Your AQI vs. {city} Avg.", value=f"{aqi}", delta=f"{aqi - avg_aqi:.1f}")
            st.metric(label=f"Your PM2.5 vs. {city} Avg.", value=f"{pm25}", delta=f"{pm25 - avg_pm25:.1f}", delta_color="inverse")
        else:
            st.info("Historical averages not available for this city.")
    
    st.write("---")
    advice_text = get_advice_and_info(risk_level, input_dict)
    st.markdown(advice_text)

    # What-If Analysis
    st.write("---")
    st.write("#### What-If Analysis")
    if risk_level == 'High Risk':
        st.info("To potentially reduce the risk to 'Medium', consider scenarios where the **AQI drops below 200** and **PM2.5 is below 50 µg/m³**.")
    elif risk_level == 'Medium Risk':
        st.info("To potentially reduce the risk to 'Low', consider scenarios where the **AQI drops below 100** and **PM2.5 is below 30 µg/m³**.")
    else:
        st.success("The risk level is already low. Great conditions!")
        
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Adjust the parameters in the 'Configure' section above and click 'Forecast' to see the analysis.")

