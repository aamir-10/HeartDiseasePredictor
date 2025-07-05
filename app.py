import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Loading of the model
model = joblib.load("heart_model.pkl")

# Page Configuration with some inline css.
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.markdown("""
<style>
.stButton > button {
    background-color: transparent !important;
    color: inherit !important;
    border: 1px solid #ccc;
    padding: 0.5rem 1rem;
    border-radius: 6px;
    transition: 0.3s ease;
}
.stButton > button:hover {
    background-color: transparent !important;
    color: white !important;
    border-color: red !important;
}
</style>
""", unsafe_allow_html=True)

# Setting the App Header
st.title("💓 Heart Disease Prediction App")
st.markdown("*From data to diagnosis — Uncover your heart disease risk instantly!* ")

# Sidebar Information
with st.sidebar:
    st.title("📘 About")
    st.markdown("*Smart Health at your fingertips!*")
    st.info("""
        - Predicts heart disease using an advanced **Random Forest** model  
        - Combines **clinical features** and **AI insights** to assess your heart health  
        - Instant results with **visually rich** indicators  
        - Designed for **educational and awareness** purposes 
    """)
    st.markdown("---")
    st.markdown("~ Input your data to get started!")

# User Input Label
st.header("🧾 Enter Patient Details")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 100, value=None)
    sex = st.radio("Sex", ["Male", "Female"], index=None)
    cp = st.selectbox("Chest Pain Type", ["0: Typical", "1: Atypical", "2: Non-anginal", "3: Asymptomatic"], index=0)
    trtbps = st.slider("Resting Blood Pressure", 80, 200, value=None)
    chol = st.slider("Cholesterol", 100, 400, value=None)
    thall = st.selectbox("Thalassemia", ["1: Normal", "2: Fixed", "3: Reversible"], index=0)
    fbs = st.radio("Fasting Blood Sugar > 120", ["Yes", "No"], index=None)

with col2:
    restecg = st.selectbox("Resting ECG", ["0: Normal", "1: ST-T wave abnormality", "2: LVH"], index=0)
    thalachh = st.slider("Max Heart Rate", 60, 220, value=None)
    exng = st.radio("Exercise Induced Angina", ["Yes", "No"], index=None)
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0,value=None)
    slp = st.selectbox("Slope", ["0: Upsloping", "1: Flat", "2: Downsloping"], index=0)
    caa = st.selectbox("Major Vessels", [0, 1, 2, 3], index=0)
    oxy = st.slider("Oxygen Saturation", 90.0, 100.0,value=None)


# Encode Input Feaures
input_data = {
    "age": age,
    "sex": 1 if sex == "Male" else 0,
    "cp": int(cp[0]),
    "trtbps": trtbps,
    "chol": chol,
    "fbs": 1 if fbs == "Yes" else 0,
    "restecg": int(restecg[0]),
    "thalachh": thalachh,
    "exng": 1 if exng == "Yes" else 0,
    "oldpeak": oldpeak,
    "slp": int(slp[0]),
    "caa": caa,
    "thall": int(thall[0]),
    "Oxygen Saturation": oxy
}

input_df = pd.DataFrame([input_data])

# Showing of Input Summary of the Patient
st.markdown("### 📋 Patient Summary")
st.dataframe(input_df)

if st.button("🔍 Predict Risk"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1] * 100

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease Detected!")
    else:
        st.success("✅ Low Risk of Heart Disease.")
        st.balloons()

    # Gauge Chart for visualization of the risk meter.
    st.subheader("🧪 Predicted Risk Level")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        number={'suffix': '%'},
        title={'text': "Heart Disease Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#113A99"},
            'steps': [
                {'range': [0, 50], 'color': "green"},
                {'range': [50, 80], 'color': "orange"},
                {'range': [80, 100], 'color': "red"}
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)

# Vertical Bar Chart: User vs Ideal
st.markdown("---")
st.subheader("📌 Patient Metrics vs Standard Ranges")

labels = ["Resting BP", "Cholesterol", "Max Heart Rate", "Oxygen Saturation"]
user_values = [trtbps, chol, thalachh, oxy]
healthy_values = [120, 180, 170, 98]  # medically sound reference values

comparison_df = pd.DataFrame({
    "Metric": labels,
    "User": user_values,
    "Healthy": healthy_values
})

fig_bar = go.Figure()

# User values bar
fig_bar.add_trace(go.Bar(
    x=comparison_df["Metric"],
    y=comparison_df["User"],
    name='Your Value',
    marker=dict(
        color='rgba(250, 3, 3, 0.85)',
        line=dict(color='rgba(192, 57, 43, 1)', width=2)
    )
))

# Healthy values bar
fig_bar.add_trace(go.Bar(
    x=comparison_df["Metric"],
    y=comparison_df["Healthy"],
    name='Ideal Range',
    marker=dict(
        color='rgba(50, 250, 96, 0.8)',  # healthy green
        line=dict(color='rgba(30, 132, 73, 1)', width=2)
    )
))

# Layout tweaks
fig_bar.update_layout(
    barmode='group',
    template="plotly_white",
    height=430,
    margin=dict(t=20, b=80),
    xaxis_title="",
    yaxis_title="",
    legend=dict(
        orientation="h",
        yanchor="top",
        y=-0.25,
        xanchor="center",
        x=0.5,
        font=dict(size=13)
    )
)

# Custom toolbar configuration
config = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "select2d",
        "lasso2d",
        "autoScale2d",
        "resetScale2d",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
        "toggleSpikelines",
        "zoom2d",
        "pan2d"
    ],
    "responsive": True
}


st.plotly_chart(fig_bar, use_container_width=True, config=config)

# Heart Rate vs Age Plot
st.subheader("🫀 Heart Rate Comparison with Age")

# Calculate ideal max heart rate through the below formula
ideal_max_hr = 220 - age

fig_hr, ax_hr = plt.subplots(figsize=(6, 4))

# Plotting of user's actual max heart rate
ax_hr.plot([age], [thalachh], marker='o', markersize=10, color='red', label="Your Max HR")

# Plotting of ideal max heart rate line
ages = np.arange(18, 101)
ideal_rates = 220 - ages
ax_hr.plot(ages, ideal_rates, linestyle='--', color='blue', label="Ideal Max HR")

# Plot formatting
ax_hr.set_title("Max Heart Rate vs Age", fontsize=14)
ax_hr.set_xlabel("Age")
ax_hr.set_ylabel("Max Heart Rate (bpm)")
ax_hr.set_xlim(18, 100)
ax_hr.set_ylim(60, 220)
ax_hr.legend()
ax_hr.grid(True)
st.pyplot(fig_hr)

# Line Chart: Chest Pain Type vs Predicted Risk
st.markdown("---")
st.subheader("💉 Risk Variation Across Chest Pain Types")

cp_labels = {
    "Typical Angina": 0,
    "Atypical Angina": 1,
    "Non-anginal Pain": 2,
    "Asymptomatic": 3
}

marker_colors = {
    "Typical Angina": "#e74c3c",       
    "Atypical Angina": "#3498db",      
    "Non-anginal Pain": "#f1c40f",     
    "Asymptomatic": "#2ecc71"          
}

risk_dict = {}
for label, cp_val in cp_labels.items():
    temp_data = input_data.copy()
    temp_data["cp"] = cp_val
    temp_df = pd.DataFrame([temp_data])
    risk = model.predict_proba(temp_df)[0][1] * 100
    risk_dict[label] = round(risk, 2)

fig_line_cp = go.Figure()

fig_line_cp.add_trace(go.Scatter(
    x=list(risk_dict.keys()),
    y=list(risk_dict.values()),
    mode='lines',
    line=dict(shape='spline', color="#822FA6", width=3),
    name="Risk Line"
))

# Add colored markers for each CP type
for label in cp_labels.keys():
    fig_line_cp.add_trace(go.Scatter(
        x=[label],
        y=[risk_dict[label]],
        mode='markers',
        marker=dict(size=10, color=marker_colors[label]),
        name=label
    ))

fig_line_cp.update_layout(
    xaxis_title="Chest Pain Type",
    yaxis_title="Predicted Risk (%)",
    template="plotly_white",
    height=400,
    margin=dict(t=30, b=60)
)

config = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': [
        'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'autoScale2d', 'resetScale2d',
        'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleSpikelines'
    ],
    'modeBarButtonsToAdd': [
        'zoomIn2d', 'zoomOut2d', 'toImage', 'toggleFullscreen'
    ]
}

st.plotly_chart(fig_line_cp, use_container_width=True, config=config)

# Feature Importance Graph 
st.markdown("---")
st.subheader("💡 Feature Importance (Random Forest)")
try:
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": input_df.columns,
        "Importance": importances
    }).sort_values("Importance")

    cmap = plt.cm.plasma
    colors = cmap(feat_df["Importance"] / max(feat_df["Importance"])) # Normalization of Values

    fig2, ax = plt.subplots()
    ax.barh(feat_df["Feature"], feat_df["Importance"], color=colors)
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance")
    plt.tight_layout()
    st.pyplot(fig2)

except Exception as e:
    st.info("Feature importance not available for this model.")

# Custom CSS for styling the app's background and Sidebar.

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(150deg, #330000, #000000);
        background-attachment: fixed;
        background-size: cover;
        color: white;
    }
    h1, h2, h3, h4, h5, h6, .stMarkdown, .st-bb, .st-cb {
        color: #ffffff;
    }
    .stSelectbox, .stSlider, .stRadio, .stNumberInput, .stTextInput, .stDataFrame, .stButton {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: white !important;
    }
    .css-1v0mbdj p {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        background: linear-gradient(150deg, #550000, #1a0000);
        box-shadow: 2px 0 6px rgba(0, 0, 0, 0.6);
    }
    </style>
""", unsafe_allow_html=True)

# --- End of the program ---