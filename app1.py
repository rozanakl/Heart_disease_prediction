import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="HeartCare AI | Predictive Health",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    return joblib.load("heart_disease_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error("⚠️ لم يتم العثور على ملف النموذج `heart_disease_model.pkl`. يرجى التأكد من وجوده في نفس المجلد.")

# =========================
# Custom CSS (Styling)
# =========================
st.markdown("""
    <style>
    /* Main Theme & Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header Gradient Banner */
    .header-box {
        background: linear-gradient(90deg, #ec4899 0%, #8b5cf6 50%, #3b82f6 100%);
        padding: 35px 25px;
        border-radius: 20px;
        box-shadow: 0 10px 25px -5px rgba(236, 72, 153, 0.3);
        text-align: center;
        margin-bottom: 30px;
    }
    .header-title {
        color: #ffffff !important;
        font-size: 2.8rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
    }
    .header-sub {
        color: #e2e8f0 !important;
        font-size: 1.1rem;
        margin-top: 8px;
    }

    /* Cards Setup */
    .css-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
    }

    /* Streamlit Form Redesign */
    [data-testid="stForm"] {
        background: rgba(30, 41, 59, 0.6);
        border-radius: 20px;
        border: 1px solid rgba(139, 92, 246, 0.3);
        padding: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }

    /* Section Labels */
    .section-label {
        font-size: 1.2rem;
        font-weight: 700;
        color: #a855f7;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Submit Button Styling */
    .stButton > button {
        background: linear-gradient(90deg, #ec4899 0%, #8b5cf6 100%) !important;
        color: white !important;
        border: none !important;
        padding: 14px 28px !important;
        border-radius: 12px !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(236, 72, 153, 0.4) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(236, 72, 153, 0.6) !important;
    }

    /* Form Inputs Custom Color */
    label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# Header Banner
# =========================
st.markdown("""
    <div class="header-box">
        <h1 class="header-title">🩺 HeartCare AI Dashboard</h1>
        <p class="header-sub">Advanced Machine Learning System for Cardiovascular Risk Assessment</p>
    </div>
""", unsafe_allow_html=True)

# =========================
# Main Layout
# =========================
col_form, col_space, col_info = st.columns([1.8, 0.1, 1.1])

with col_form:
    st.markdown('<div class="section-label">📋 Patient Medical Parameters</div>', unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        f_col1, f_col2 = st.columns(2)
        
        with f_col1:
            Age = st.number_input("👤 Age", min_value=1, max_value=120, value=50)
            Sex = st.selectbox("🚻 Sex", ["M", "F"], format_func=lambda x: "Male (M)" if x == "M" else "Female (F)")
            ChestPainType = st.selectbox("💔 Chest Pain Type", ["TA", "ATA", "NAP", "ASY"], 
                                         help="TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic")
            RestingBP = st.number_input("🩺 Resting BP (mm Hg)", min_value=0, max_value=300, value=120)
            Cholesterol = st.number_input("🧪 Cholesterol (mm/dl)", min_value=0, max_value=700, value=200)
            FastingBS = st.selectbox("🍬 Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")

        with f_col2:
            RestingECG = st.selectbox("📈 Resting ECG Results", ["Normal", "ST", "LVH"])
            MaxHR = st.number_input("⚡ Max Heart Rate", min_value=0, max_value=250, value=150)
            ExerciseAngina = st.selectbox("🏃 Exercise Induced Angina", ["N", "Y"], format_func=lambda x: "Yes (Y)" if x == "Y" else "No (N)")
            Oldpeak = st.number_input("📉 Oldpeak (ST Depression)", value=0.0, step=0.1)
            ST_Slope = st.selectbox("📐 ST Slope", ["Up", "Flat", "Down"])

        st.write("")
        submit = st.form_submit_button("🚀 Run Risk Analysis")

with col_info:
    st.markdown('<div class="section-label">📊 Diagnostic Output</div>', unsafe_allow_html=True)
    
    if submit:
        # Prepare Data Frame
        user_data = pd.DataFrame({
            "Age": [Age],
            "Sex": [Sex],
            "ChestPainType": [ChestPainType],
            "RestingBP": [RestingBP],
            "Cholesterol": [Cholesterol],
            "FastingBS": [FastingBS],
            "RestingECG": [RestingECG],
            "MaxHR": [MaxHR],
            "ExerciseAngina": [ExerciseAngina],
            "Oldpeak": [Oldpeak],
            "ST_Slope": [ST_Slope]
        })

        # Run Prediction
        prediction = model.predict(user_data)[0]
        
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(user_data)[0][1] * 100
        else:
            prob = 100.0 if prediction == 1 else 0.0

        # Gauge Chart Visualization
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            number={'suffix': "%", 'font': {'color': "white", 'size': 40}},
            title={'text': "Calculated Risk Score", 'font': {'size': 18, 'color': "#cbd5e1"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#ec4899" if prob > 50 else "#10b981"},
                'bgcolor': "rgba(255, 255, 255, 0.05)",
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 35], 'color': 'rgba(16, 185, 129, 0.2)'},
                    {'range': [35, 65], 'color': 'rgba(245, 158, 11, 0.2)'},
                    {'range': [65, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
                ],
            }
        ))
        
        fig_gauge.update_layout(
            height=250, 
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': "white"}
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Status Message Card
        if prediction == 1:
            st.markdown("""
                <div style="background: rgba(239, 68, 68, 0.2); border-left: 5px solid #ef4444; padding: 15px; border-radius: 8px; margin-top: 10px;">
                    <h4 style="color: #fca5a5; margin:0;">⚠️ High Risk Detected</h4>
                    <p style="color: #f8fafc; margin: 5px 0 0 0; font-size: 0.95rem;">The model indicates a high probability of cardiovascular disease.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="background: rgba(16, 185, 129, 0.2); border-left: 5px solid #10b981; padding: 15px; border-radius: 8px; margin-top: 10px;">
                    <h4 style="color: #6ee7b7; margin:0;">✅ Normal Assessment</h4>
                    <p style="color: #f8fafc; margin: 5px 0 0 0; font-size: 0.95rem;">The model predicts low likelihood of heart disease presence.</p>
                </div>
            """, unsafe_allow_html=True)

        # Radar Chart Visualization for Feature Context
        categories = ['Age', 'Resting BP', 'Cholesterol', 'Max HR', 'Oldpeak']
        # Normalized values for representation (0-100 scale approximation)
        values = [
            (Age / 100) * 100,
            (RestingBP / 200) * 100,
            (Cholesterol / 400) * 100,
            (MaxHR / 220) * 100,
            (Oldpeak / 6) * 100
        ]

        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor='rgba(139, 92, 246, 0.3)',
            line=dict(color='#8b5cf6', width=2)
        ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], showticklabels=False),
                bgcolor='rgba(0,0,0,0)'
            ),
            showlegend=False,
            height=260,
            margin=dict(l=40, r=40, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a855f7')
        )

        st.plotly_chart(fig_radar, use_container_width=True)

    else:
        st.markdown("""
            <div class="css-card" style="text-align: center; padding: 40px 20px;">
                <p style="font-size: 3rem; margin: 0;">👈</p>
                <h4 style="color: #94a3b8;">Waiting for Input Data</h4>
                <p style="color: #64748b; font-size: 0.9rem;">Fill out the patient form on the left and click <b>Run Risk Analysis</b> to render real-time diagnostics.</p>
            </div>
        """, unsafe_allow_html=True)