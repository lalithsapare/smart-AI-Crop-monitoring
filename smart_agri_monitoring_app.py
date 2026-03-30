import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from datetime import datetime, timedelta

st.set_page_config(
    page_title="AI Crop Monitoring",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: #0e1117;
    color: white;
}
[data-testid="stSidebar"] {
    background: #262730;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}
.main-title {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 800;
    color: #f7f3ea;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}
.sub-title {
    text-align: center;
    font-size: 1.05rem;
    color: #4f8f3a;
    margin-bottom: 1.8rem;
}
.top-banner {
    text-align: center;
    color: #2f7d32;
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
}
.section-title {
    color: #f8f4ec;
    font-size: 1.1rem;
    font-weight: 700;
    margin-top: 1.2rem;
    margin-bottom: 0.6rem;
}
.metric-box {
    background: #f6f2eb;
    border-radius: 18px;
    padding: 22px 10px;
    text-align: center;
    color: #ffffff;
    box-shadow: 0 2px 12px rgba(0,0,0,0.15);
}
.metric-label {
    font-size: 0.9rem;
    color: #8a8a8a;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #f4f1ea;
}
.custom-divider {
    border-top: 1px solid rgba(255,255,255,0.12);
    margin-top: 1.4rem;
    margin-bottom: 1.4rem;
}
.note-line {
    color: #f1e8d8;
    font-weight: 600;
    margin-top: 0.8rem;
}
.stButton > button {
    width: 100%;
    background: #ff4b4b;
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    padding: 0.85rem 1rem;
    font-size: 1.05rem;
}
.stButton > button:hover {
    background: #ff3838;
    color: white;
}
.upload-label {
    color: #f7f3ea;
    font-weight: 600;
    margin-bottom: 0.3rem;
}
.table-wrap {
    background: rgba(255,255,255,0.02);
    border-radius: 12px;
    padding: 8px;
}
</style>
""", unsafe_allow_html=True)

TOMATO_CLASSES = ["Healthy", "Early Blight", "Late Blight", "Leaf Mold"]
RICE_CLASSES = ["Healthy", "Brown Spot", "Blast", "Hispa"]
PEST_CLASSES = ["No Pest", "Aphids", "Armyworm", "Whitefly"]

def preprocess_image(uploaded_file, target_size=(224, 224)):
    image = Image.open(uploaded_file).convert("RGB")
    display_image = image.copy()
    resized = image.resize(target_size)
    img_array = np.array(resized).astype("float32") / 255.0
    batched = np.expand_dims(img_array, axis=0)
    return display_image, batched

def fake_predict(class_names, seed_value=1):
    rng = np.random.default_rng(seed_value)
    probs = rng.dirichlet(np.ones(len(class_names)))
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx]), probs

def synthetic_tabular_predictions(temp, humidity, soil_moisture, ndvi):
    disease_risk = min(max((humidity * 0.35 + temp * 0.25 + (1 - ndvi) * 40) / 100, 0), 1)
    stress_score = min(max(((35 - soil_moisture) * 0.9 + (1 - ndvi) * 45 + max(temp - 32, 0) * 1.5) / 100, 0), 1)
    yield_score = min(max((ndvi * 0.55 + soil_moisture * 0.20 + (1 - abs(temp - 28) / 20) * 0.25), 0), 1)
    irrigation_need = min(max((1 - soil_moisture / 100) * 0.6 + max(temp - 30, 0) / 100 + (1 - ndvi) * 0.25, 0), 1)
    return disease_risk, stress_score, yield_score, irrigation_need

def generate_ndvi_curve(base_ndvi):
    dates = [datetime.today().date() - timedelta(days=i) for i in range(14)][::-1]
    values = np.clip(base_ndvi + np.random.normal(0, 0.03, len(dates)), 0.2, 0.95)
    return pd.DataFrame({"Date": dates, "NDVI": values})

def generate_stress_curve(base_score):
    dates = [datetime.today().date() - timedelta(days=i) for i in range(14)][::-1]
    values = np.clip(base_score + np.random.normal(0, 0.05, len(dates)), 0.05, 0.98)
    return pd.DataFrame({"Date": dates, "Stress": values})

with st.sidebar:
    st.markdown("## Field Inputs")
    crop_name = st.selectbox("Crop", ["Rice", "Tomato", "Maize", "Cotton", "Chili", "Banana"])
    location = st.text_input("Farm / Field Name", "Demo Farm")
    temperature = st.slider("Temperature (°C)", 15, 45, 35)
    humidity = st.slider("Humidity (%)", 20, 100, 85)
    soil_moisture = st.slider("Soil Moisture (%)", 5, 100, 42)
    ndvi_value = st.slider("NDVI Estimate", 0.10, 0.95, 0.63)

    st.markdown('<div class="upload-label">Upload leaf image</div>', unsafe_allow_html=True)
    uploaded_leaf = st.file_uploader("", type=["jpg", "jpeg", "png"], key="leaf_uploader")

    st.markdown('<div class="upload-label">Upload pest image</div>', unsafe_allow_html=True)
    uploaded_pest = st.file_uploader("", type=["jpg", "jpeg", "png"], key="pest_uploader")

st.markdown('<div class="top-banner">🌿 Smart Agriculture Complete AI System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Weather Forecast • Crop Recommendation • Yield Prediction • Fertilizer & Irrigation Optimization</div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">AI Crop Monitoring</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Disease detection, pest analysis, NDVI monitoring, crop stress analysis, irrigation advisory, and combined prediction dashboard.</div>', unsafe_allow_html=True)

run_btn = st.button("🚀 RUN COMPLETE AI ANALYSIS")

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="note-line">💡 Smart Farming AI for AP/Telangana Farmers | 🌾 v2.0</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

if run_btn:
    disease_risk, stress_score, yield_score, irrigation_need = synthetic_tabular_predictions(
        temperature, humidity, soil_moisture, ndvi_value
    )

    tomato_pred = ("No image", 0.0, np.zeros(len(TOMATO_CLASSES)))
    rice_pred = ("No image", 0.0, np.zeros(len(RICE_CLASSES)))
    pest_pred = ("No image", 0.0, np.zeros(len(PEST_CLASSES)))

    leaf_display = None
    pest_display = None

    if uploaded_leaf is not None:
        leaf_display, _ = preprocess_image(uploaded_leaf)
        if crop_name == "Tomato":
            tomato_pred = fake_predict(TOMATO_CLASSES, seed_value=11)
        elif crop_name == "Rice":
            rice_pred = fake_predict(RICE_CLASSES, seed_value=22)
        else:
            tomato_pred = fake_predict(TOMATO_CLASSES, seed_value=33)

    if uploaded_pest is not None:
        pest_display, _ = preprocess_image(uploaded_pest)
        pest_pred = fake_predict(PEST_CLASSES, seed_value=44)

    crop_health_label = "Healthy"
    if stress_score > 0.70:
        crop_health_label = "Severe Stress"
    elif stress_score > 0.40:
        crop_health_label = "Moderate Stress"

    irrigation_label = "Low"
    if irrigation_need > 0.70:
        irrigation_label = "High"
    elif irrigation_need > 0.40:
        irrigation_label = "Moderate"

    results_df = pd.DataFrame([
        {"Model": "Tomato Disease CNN", "Prediction": tomato_pred[0], "Confidence": round(tomato_pred[1] * 100, 2)},
        {"Model": "Rice Disease CNN", "Prediction": rice_pred[0], "Confidence": round(rice_pred[1] * 100, 2)},
        {"Model": "Pest Detection CNN", "Prediction": pest_pred[0], "Confidence": round(pest_pred[1] * 100, 2)},
        {"Model": "NDVI Stress Model", "Prediction": crop_health_label, "Confidence": round(stress_score * 100, 2)},
        {"Model": "Crop Health Model", "Prediction": crop_health_label, "Confidence": round((1 - stress_score) * 100, 2)},
        {"Model": "Yield Model", "Prediction": f"{yield_score * 100:.1f}% Yield Potential", "Confidence": round(yield_score * 100, 2)},
        {"Model": "Irrigation Model", "Prediction": irrigation_label, "Confidence": round(irrigation_need * 100, 2)},
    ])

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{ndvi_value:.2f}</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{disease_risk*100:.1f}%</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{stress_score*100:.1f}%</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-box"><div class="metric-value">{yield_score*100:.1f}%</div></div>', unsafe_allow_html=True)

    st.markdown("## Combined Predictions")
    st.markdown('<div class="table-wrap">', unsafe_allow_html=True)
    st.dataframe(results_df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("## Uploaded Images")
    c1, c2 = st.columns(2)
    with c1:
        if leaf_display is not None:
            st.image(leaf_display, caption="Disease / Leaf Image", use_column_width=True)
        else:
            st.info("Leaf image not uploaded.")
    with c2:
        if pest_display is not None:
            st.image(pest_display, caption="Pest Image", use_column_width=True)
        else:
            st.info("Pest image not uploaded.")

    st.markdown("## Confidence Overview")
    st.bar_chart(results_df.set_index("Model")[["Confidence"]])

    st.markdown("## NDVI Trend Analysis")
    ndvi_df = generate_ndvi_curve(ndvi_value).set_index("Date")
    st.line_chart(ndvi_df)

    st.markdown("## Crop Stress Monitoring")
    stress_df = generate_stress_curve(stress_score).set_index("Date")
    st.line_chart(stress_df)

    if uploaded_leaf is not None:
        if crop_name == "Tomato":
            st.markdown("## Tomato Disease Probability")
            tomato_df = pd.DataFrame({
                "Class": TOMATO_CLASSES,
                "Probability": np.round(tomato_pred[2] * 100, 2)
            }).set_index("Class")
            st.bar_chart(tomato_df)
        elif crop_name == "Rice":
            st.markdown("## Rice Disease Probability")
            rice_df = pd.DataFrame({
                "Class": RICE_CLASSES,
                "Probability": np.round(rice_pred[2] * 100, 2)
            }).set_index("Class")
            st.bar_chart(rice_df)

    if uploaded_pest is not None:
        st.markdown("## Pest Probability")
        pest_df = pd.DataFrame({
            "Class": PEST_CLASSES,
            "Probability": np.round(pest_pred[2] * 100, 2)
        }).set_index("Class")
        st.bar_chart(pest_df)

    st.markdown("## Recommendation")
    recommendations = []
    if disease_risk > 0.65:
        recommendations.append("High disease risk detected; inspect infected plants immediately.")
    if stress_score > 0.60:
        recommendations.append("Crop stress is elevated; check water, heat, and nutrient balance.")
    if irrigation_need > 0.60:
        recommendations.append("Irrigation need is high; plan watering soon.")
    if ndvi_value < 0.45:
        recommendations.append("NDVI is low; crop vigor may be reduced and field scouting is advised.")
    if yield_score > 0.70:
        recommendations.append("Yield outlook is favorable under current field conditions.")
    if not recommendations:
        recommendations.append("Field condition appears stable with no major warning at this time.")

    for rec in recommendations:
        st.markdown(f"- {rec}")

    csv_data = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Combined Prediction CSV",
        data=csv_data,
        file_name="ai_crop_monitoring_predictions.csv",
        mime="text/csv"
    )

else:
    st.info("Set your field inputs, upload images, and click the main red button to run analysis.")