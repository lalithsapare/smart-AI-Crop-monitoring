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
.main {
    background: linear-gradient(180deg, #f5fbf7 0%, #eef8f1 100%);
}
.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}
h1, h2, h3 {
    color: #114835;
}
div[data-testid="stMetric"] {
    background: white;
    border-radius: 16px;
    padding: 10px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.05);
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

st.title("AI Crop Monitoring")
st.caption("Disease detection, pest analysis, NDVI monitoring, crop stress analysis, irrigation advisory, and combined prediction dashboard.")

with st.sidebar:
    st.header("Field Inputs")
    crop_name = st.selectbox("Crop", ["Tomato", "Rice", "Maize", "Cotton", "Chili", "Banana"])
    location = st.text_input("Farm / Field Name", "Demo Farm")
    temperature = st.slider("Temperature (°C)", 15, 45, 30)
    humidity = st.slider("Humidity (%)", 20, 100, 72)
    soil_moisture = st.slider("Soil Moisture (%)", 5, 100, 42)
    ndvi_value = st.slider("NDVI Estimate", 0.10, 0.95, 0.63)
    uploaded_leaf = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])
    uploaded_pest = st.file_uploader("Upload pest image", type=["jpg", "jpeg", "png"])
    run_btn = st.button("Run Full Prediction")

if run_btn:
    disease_risk, stress_score, yield_score, irrigation_need = synthetic_tabular_predictions(
        temperature, humidity, soil_moisture, ndvi_value
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("NDVI Score", f"{ndvi_value:.2f}")
    c2.metric("Disease Risk", f"{disease_risk * 100:.1f}%")
    c3.metric("Crop Stress", f"{stress_score * 100:.1f}%")
    c4.metric("Yield Potential", f"{yield_score * 100:.1f}%")

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

    st.markdown("## Combined Predictions")
    st.dataframe(results_df, use_container_width=True)

    st.markdown("## Uploaded Images")
    i1, i2 = st.columns(2)
    with i1:
        if leaf_display is not None:
            st.image(leaf_display, caption="Disease / Leaf Image", use_column_width=True)
        else:
            st.info("Leaf image not uploaded.")
    with i2:
        if pest_display is not None:
            st.image(pest_display, caption="Pest Image", use_column_width=True)
        else:
            st.info("Pest image not uploaded.")

    st.markdown("## Confidence Bar Graph")
    chart_df = results_df.set_index("Model")[["Confidence"]]
    st.bar_chart(chart_df)

    st.markdown("## NDVI Trend Analysis")
    ndvi_df = generate_ndvi_curve(ndvi_value).set_index("Date")
    st.line_chart(ndvi_df)

    st.markdown("## Crop Stress Monitoring")
    stress_df = generate_stress_curve(stress_score).set_index("Date")
    st.line_chart(stress_df)

    st.markdown("## Disease and Pest Class Probabilities")
    if uploaded_leaf is not None:
        if crop_name == "Tomato":
            st.subheader("Tomato Disease Probability")
            prob_df = pd.DataFrame({
                "Class": TOMATO_CLASSES,
                "Probability": np.round(tomato_pred[2] * 100, 2)
            }).set_index("Class")
            st.bar_chart(prob_df)
        elif crop_name == "Rice":
            st.subheader("Rice Disease Probability")
            prob_df = pd.DataFrame({
                "Class": RICE_CLASSES,
                "Probability": np.round(rice_pred[2] * 100, 2)
            }).set_index("Class")
            st.bar_chart(prob_df)

    if uploaded_pest is not None:
        st.subheader("Pest Probability")
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

    st.markdown("## Download Report")
    csv_data = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Combined Prediction CSV",
        data=csv_data,
        file_name="ai_crop_monitoring_predictions.csv",
        mime="text/csv"
    )

else:
    st.info("Choose field inputs, upload images, and click 'Run Full Prediction' to start.")

st.markdown("---")
st.caption("AI Crop Monitoring dashboard with deploy-safe charts and lightweight prediction simulation.")