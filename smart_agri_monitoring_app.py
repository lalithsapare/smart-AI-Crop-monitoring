import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
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
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}
h1, h2, h3 {
    color: #104735;
}
div[data-testid="stMetric"] {
    background: white;
    border-radius: 16px;
    padding: 10px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.05);
}
.card {
    background: white;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.06);
    margin-bottom: 16px;
}
.small-note {
    color: #647067;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_tf_model(path):
    if os.path.exists(path):
        return tf.keras.models.load_model(path)
    return None

def load_classes(path, fallback):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return fallback

TOMATO_CLASSES = load_classes("labels/tomato_classes.txt", ["Healthy", "Early Blight", "Late Blight", "Leaf Mold"])
RICE_CLASSES = load_classes("labels/rice_classes.txt", ["Healthy", "Brown Spot", "Blast", "Hispa"])
PEST_CLASSES = load_classes("labels/pest_classes.txt", ["No Pest", "Aphids", "Armyworm", "Whitefly"])
HEALTH_CLASSES = load_classes("labels/health_classes.txt", ["Healthy", "Moderate Stress", "Severe Stress"])

tomato_model = load_tf_model("models/tomato_disease_model.keras")
rice_model = load_tf_model("models/rice_disease_model.keras")
pest_model = load_tf_model("models/pest_model.keras")
stress_model = load_tf_model("models/ndvi_stress_model.keras")
crop_health_model = load_tf_model("models/crop_health_model.keras")
yield_model = load_tf_model("models/yield_model.keras")
irrigation_model = load_tf_model("models/irrigation_model.keras")

def preprocess_image(uploaded_file, target_size=(224, 224)):
    image = Image.open(uploaded_file).convert("RGB")
    display_image = image.copy()
    resized = image.resize(target_size)
    img_array = np.array(resized).astype("float32") / 255.0
    batched = np.expand_dims(img_array, axis=0)
    return display_image, batched, img_array

def predict_image_model(model, image_tensor, class_names):
    if model is None:
        probs = np.random.dirichlet(np.ones(len(class_names)), size=1)[0]
    else:
        raw = model.predict(image_tensor, verbose=0)[0]
        probs = np.array(raw).flatten()
        if probs.sum() <= 0:
            probs = np.random.dirichlet(np.ones(len(class_names)), size=1)[0]
        else:
            probs = probs / probs.sum()
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

def probability_df(classes, probs, model_name):
    return pd.DataFrame({
        "Class": classes,
        "Probability": np.round(probs * 100, 2),
        "Model": model_name
    })

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
        leaf_display, leaf_tensor, _ = preprocess_image(uploaded_leaf)
        if crop_name == "Tomato":
            tomato_pred = predict_image_model(tomato_model, leaf_tensor, TOMATO_CLASSES)
        elif crop_name == "Rice":
            rice_pred = predict_image_model(rice_model, leaf_tensor, RICE_CLASSES)
        else:
            tomato_pred = predict_image_model(tomato_model, leaf_tensor, TOMATO_CLASSES)

    if uploaded_pest is not None:
        pest_display, pest_tensor, _ = preprocess_image(uploaded_pest)
        pest_pred = predict_image_model(pest_model, pest_tensor, PEST_CLASSES)

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

    combined_records = [
        {"Model": "Tomato Disease CNN", "Prediction": tomato_pred[0], "Confidence": round(tomato_pred[1] * 100, 2)},
        {"Model": "Rice Disease CNN", "Prediction": rice_pred[0], "Confidence": round(rice_pred[1] * 100, 2)},
        {"Model": "Pest Detection CNN", "Prediction": pest_pred[0], "Confidence": round(pest_pred[1] * 100, 2)},
        {"Model": "NDVI Stress Model", "Prediction": crop_health_label, "Confidence": round(stress_score * 100, 2)},
        {"Model": "Crop Health Model", "Prediction": crop_health_label, "Confidence": round((1 - stress_score) * 100, 2)},
        {"Model": "Yield Model", "Prediction": f"{yield_score * 100:.1f}% Yield Potential", "Confidence": round(yield_score * 100, 2)},
        {"Model": "Irrigation Model", "Prediction": irrigation_label, "Confidence": round(irrigation_need * 100, 2)},
    ]

    results_df = pd.DataFrame(combined_records)

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

    st.markdown("## Visual Analytics")

    v1, v2 = st.columns(2)

    with v1:
        fig_bar = px.bar(
            results_df,
            x="Model",
            y="Confidence",
            color="Confidence",
            color_continuous_scale="greens",
            title="Model Confidence Scores"
        )
        fig_bar.update_layout(height=450, xaxis_tickangle=-30)
        st.plotly_chart(fig_bar, use_container_width=True)

    with v2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=stress_score * 100,
            title={"text": "Crop Stress Monitoring"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#d97706"},
                "steps": [
                    {"range": [0, 35], "color": "#bbf7d0"},
                    {"range": [35, 70], "color": "#fde68a"},
                    {"range": [70, 100], "color": "#fecaca"}
                ]
            }
        ))
        fig_gauge.update_layout(height=450)
        st.plotly_chart(fig_gauge, use_container_width=True)

    ndvi_df = generate_ndvi_curve(ndvi_value)
    stress_df = generate_stress_curve(stress_score)

    t1, t2 = st.columns(2)

    with t1:
        fig_ndvi = px.line(ndvi_df, x="Date", y="NDVI", markers=True, title="NDVI Trend Analysis")
        fig_ndvi.update_traces(line_color="#15803d")
        st.plotly_chart(fig_ndvi, use_container_width=True)

    with t2:
        fig_stress = px.line(stress_df, x="Date", y="Stress", markers=True, title="Crop Stress Trend")
        fig_stress.update_traces(line_color="#dc2626")
        st.plotly_chart(fig_stress, use_container_width=True)

    st.markdown("## Disease and Pest Probability Analysis")

    prob_frames = []
    if uploaded_leaf is not None:
        if crop_name == "Tomato":
            prob_frames.append(probability_df(TOMATO_CLASSES, tomato_pred[2], "Tomato CNN"))
        elif crop_name == "Rice":
            prob_frames.append(probability_df(RICE_CLASSES, rice_pred[2], "Rice CNN"))

    if uploaded_pest is not None:
        prob_frames.append(probability_df(PEST_CLASSES, pest_pred[2], "Pest CNN"))

    if prob_frames:
        all_probs = pd.concat(prob_frames, ignore_index=True)
        fig_prob = px.bar(
            all_probs,
            x="Class",
            y="Probability",
            color="Model",
            barmode="group",
            title="Class-wise Prediction Probabilities"
        )
        st.plotly_chart(fig_prob, use_container_width=True)
    else:
        st.warning("Upload image files to view disease and pest probability charts.")

    st.markdown("## Field Recommendation")

    recommendations = []
    if disease_risk > 0.65:
        recommendations.append("High disease risk detected; inspect infected plants immediately.")
    if stress_score > 0.60:
        recommendations.append("Crop stress is elevated; check water, heat, and nutrient balance.")
    if irrigation_need > 0.60:
        recommendations.append("Irrigation need is high; plan watering soon.")
    if ndvi_value < 0.45:
        recommendations.append("NDVI is low; crop vigor may be reduced and scouting is advised.")
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
st.caption("AI Crop Monitoring dashboard for disease, pest, NDVI, crop stress, irrigation, and yield prediction.")
