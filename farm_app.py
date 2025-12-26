import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from farm_weight_predictor import (
    load_configuration, 
    predict_live_weight_pipeline, 
    classify_species,
    detect_reference_marker # New import
)

st.set_page_config(page_title="AI Farm Weight Estimator", layout="wide")

st.title("🚜 AI Farm Weight Estimator")

# --- SIDEBAR: CALIBRATION MODE ---
st.sidebar.header("📏 Calibration Settings")
cal_mode = st.sidebar.radio("Calibration Mode", ["Manual Slider", "Automatic (Blue Marker)"])

marker_size = 0.1 # Default 10cm
manual_scale = 0.0020

if cal_mode == "Manual Slider":
    manual_scale = st.sidebar.slider("Calibration Scale (m/px)", 0.0005, 0.0100, 0.0020, format="%.4f")
else:
    st.sidebar.info("💡 Place a 10cm x 10cm **Bright Blue** square next to the animal.")
    marker_size = st.sidebar.number_input("Marker Real Width (meters)", value=0.10)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    fd, image_path = tempfile.mkstemp(suffix=".jpg")
    try:
        with os.fdopen(fd, 'wb') as tmp:
            tmp.write(uploaded_file.read())
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image_path, use_container_width=True)

        with st.spinner("Analyzing..."):
            species = classify_species(image_path)
            cfg = load_configuration(species)

            # --- APPLY CALIBRATION ---
            if cal_mode == "Automatic (Blue Marker)":
                auto_scale = detect_reference_marker(image_path, marker_size)
                if auto_scale:
                    cfg['calibration_m_per_pixel'] = auto_scale
                    st.sidebar.success(f"✅ Marker detected! Scale: {auto_scale:.5f} m/px")
                else:
                    st.sidebar.error("❌ Blue marker not found. Using default scale.")
                    cfg['calibration_m_per_pixel'] = 0.0020
            else:
                cfg['calibration_m_per_pixel'] = manual_scale
            
            cfg['area_scale'] = cfg['calibration_m_per_pixel'] ** 2
            
            # Run prediction
            weight = predict_live_weight_pipeline(image_path, cfg, species)

            with col2:
                st.subheader("Analysis Results")
                if weight > 0:
                    st.success(f"### Predicted Species: {species.upper()}")
                    st.metric(label="Estimated Live Weight", value=f"{weight:.2f} kg")
                    st.write(f"**Calculated Scale:** {cfg['calibration_m_per_pixel']:.5f} m/px")
                else:
                    st.warning("Could not isolate animal.")

    finally:
        if os.path.exists(image_path):
            os.remove(image_path)
