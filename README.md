# 🚜 AI Farm Weight Estimator

An AI-powered web application built with **Streamlit** and **YOLOv8** to estimate the live weight of farm animals using computer vision and reference marker calibration.

## 🌟 Features
* **Multi-Species Support**: Specialized logic for Cattle, Pigs, Goats, Sheep, and Poultry.
* **Smart Segmentation**: Uses YOLOv8-seg to accurately isolate the animal from the background.
* **Dual Calibration**: 
    * **Manual**: Adjust scale using a slider.
    * **Automatic**: Uses a 10cm blue square marker for precise pixel-to-meter scaling.
* **Live Camera**: Capture photos directly from your Android/iOS device.
* **PDF Reports**: Generate and download a professional weight estimation report.

## 🛠️ Installation & Local Setup

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/ai-farm-weight-predictor.git](https://github.com/YOUR_USERNAME/ai-farm-weight-predictor.git)
   cd ai-farm-weight-predictor
