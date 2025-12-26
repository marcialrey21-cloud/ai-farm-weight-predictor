import numpy as np
import cv2
import os
import joblib 
import datetime
from ultralytics import YOLO
from sklearn.linear_model import LinearRegression
from fpdf import FPDF

# --- I. REGISTRY & GLOBAL MODELS ---

SPECIES_CONFIG = {
    "pig": {"yolo_class_id": 19, "calibration_m_per_pixel": 0.005},
    "cattle": {"yolo_class_id": 19, "calibration_m_per_pixel": 0.006},
    "goat": {"yolo_class_id": 18, "calibration_m_per_pixel": 0.004},
    "buffalo": {"yolo_class_id": 19, "calibration_m_per_pixel": 0.007},
    "sheep": {"yolo_class_id": 18, "calibration_m_per_pixel": 0.004},
    "chicken": {"yolo_class_id": 14, "calibration_m_per_pixel": 0.001},
    "turkey": {"yolo_class_id": 14, "calibration_m_per_pixel": 0.0015},
    "duck": {"yolo_class_id": 14, "calibration_m_per_pixel": 0.0012}
}

OUTPUT_MASK_FILE = 'temp_mask.png'

try:
    # Using 'n' model for speed; use 's' or 'm' for better accuracy
    SEGMENTATION_MODEL = YOLO('yolov8n-seg.pt')
    print("✅ Segmentation Model Ready.")
except Exception as e:
    print(f"❌ YOLO Load Error: {e}")
    SEGMENTATION_MODEL = None

# --- II. HELPER FUNCTIONS ---

def generate_pdf_report(species, weight, scale):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="AI Farm Weight Estimation Report", ln=True, align='C')
    
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    pdf.ln(10)
    
    # Table Header
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(95, 10, "Metric", border=1)
    pdf.cell(95, 10, "Value", border=1, ln=True)
    
    # Table Content
    pdf.set_font("Arial", size=12)
    data = [
        ("Animal Species", species.capitalize()),
        ("Estimated Weight", f"{weight:.2f} kg"),
        ("Calibration Scale", f"{scale:.5f} m/px")
    ]
    for metric, value in data:
        pdf.cell(95, 10, metric, border=1)
        pdf.cell(95, 10, value, border=1, ln=True)
    
    return pdf.output(dest='S').encode('latin-1')

def train_and_save_model(species, save_path):
    # Standardizing training data to always have 2 features: [Area, Length]
    if species in ["cattle", "buffalo"]:
        X = np.array([[2.0, 2.0], [3.0, 2.4], [4.0, 2.8]])
        Y = np.array([550, 850, 1150]) if species == "cattle" else np.array([650, 950, 1250])
    elif species in ["pig", "sheep", "goat"]:
        X = np.array([[0.5, 1.1], [1.0, 1.4], [1.5, 1.7]])
        Y = np.array([60, 125, 210])
    else:
        # Poultry logic with 2 features to match extract_features output
        X = np.array([[0.05, 0.25], [0.1, 0.35], [0.15, 0.45]])
        mult = {"chicken": 150, "duck": 250, "turkey": 600}
        Y = X[:, 0] * mult.get(species, 100)
    
    model = LinearRegression()
    model.fit(X, Y)
    joblib.dump(model, save_path)
    return model

def load_trained_model(species, save_path):
    if os.path.exists(save_path):
        return joblib.load(save_path)
    return train_and_save_model(species, save_path)

def load_configuration(species):
    if species not in SPECIES_CONFIG:
        raise ValueError(f"Species '{species}' not supported.")
    
    config = SPECIES_CONFIG[species].copy()
    config['species'] = species
    config['model_file'] = f'{species}_weight_model.joblib'
    config['area_scale'] = config['calibration_m_per_pixel'] ** 2
    config['weight_model'] = load_trained_model(species, config['model_file'])
    return config

# --- III. PROCESSING FUNCTIONS ---

def create_animal_mask(image_path, target_id, output_path):
    if SEGMENTATION_MODEL is None: return None
    results = SEGMENTATION_MODEL(image_path, verbose=False)
    
    # Original image dimensions
    img_orig = results[0].orig_img
    h, w = img_orig.shape[:2]
    final_mask = np.zeros((h, w), dtype=np.uint8)
    found = False
    
    for result in results:
        if result.masks is None: continue
        
        # Iterate through detected objects
        for i, mask_data in enumerate(result.masks.data):
            cls = int(result.boxes.cls[i])
            if cls == target_id:
                # Move to CPU, convert to numpy, and resize to match original image
                m = mask_data.cpu().numpy()
                m = cv2.resize(m, (w, h))
                mask_array = (m > 0.5).astype(np.uint8) * 255
                final_mask = cv2.bitwise_or(final_mask, mask_array)
                found = True
    
    if found:
        cv2.imwrite(output_path, final_mask)
        return output_path
    return None

def extract_features(mask_path, area_scale, length_scale):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None: return 0.0, 0.0
    
    area_px = np.count_nonzero(mask)
    real_area = area_px * area_scale
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return real_area, 0.0
    
    # Use the largest contour for length calculation
    cnt = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    # The 'length' is the longer side of the bounding box
    pixel_length = max(rect[1]) 
    real_length = pixel_length * length_scale
    
    return real_area, real_length

def classify_species(image_path):
    fn = os.path.basename(image_path).lower()
    for s in SPECIES_CONFIG.keys():
        if s in fn: return s
    return "pig"

def detect_reference_marker(image_path, real_marker_width_m=0.1):
    img = cv2.imread(image_path)
    if img is None: return None
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return None
    
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    if w < 5: return None 
    return real_marker_width_m / w

# --- IV. THE PIPELINE ---

def predict_live_weight_pipeline(image_path, config, species_name):
    mask_path = create_animal_mask(image_path, config['yolo_class_id'], OUTPUT_MASK_FILE)
    if not mask_path: return 0.0
    
    area, length = extract_features(mask_path, config['area_scale'], config['calibration_m_per_pixel'])
    
    # Cleanup mask file after processing
    if os.path.exists(mask_path):
        os.remove(mask_path)
        
    if area == 0.0: return 0.0
    
    # Prepare input for model: [[Area, Length]]
    features = np.array([[area, length]])
    prediction = config['weight_model'].predict(features)[0]
    
    return float(prediction)
