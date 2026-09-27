import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AI Deepfake Detector",
    page_icon="🔍",
    layout="wide"
)

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("deepfake_xai_model.keras")

model = load_model()
base_model = model.layers[0]

# ============================================================
# TITLE
# ============================================================
st.title("🔍 AI Deepfake Image Detection")
st.subheader("Explainable AI using XceptionNet")
st.write(
    "Upload a facial image to detect whether it is potentially Real or Fake "
    "and identify the image regions that influenced the model's prediction."
)

# ============================================================
# FILE UPLOAD
# ============================================================
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# ============================================================
# PREPROCESS IMAGE
# ============================================================
def preprocess_image(image):
    image = image.resize((299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.xception.preprocess_input(image)
    return image

# ============================================================
# PREDICTION
# ============================================================
def predict(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed, verbose=0)[0][0]
    
    if prediction >= 0.5:
        label = "REAL"
        confidence = prediction * 100
    else:
        label = "FAKE"
        confidence = (1 - prediction) * 100
    return label, confidence, prediction

# ============================================================
# GRAD-CAM
# ============================================================
def make_gradcam_heatmap(image_array, prediction, last_conv_layer_name="block14_sepconv2_act"):
    last_conv_layer = base_model.get_layer(last_conv_layer_name)
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_array)
        if prediction < 0.5:
            class_output = 1 - predictions[:, 0]
        else:
            class_output = predictions[:, 0]

    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

# ============================================================
# CREATE GRAD-CAM IMAGE
# ============================================================
def create_gradcam(image, prediction):
    resized_image = image.resize((299, 299))
    original = np.array(resized_image).astype(np.uint8)
    
    image_array = img_to_array(resized_image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.keras.applications.xception.preprocess_input(image_array)

    heatmap = make_gradcam_heatmap(image_array, prediction)
    heatmap_resized = cv2.resize(heatmap, (299, 299))
    
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original, 0.60, heatmap_color, 0.40, 0)
    return original, heatmap_resized, heatmap_color, overlay

# ============================================================
# ANALYZE FACIAL REGIONS
# ============================================================
def analyze_regions(heatmap):
    h, w = heatmap.shape
    regions = {
        "Forehead / Upper Face": (int(h * 0.00), int(h * 0.25), int(w * 0.20), int(w * 0.80)),
        "Eyes / Eyebrows": (int(h * 0.20), int(h * 0.43), int(w * 0.10), int(w * 0.90)),
        "Nose / Center Face": (int(h * 0.35), int(h * 0.65), int(w * 0.25), int(w * 0.75)),
        "Mouth / Lips": (int(h * 0.55), int(h * 0.78), int(w * 0.15), int(w * 0.85)),
        "Chin / Lower Face": (int(h * 0.75), int(h * 0.98), int(w * 0.20), int(w * 0.80)),
        "Left Face": (int(h * 0.30), int(h * 0.80), int(w * 0.00), int(w * 0.30)),
        "Right Face": (int(h * 0.30), int(h * 0.80), int(w * 0.70), int(w * 1.00))
    }

    scores = {}
    for region, (y1, y2, x1, x2) in regions.items():
        region_heatmap = heatmap[y1:y2, x1:x2]
        scores[region] = float(np.mean(region_heatmap)) if region_heatmap.size > 0 else 0.0
    return scores

# ============================================================
# FIND IMPORTANT REGIONS
# ============================================================
def get_artifact_regions(scores):
    sorted_regions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    max_score = sorted_regions[0][1]
    
    important_regions = []
    for region, score in sorted_regions:
        if max_score > 0:
            relative_score = (score / max_score)
            if relative_score >= 0.65:
                important_regions.append(region)
                
    return important_regions[:3], sorted_regions

# ============================================================
# RUN & RENDER THE INTERFACE (New Addition)
# ============================================================
if uploaded_file is not None:
    # 1. Open the file image
    image = Image.open(uploaded_file).convert("RGB")
    
    # 2. Run prediction calculations
    with st.spinner("Analyzing image features..."):
        label, confidence, raw_pred = predict(image)
        original, heatmap, heatmap_color, overlay = create_gradcam(image, raw_pred)
        scores = analyze_regions(heatmap)
        important_regions, sorted_scores = get_artifact_regions(scores)
        
    st.divider()
    
    # 3. Present Metric Dashboard Status
    col_metric1, col_metric2 = st.columns(2)
    with col_metric1:
        if label == "REAL":
            st.success(f"### Prediction: **{label}**")
        else:
            st.error(f"### Prediction: **{label}**")
            
    with col_metric2:
        st.metric(label="Confidence Level", value=f"{confidence:.2f}%")
        
    st.write("---")
    
    # 4. Display Images Side-by-Side
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image(original, caption="Uploaded Original Image (Resized 299x299)", use_container_width=True)
    with col_img2:
        st.image(overlay, caption="Grad-CAM XAI Heatmap Overlay", use_container_width=True)
        
    st.write("---")
    
    # 5. Display Explainable AI Analysis
    st.subheader("💡 XAI Regional Influence Breakdown")
    st.write("The regions showing higher activation scores indicate where the neural network focused its decision criteria:")
    
    col_xai1, col_xai2 = st.columns([1, 1])
    with col_xai1:
        st.write("#### 🔥 Top Focus Zones")
        for region in important_regions:
            st.info(f"• **{region}**")
            
    with col_xai2:
        st.write("#### 📊 Full Anatomical Region Scores")
        for region, score in sorted_scores:
            st.progress(min(max(score, 0.0), 1.0), text=f"{region}: {score:.4f}")
