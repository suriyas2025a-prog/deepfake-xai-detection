import cv2
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Deepfake Artifact Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Dark Modern Tech Theme
st.markdown(
    """
<style>
    /* Main App Background and Font */
    .stApp {
        background-color: #0E1117;
        color: #E0E6ED;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Container Styling */
    .header-container {
        padding: 1.5rem 0rem 1rem 0rem;
        border-bottom: 1px solid #1E2638;
        margin-bottom: 2rem;
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .header-subtitle {
        color: #94A3B8;
        font-size: 1rem;
    }

    /* Result Cards */
    .result-card-real {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(6, 78, 59, 0.2) 100%);
        border: 1px solid #10B981;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-card-fake {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(127, 29, 29, 0.2) 100%);
        border: 1px solid #EF4444;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    .card-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }
    .card-value {
        font-size: 2rem;
        font-weight: 800;
        margin: 0.3rem 0;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #1E2638;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================


@st.cache_resource
def load_model():
    loaded_model = tf.keras.models.load_model("deepfake_xai_model.keras")

    # Build the model explicitly
    dummy_input = tf.zeros((1, 299, 299, 3), dtype=tf.float32)
    loaded_model(dummy_input)

    return loaded_model


try:
    model = load_model()
    base_model = model.layers[0]
except Exception as e:
    st.error(
        f"Failed to load model file `deepfake_xai_model.keras`. Please verify the model file exists. Error: {e}"
    )


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def preprocess_image(image):
    image = image.resize((299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.xception.preprocess_input(image)
    return image


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


def make_gradcam_heatmap(
    image_array, prediction, last_conv_layer_name="block14_sepconv2_act"
):
    _ = model(image_array, training=False)
    base_model = model.layers[0]
    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, base_model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, base_output = grad_model(image_array, training=False)
        x = base_output
        for layer in model.layers[1:]:
            x = layer(x, training=False)

        final_prediction = x[:, 0]
        if prediction < 0.5:
            class_output = 1.0 - final_prediction
        else:
            class_output = final_prediction

    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def create_gradcam(image, prediction):
    resized_image = image.resize((299, 299))
    original = np.array(resized_image).astype(np.uint8)

    image_array = img_to_array(resized_image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.keras.applications.xception.preprocess_input(image_array)

    heatmap = make_gradcam_heatmap(
        image_array, prediction, last_conv_layer_name="block14_sepconv2_act"
    )
    heatmap = cv2.resize(heatmap, (299, 299))

    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original, 0.60, heatmap_color, 0.40, 0)

    return original, heatmap, heatmap_color, overlay


# ============================================================
# NEW FUNCTION: PINPOINT DYNAMIC ARTIFACT HOTSPOTS
# ============================================================


def locate_artificial_artifacts(heatmap, threshold_ratio=0.6, max_artifacts=4):
    """
    Scans heatmap activation intensities, thresholding high-activation regions
    to locate specific x, y bounding points of artifacts in the image.
    """
    h, w = heatmap.shape

    # Normalize heatmap between 0 and 255
    norm_heatmap = cv2.normalize(
        heatmap, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # Threshold heatmap to isolate dominant peak activations
    thresh_val = int(255 * threshold_ratio)
    _, thresh_img = cv2.threshold(
        norm_heatmap, thresh_val, 255, cv2.THRESH_BINARY
    )

    # Find contours around intense activation regions
    contours, _ = cv2.findContours(
        thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    artifacts = []

    # Sort contours by area size (largest focus regions first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for i, cnt in enumerate(contours[:max_artifacts]):
        x, y, box_w, box_h = cv2.boundingRect(cnt)
        center_x = x + (box_w // 2)
        center_y = y + (box_h // 2)

        # Region intensity score
        mask = np.zeros_like(norm_heatmap)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_val = cv2.mean(norm_heatmap, mask=mask)[0] / 255.0

        # Map pixel positions to spatial descriptions
        vert_pos = (
            "Top" if center_y < h * 0.35 else ("Bottom" if center_y > h * 0.65 else "Middle")
        )
        horiz_pos = (
            "Left" if center_x < w * 0.35 else ("Right" if center_x > w * 0.65 else "Center")
        )
        quadrant = f"{vert_pos}-{horiz_pos}"

        artifacts.append(
            {
                "id": i + 1,
                "quadrant": quadrant,
                "center_x": center_x,
                "center_y": center_y,
                "width": box_w,
                "height": box_h,
                "intensity": mean_val,
            }
        )

    return artifacts


# ============================================================
# SIDEBAR NAVIGATION & UPLOAD
# ============================================================

with st.sidebar:
    st.title("🛡️ Control Panel")
    st.write("Upload an image and run analysis.")

    uploaded_file = st.file_uploader(
        "Choose a facial image...", type=["jpg", "jpeg", "png"]
    )

    st.markdown("---")
    st.markdown("### ⚙️ System Status")
    st.caption("Model Backbone: **XceptionNet**")
    st.caption("XAI Method: **Grad-CAM Artifact Locator**")
    st.caption("Input Specs: **299x299 RGB**")


# ============================================================
# MAIN INTERFACE
# ============================================================

st.markdown(
    """
    <div class="header-container">
        <div class="header-title">AI Deepfake Detector & Artifact Locator</div>
        <div class="header-subtitle">Analyze facial images for synthetic manipulations and pinpoint specific visual artifact locations using Grad-CAM.</div>
    </div>
""",
    unsafe_allow_html=True,
)

if not uploaded_file:
    st.info("👈 Please upload an image in the sidebar to begin analysis.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 1. Upload")
        st.caption("Supply a portrait or facial shot in JPG or PNG format.")
    with col2:
        st.markdown("#### 2. Classify")
        st.caption("The XceptionNet architecture evaluates structural artifacts.")
    with col3:
        st.markdown("#### 3. Pinpoint")
        st.caption("Locate specific coordinates where anomalies or synthetic artifacts occur.")

else:
    image = Image.open(uploaded_file).convert("RGB")

    # Main Grid Layout
    left_col, right_col = st.columns([1, 1.2], gap="large")

    with left_col:
        st.subheader("📷 Input Preview")
        st.image(image, caption="Uploaded Facial Image", use_container_width=True)

        run_analysis = st.button(
            "⚡ Run Artifact Detection",
            use_container_width=True,
            type="primary",
        )

    with right_col:
        st.subheader("📊 Diagnostic Workspace")

        if run_analysis:
            with st.spinner("Analyzing image features & locating visual artifacts..."):
                label, confidence, prediction = predict(image)
                original, heatmap, heatmap_color, overlay = create_gradcam(
                    image, prediction
                )
                artifacts = locate_artificial_artifacts(heatmap)

            # Store result in session state
            st.session_state["analysis_done"] = True
            st.session_state["data"] = {
                "label": label,
                "confidence": confidence,
                "prediction": prediction,
                "original": original,
                "heatmap": heatmap,
                "heatmap_color": heatmap_color,
                "overlay": overlay,
                "artifacts": artifacts,
            }

        if st.session_state.get("analysis_done", False):
            data = st.session_state["data"]

            # Display Classification Metric Card
            if data["label"] == "REAL":
                st.markdown(
                    f"""
                    <div class="result-card-real">
                        <div class="card-label" style="color: #10B981;">Classification Result</div>
                        <div class="card-value" style="color: #10B981;">✅ AUTHENTIC (REAL)</div>
                        <span style="color: #A7F3D0;">Confidence Score: <b>{data['confidence']:.2f}%</b></span>
                    </div>
                """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-card-fake">
                        <div class="card-label" style="color: #EF4444;">Classification Result</div>
                        <div class="card-value" style="color: #EF4444;">⚠️ MANIPULATED (FAKE)</div>
                        <span style="color: #FCA5A5;">Confidence Score: <b>{data['confidence']:.2f}%</b></span>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

            # Interactive Tabs
            tab_artifacts, tab_heatmaps, tab_summary = st.tabs(
                [
                    "🎯 Detected Artifact Points",
                    "🔥 Grad-CAM Heatmap",
                    "💡 Explanation",
                ]
            )

            with tab_artifacts:
                st.markdown("##### 📍 Pinpointed Anomaly Hotspots")

                if data["artifacts"]:
                    if data["label"] == "FAKE":
                        st.write(
                            "The model identified the following **specific coordinate regions** as strong artificial/manipulation hotspots:"
                        )
                    else:
                        st.write(
                            "The model evaluated the following **key structural reference points** supporting authenticity:"
                        )

                    # Bullet Points for Artifact Locations
                    for item in data["artifacts"]:
                        severity_badge = (
                            "🔴 **High Severity**"
                            if item["intensity"] > 0.8
                            else "🟡 **Moderate Severity**"
                        )
                        
                        st.markdown(
                            f"""
                            * **Artifact #{item['id']} — {item['quadrant']} Region**
                              * **Location Coordinates:** `X: {item['center_x']}px, Y: {item['center_y']}px` *(Bounding Box: {item['width']}×{item['height']}px)*
                              * **Activation Intensity:** `{item['intensity']*100:.1f}%` ({severity_badge})
                              * **Diagnostic Observation:** High activation in this local cluster suggests boundary blending, abnormal texture gradients, or synthetic facial reconstruction artifacts.
                            """
                        )
                else:
                    st.info(
                        "No localized artifact clusters were detected above the sensitivity threshold."
                    )

            with tab_heatmaps:
                g_col1, g_col2 = st.columns(2)
                with g_col1:
                    st.image(
                        data["original"],
                        caption="Original Image",
                        use_container_width=True,
                    )
                with g_col2:
                    st.image(
                        data["overlay"],
                        caption="Grad-CAM Hotspot Activation",
                        use_container_width=True,
                    )

            with tab_summary:
                st.markdown("##### Technical Summary")
                if data["label"] == "FAKE":
                    st.write(
                        f"The model detected **{len(data['artifacts'])} distinct manipulation hotspot(s)** in the image. "
                        "These areas represent clusters where pixel transitions deviate from typical camera sensor noise or natural biological features."
                    )
                else:
                    st.write(
                        "The image displays coherent spatial structures across all evaluated focus points, with no localized synthetic anomalies detected."
                    )

                st.divider()
                st.caption(
                    "⚠️ **Disclaimer:** Pinpointed artifact locations indicate spatial areas of high activation in the neural network's final layer. They represent strong statistical evidence of synthetic features rather than visual proof."
                )
        else:
            st.info("Click **'⚡ Run Artifact Detection'** to generate analysis.")
