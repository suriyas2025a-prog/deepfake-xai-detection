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

    return tf.keras.models.load_model(
        "deepfake_xai_model.keras"
    )


model = load_model()

# Get the Xception base model
# This assumes your model was created as:
#
# Sequential([
#     base_model,
#     GlobalAveragePooling2D(),
#     ...
# ])

base_model = model.layers[0]


# ============================================================
# TITLE
# ============================================================

st.title("🔍 AI Deepfake Image Detection")

st.subheader(
    "Explainable AI using XceptionNet"
)

st.write(
    "Upload a facial image to detect whether it is "
    "potentially Real or Fake and identify the image "
    "regions that influenced the model's prediction."
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):

    image = image.resize(
        (299, 299)
    )

    image = img_to_array(image)

    image = np.expand_dims(
        image,
        axis=0
    )

    image = tf.keras.applications.xception.preprocess_input(
        image
    )

    return image


# ============================================================
# PREDICTION
# ============================================================

def predict(image):

    processed = preprocess_image(
        image
    )

    prediction = model.predict(
        processed,
        verbose=0
    )[0][0]

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

def make_gradcam_heatmap(
    image_array,
    prediction,
    last_conv_layer_name="block14_sepconv2_act"
):

    # Find the final convolutional layer
    last_conv_layer = base_model.get_layer(
        last_conv_layer_name
    )

    # Create model that returns:
    # 1. convolutional feature maps
    # 2. final prediction
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(
            image_array
        )

        # For Fake:
        # fake probability = 1 - real probability
        #
        # For Real:
        # real probability = real probability

        if prediction < 0.5:

            class_output = 1 - predictions[:, 0]

        else:

            class_output = predictions[:, 0]

    # Calculate gradients
    grads = tape.gradient(
        class_output,
        conv_outputs
    )

    # Average gradients over spatial dimensions
    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    # Weight feature maps
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    # ReLU
    heatmap = tf.maximum(
        heatmap,
        0
    )

    # Normalize
    heatmap = heatmap / (
        tf.reduce_max(heatmap) + 1e-8
    )

    return heatmap.numpy()


# ============================================================
# CREATE GRAD-CAM IMAGE
# ============================================================

def create_gradcam(
    image,
    prediction
):

    # Resize
    resized_image = image.resize(
        (299, 299)
    )

    original = np.array(
        resized_image
    ).astype(
        np.uint8
    )

    # Preprocess
    image_array = img_to_array(
        resized_image
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    image_array = tf.keras.applications.xception.preprocess_input(
        image_array
    )

    # Generate heatmap
    heatmap = make_gradcam_heatmap(
        image_array,
        prediction
    )

    # Resize heatmap
    heatmap = cv2.resize(
        heatmap,
        (299, 299)
    )

    # Convert to 0-255
    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    # Apply color map
    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    # Convert BGR to RGB
    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    # Overlay
    overlay = cv2.addWeighted(
        original,
        0.60,
        heatmap_color,
        0.40,
        0
    )

    return (
        original,
        heatmap,
        heatmap_color,
        overlay
    )


# ============================================================
# ANALYZE FACIAL REGIONS
# ============================================================

def analyze_regions(heatmap):

    h, w = heatmap.shape

    # Define approximate facial regions
    #
    # These are approximate regions, not exact facial
    # landmark locations.

    regions = {

        "Forehead / Upper Face": (
            int(h * 0.00),
            int(h * 0.25),
            int(w * 0.20),
            int(w * 0.80)
        ),

        "Eyes / Eyebrows": (
            int(h * 0.20),
            int(h * 0.43),
            int(w * 0.10),
            int(w * 0.90)
        ),

        "Nose / Center Face": (
            int(h * 0.35),
            int(h * 0.65),
            int(w * 0.25),
            int(w * 0.75)
        ),

        "Mouth / Lips": (
            int(h * 0.55),
            int(h * 0.78),
            int(w * 0.15),
            int(w * 0.85)
        ),

        "Chin / Lower Face": (
            int(h * 0.75),
            int(h * 0.98),
            int(w * 0.20),
            int(w * 0.80)
        ),

        "Left Face": (
            int(h * 0.30),
            int(h * 0.80),
            int(w * 0.00),
            int(w * 0.30)
        ),

        "Right Face": (
            int(h * 0.30),
            int(h * 0.80),
            int(w * 0.70),
            int(w * 1.00)
        )
    }

    scores = {}

    for region, (
        y1,
        y2,
        x1,
        x2
    ) in regions.items():

        region_heatmap = heatmap[
            y1:y2,
            x1:x2
        ]

        if region_heatmap.size > 0:

            scores[region] = float(
                np.mean(region_heatmap)
            )

        else:

            scores[region] = 0.0

    return scores


# ============================================================
# FIND IMPORTANT REGIONS
# ============================================================

def get_artifact_regions(scores):

    # Sort by heatmap importance
    sorted_regions = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Calculate maximum score
    max_score = sorted_regions[0][1]

    important_regions = []

    # Only include regions with meaningful activation
    for region, score in sorted_regions:

        if max_score > 0:

            relative_score = (
                score / max_score
            )

            if relative_score >= 0.65:

                important_regions.append(
                    region
                )

    # Maximum 3 regions
    important_regions = important_regions[:3]

    return important_regions, sorted_regions


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    # --------------------------------------------------------
    # ANALYZE BUTTON
    # --------------------------------------------------------

    with col2:

        if st.button(
            "🔍 Analyze Image",
            use_container_width=True
        ):

            # -----------------------------------------------
            # PREDICTION
            # -----------------------------------------------

            label, confidence, prediction = predict(
                image
            )

            st.markdown(
                "### 🧠 Model Prediction"
            )

            if label == "FAKE":

                st.error(
                    f"⚠️ Prediction: {label}"
                )

            else:

                st.success(
                    f"✅ Prediction: {label}"
                )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )


            # -----------------------------------------------
            # GRAD-CAM
            # -----------------------------------------------

            st.markdown(
                "### 🔥 Grad-CAM Explanation"
            )

            (
                original,
                heatmap,
                heatmap_color,
                overlay
            ) = create_gradcam(
                image,
                prediction
            )

            grad_col1, grad_col2 = st.columns(2)

            with grad_col1:

                st.image(
                    original,
                    caption="Original Image",
                    use_container_width=True
                )

            with grad_col2:

                st.image(
                    overlay,
                    caption="Grad-CAM: Important Regions",
                    use_container_width=True
                )


            # -----------------------------------------------
            # REGION ANALYSIS
            # -----------------------------------------------

            st.markdown(
                "### 📍 Potential Artifact / Influential Regions"
            )

            scores = analyze_regions(
                heatmap
            )

            important_regions, sorted_regions = (
                get_artifact_regions(scores)
            )


            if label == "FAKE":

                if important_regions:

                    region_text = ", ".join(
                        important_regions
                    )

                    st.warning(
                        f"⚠️ The model's strongest "
                        f"activation is concentrated around: "
                        f"**{region_text}**."
                    )

                    st.write(
                        "These regions are potential areas "
                        "where manipulation-related visual "
                        "patterns may have influenced the "
                        "model's Fake prediction."
                    )

                else:

                    st.info(
                        "No strongly concentrated facial "
                        "region was identified by Grad-CAM."
                    )

            else:

                if important_regions:

                    region_text = ", ".join(
                        important_regions
                    )

                    st.info(
                        f"The model mainly focused on: "
                        f"**{region_text}**."
                    )

                    st.write(
                        "These are regions that influenced "
                        "the model's Real prediction."
                    )

                else:

                    st.info(
                        "No strongly concentrated region "
                        "was identified."
                    )


            # -----------------------------------------------
            # REGION SCORES
            # -----------------------------------------------

            st.markdown(
                "### 📊 Region Importance"
            )

            for region, score in sorted_regions:

                st.write(
                    f"**{region}**"
                )

                st.progress(
                    min(
                        int(score * 100),
                        100
                    )
                )


            # -----------------------------------------------
            # INTERPRETATION
            # -----------------------------------------------

            st.markdown(
                "### 💡 Explanation"
            )

            if label == "FAKE":

                st.write(
                    "The XceptionNet model classified the "
                    "image as potentially fake. The Grad-CAM "
                    "heatmap shows the regions that contributed "
                    "most strongly to this prediction."
                )

            else:

                st.write(
                    "The XceptionNet model classified the "
                    "image as potentially real. The Grad-CAM "
                    "heatmap shows the regions that contributed "
                    "most strongly to this prediction."
                )


            # -----------------------------------------------
            # IMPORTANT DISCLAIMER
            # -----------------------------------------------

            st.caption(
                "⚠️ Note: The highlighted regions represent "
                "areas that influenced the model's prediction. "
                "They should not be interpreted as definitive "
                "proof that an actual manipulation artifact "
                "exists in that specific region."
            )
