import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array


st.set_page_config(
    page_title="AI Deepfake Detector",
    page_icon="🔍",
    layout="wide"
)


@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "deepfake_xai_model.keras"
    )


model = load_model()


st.title("🔍 AI Deepfake Image Detection")
st.subheader(
    "Explainable AI using XceptionNet"
)


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


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

    return label, confidence


if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with col2:

        if st.button(
            "🔍 Analyze Image"
        ):

            label, confidence = predict(
                image
            )

            if label == "FAKE":

                st.error(
                    f"Prediction: {label}"
                )

            else:

                st.success(
                    f"Prediction: {label}"
                )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )
