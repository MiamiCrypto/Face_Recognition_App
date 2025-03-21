import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import tempfile
import time

# Streamlit UI Setup
st.set_page_config(page_title="Feature Detection App", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; font-size: 3em; margin-bottom: 0.5em; color: #ffffff;'>Face Feature Recognition</h1>
    <h3 style='text-align: center; font-size: 1.5em; color: #555;'>Detect faces, eyes, and smiles with ease!</h3>
    """,
    unsafe_allow_html=True
)


# Load OpenCV's pre-trained deep learning face detection model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
prototxt_path = os.path.join(BASE_DIR, "deploy.prototxt")
caffemodel_path = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000_fp16.caffemodel")

eye_cascade_path = os.path.join(BASE_DIR, "haarcascade_eye.xml")
mouth_cascade_path = os.path.join(BASE_DIR, "haarcascade_smile.xml")

try:
    net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
    #st.success("Face model loaded successfully!")
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)
    #st.success("Eye and mouth models loaded successfully!")
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# Sidebar Settings
logo_path = os.path.join(BASE_DIR, "Mask.png")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)

st.sidebar.title("Settings")
st.sidebar.markdown("Confidence Threshold: Adjusting this value controls the sensitivity of detection. Lower values detect more features, but may include false positives. Higher values are more selective.")
detection_type = st.sidebar.selectbox("Choose Detection Type", ["Face", "Eyes", "Mouth"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
bbox_color = st.sidebar.color_picker("Bounding Box Color", "#00FF00")
show_confidence = st.sidebar.checkbox("Show Confidence Scores", value=True)

# Upload Image
uploaded_file = st.file_uploader("Choose a File", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    h, w = image_np.shape[:2]

    if detection_type == "Face":
        blob = cv2.dnn.blobFromImage(image_np, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                color = tuple(int(bbox_color[i:i+2], 16) for i in (1, 3, 5))
                cv2.rectangle(image_np, (startX, startY), (endX, endY), color, 2)
    elif detection_type == "Eyes":
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            color = tuple(int(bbox_color[i:i+2], 16) for i in (1, 3, 5))
            cv2.rectangle(image_np, (ex, ey), (ex+ew, ey+eh), color, 2)
    elif detection_type == "Mouth":
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        mouths = mouth_cascade.detectMultiScale(gray, 1.7, 11)
        for (mx, my, mw, mh) in mouths:
            color = tuple(int(bbox_color[i:i+2], 16) for i in (1, 3, 5))
            cv2.rectangle(image_np, (mx, my), (mx+mw, my+mh), color, 2)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    with col2:
        st.image(image_np, caption="Output Image", use_container_width=True)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temp_file.name, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    with open(temp_file.name, "rb") as file:
        st.download_button("Download Output Image", data=file, file_name="output.jpg", mime="image/jpeg")

# # About Section
# if st.sidebar.button("About"):
#     st.sidebar.markdown("### Feature Detection App\nThis application uses OpenCV's deep learning models to detect faces, eyes, and mouths.\n- Adjust the confidence threshold to control sensitivity.\n- Choose bounding box colors to customize output.\n- Use the download button to save the output image.\n\n**Developed using Streamlit and OpenCV.**")

# About Section
if st.sidebar.button("About"):
    st.sidebar.markdown("### Face Feature Recognition App\nThis application uses OpenCV's deep learning models to detect faces, eyes, and mouths.\n- Adjust the confidence threshold to control sensitivity.\n- Choose bounding box colors to customize output.\n- Use the download button to save the output image.\n\n**Developed using Streamlit and OpenCV.**")

