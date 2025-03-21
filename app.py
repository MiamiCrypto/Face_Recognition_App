import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import tempfile

# Streamlit UI Setup (MUST BE FIRST Streamlit command)
st.set_page_config(page_title="Face Detection App", layout="wide")

# Load OpenCV's pre-trained deep learning face detection model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
prototxt_path = os.path.join(BASE_DIR, "deploy.prototxt")
caffemodel_path = os.path.join(BASE_DIR, "res10_300x300_ssd_iter_140000_fp16.caffemodel")

# Load the model with error handling
try:
    net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
    if net.empty():
        st.error("Error: Model failed to load. Ensure 'deploy.prototxt' and 'res10_300x300_ssd_iter_140000_fp16.caffemodel' are correct.")
        st.stop()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()

# Add Logo at the Top of the Sidebar
logo_path = os.path.join(BASE_DIR, "Mask.png")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Face Detection", "About"])

if page == "Face Detection":
    st.title("OpenCV Deep Learning Based Face Detection")
    
    uploaded_file = st.file_uploader("Choose a File", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        if len(image_np.shape) == 2:  # Convert grayscale to RGB
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

        h, w = image_np.shape[:2]

        # Convert to blob and detect faces
        blob = cv2.dnn.blobFromImage(image_np, scalefactor=1.0, size=(300, 300), 
                                     mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)
        st.write(f"Blob shape: {blob.shape}")  # Debugging log
        net.setInput(blob)

        try:
            detections = net.forward()
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    cv2.rectangle(image_np, (startX, startY), (endX, endY), (0, 255, 0), 2)

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            with col2:
                st.image(image_np, caption="Output Image", use_container_width=True)

            # # Save processed image for download
            # temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            # cv2.imwrite(temp_file.name, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
            # st.markdown(f"[Download Output Image](data:image/jpg;base64,{temp_file.name})")
        # Save processed image for download
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cv2.imwrite(temp_file.name, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            with open(temp_file.name, "rb") as file:
                btn = st.download_button(
                    label="Download Output Image",
                    data=file,
                    file_name="output.jpg",
                    mime="image/jpeg"
                )

        except cv2.error as e:
            st.error(f"OpenCV Error: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected Error: {e}")

elif page == "About":
    st.title("About This App")
    st.write("This app detects faces using OpenCV's deep learning model. Upload an image and it will detect faces in it.")
