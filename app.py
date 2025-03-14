import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Load OpenCV's pre-trained deep learning face detection model
prototxt_path = "deploy.prototxt"
caffemodel_path = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

st.set_page_config(page_title="Face Detection App", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Face Detection", "About"])

if page == "Face Detection":
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Face_icon_black.svg/1024px-Face_icon_black.svg.png", width=100)
    st.title("OpenCV Deep Learning Based Face Detection")
    
    uploaded_file = st.file_uploader("Choose a File", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        h, w = image_np.shape[:2]

        # Convert to blob and detect faces
        blob = cv2.dnn.blobFromImage(image_np, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(image_np, (startX, startY), (endX, endY), (0, 255, 0), 2)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        with col2:
            st.image(image_np, caption="Output Image", use_column_width=True)

        # Save processed image for download
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(temp_file.name, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        st.markdown(f"[Download Output Image](data:image/jpg;base64,{temp_file.name})")

elif page == "About":
    st.title("About This App")
    st.write("This app detects faces using OpenCV's deep learning model. Upload an image and it will detect faces in it.")
