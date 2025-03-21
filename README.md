Face Feature Detection App

Overview

The Face Feature Detection App is a powerful and interactive tool built using Streamlit and OpenCV. It allows users to detect faces, eyes, and mouths within images, offering a variety of customization options for detection sensitivity and bounding box colors. The app leverages deep learning models to ensure accurate and efficient detection.

Features

Face Detection: Uses OpenCV's deep learning model (Caffe) for robust face detection.

Eye and Mouth Detection: Uses Haar cascades for detecting eyes and mouths.

Confidence Threshold Adjustment: Customize detection sensitivity.

Bounding Box Color: Choose any color for detected feature boxes.

Confidence Scores: Toggle visibility of confidence scores.

Side-by-Side Comparison: View uploaded and processed images side by side.

Download Output: Save the processed image with bounding boxes.

Installation

Prerequisites

Python 3.8+

pip

Clone the Repository

git clone https://github.com/MiamiCrypto/Face_Recognition_App.git
cd Face_Recognition_App

Install Dependencies

pip install -r requirements.txt

Usage

Running the App

streamlit run app.py

Upload an Image

Drag and drop or browse to select an image file (.jpg, .jpeg, .png).

The image will be displayed on the left, and the processed output on the right.

Adjust Settings

Choose the detection type (Face, Eyes, Mouth) from the dropdown.

Set the confidence threshold to control detection sensitivity.

Select a bounding box color.

Toggle confidence scores on or off.

Download the output image after processing.

Models Used

Face Detection: Caffe model (res10_300x300_ssd_iter_140000_fp16.caffemodel)

Eye and Mouth Detection: Haar cascades (haarcascade_eye.xml, haarcascade_smile.xml)

License

MIT License

Acknowledgements

Streamlit for the interactive UI

OpenCV for image processing and detection models

Contact

For questions or suggestions, please contact MiamiCrypto.
