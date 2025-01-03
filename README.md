# Handwritten Digit Recognition Using Webcam

## Overview
This project implements real-time handwritten digit recognition using a webcam. It uses a Convolutional Neural Network (CNN) trained on the MNIST dataset to recognize handwritten digits (0-9) shown to the webcam.

## Features
- Real-time digit recognition through webcam
- Visual feedback with bounding box
- Confidence score display
- Pre-trained CNN model using MNIST dataset
- Live video preprocessing and prediction

## Setup Instructions
1. Install required dependencies:
    ```bash
    pip install flask opencv-python tensorflow numpy
    ```
2. Run the application:
    ```bash
    python app.py
    ```

## Project Structure
- [app.py](http://_vscodecontentref_/1): Main Flask application that handles the webcam feed and digit recognition.
- [handwritten_digit_model.h5](http://_vscodecontentref_/2): Pre-trained CNN model for digit recognition.
- [requirements.txt](http://_vscodecontentref_/3): List of dependencies required for the project.
- [static](http://_vscodecontentref_/4): Contains static files (CSS and JavaScript).
  - [style.css](http://_vscodecontentref_/5): Styles for the web application.
  - [main.js](http://_vscodecontentref_/6): JavaScript file (currently empty).
- [templates](http://_vscodecontentref_/7): Contains HTML templates.
  - [index.html](http://_vscodecontentref_/8): Main HTML template for the web application.
- [.gitattributes](http://_vscodecontentref_/9): Git configuration file for handling line endings.

## How It Works
1. The Flask application ([app.py](http://_vscodecontentref_/10)) starts a web server and captures video from the webcam.
2. The video frames are processed in real-time to detect and recognize handwritten digits.
3. The region of interest (ROI) is extracted from the video frame and preprocessed (grayscale, blurred, and thresholded).
4. The preprocessed ROI is resized to 28x28 pixels and fed into the pre-trained CNN model.
5. The model predicts the digit and its confidence score, which are displayed on the video frame.
6. The processed video frames are streamed to the web application, providing real-time feedback.

## Usage Instructions
1. Open a web browser and navigate to `http://127.0.0.1:5000/`.
2. Hold up a handwritten digit (0-9) to your webcam.
3. Position the digit within the green box on the video feed.
4. Ensure the digit is clear and well-lit.
5. The predicted digit and confidence score will appear at the top of the video.

## License
This project is licensed under the MIT License.