
from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64

app = Flask(__name__)
model = load_model("handwritten_digit_model.h5")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_frame', methods=['POST'])
def process_frame():
    # Get frame from request
    file = request.files['frame']
    npimg = np.fromstring(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Process frame
    height, width = frame.shape[:2]
    square_size = min(height, width) // 2
    x1 = (width - square_size) // 2
    y1 = (height - square_size) // 2
    x2 = x1 + square_size
    y2 = y1 + square_size

    # Draw rectangle for ROI
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Extract and process ROI
    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

    # Predict digit
    digit = cv2.resize(thresh, (28, 28))
    digit = digit / 255.0
    digit = np.reshape(digit, (1, 28, 28, 1))
    prediction = model.predict(digit)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # Add prediction text
    cv2.putText(frame, f"Digit: {predicted_digit} ({confidence:.1f}%)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Convert to base64
    _, buffer = cv2.imencode('.jpg', frame)
    processed_frame = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'processed_frame': processed_frame})


if __name__ == '__main__':
    app.run(debug=True)
