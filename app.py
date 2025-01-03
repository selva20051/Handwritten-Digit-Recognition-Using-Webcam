from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("handwritten_digit_model.h5")
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Get frame dimensions
        height, width = frame.shape[:2]
        square_size = min(height, width) // 2
        x1 = (width - square_size) // 2
        y1 = (height - square_size) // 2
        x2 = x1 + square_size
        y2 = y1 + square_size
        
        # Draw ROI rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        roi = frame[y1:y2, x1:x2]

        # Process ROI
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Make prediction
        digit = cv2.resize(thresh, (28, 28))
        digit = digit / 255.0
        digit = np.reshape(digit, (1, 28, 28, 1))
        prediction = model.predict(digit, verbose=0)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # Add text to frame
        cv2.putText(frame, f"Digit: {predicted_digit} ({confidence:.1f}%)", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)