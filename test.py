import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model("handwritten_digit_model.h5")

try:
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        raise Exception("Could not open video capture device")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Define region of interest (ROI) - center square of the frame
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

        # Resize to match MNIST format (28x28)
        digit = cv2.resize(thresh, (28, 28))

        # Normalize the pixel values
        digit = digit / 255.0

        # Reshape for model input (1x28x28x1)
        digit = np.reshape(digit, (1, 28, 28, 1))

        # Make prediction
        prediction = model.predict(digit, verbose=0)
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # Display prediction and confidence
        cv2.putText(frame, f"Digit: {predicted_digit} ({confidence:.1f}%)", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show both the main frame and the processed ROI
        cv2.imshow('Webcam', frame)
        cv2.imshow('Processed ROI', thresh)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()
    for i in range(4):  # Ensure windows are destroyed
        cv2.waitKey(1)