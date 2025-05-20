import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import time

# Load model
try:
    model = load_model("drowsiness_eye_cnn_model.h5")
    print("Model drowsiness_eye_cnn_model loaded successfully!")
except:
    try:
        model = load_model("best_model.keras")
        print("Model best_model.keras loaded successfully!")
    except:
        print("Error: Model not found!")
        exit()

# Declare classes
class_names = {0: 'Closed', 1: 'Open'}

# Configure parameters
detection_threshold = 10  # Number of consecutive frames detecting drowsiness to trigger alert
frame_counter = 0
drowsy_status = False

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_array = cv2.resize(frame, (224,224))
    normalization_layer = layers.Normalization()
    normalization_layer.adapt(frame_array)
    frame_array = normalization_layer(frame_array)
    
    return np.expand_dims(frame_array, axis=0)

def get_prediction(preprocessed_frame):
    # Predict on preprocessed frame
    prediction = model.predict(preprocessed_frame, verbose=0)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx]
    class_name = class_names[class_idx]
    print(prediction)
    return class_name, confidence, class_idx

def is_drowsy(class_idx):
    # Determine drowsy state: closed eyes (0) are considered drowsy
    return class_idx == 0

# Variable for FPS tracking
prev_time = time.time()
fps_counter = 0
fps = 0

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame from camera")
        break
        
    # Flip frame horizontally for easier front camera usage
    frame = cv2.flip(frame, 1)
    
    # Calculate FPS
    fps_counter += 1
    if (time.time() - prev_time) > 1.0:
        fps = fps_counter
        fps_counter = 0
        prev_time = time.time()
        
    # Display FPS
    cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Preprocess frame
    processed_frame = preprocess_frame(frame)
    
    # Get prediction
    class_name, confidence, class_idx = get_prediction(processed_frame)
    
    # Check drowsy state
    if is_drowsy(class_idx):
        frame_counter += 1
    else:
        frame_counter = max(0, frame_counter - 1)  # Decrease counter gradually
        
    # Display status information
    cv2.putText(frame, f"Status: {class_name} ({confidence:.2f})", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
    # Display drowsiness status bar
    drowsiness_bar_length = int((frame_counter / detection_threshold) * 100)
    drowsiness_bar_length = min(100, drowsiness_bar_length)
    cv2.rectangle(frame, (10, 90), (10 + 100, 110), (0, 0, 255), 1)
    cv2.rectangle(frame, (10, 90), (10 + drowsiness_bar_length, 110), (0, 0, 255), -1)
    
    # Trigger alert when drowsiness detected
    if frame_counter >= detection_threshold:
        # Display alert on frame
        cv2.putText(frame, "DROWSINESS ALERT!", (frame.shape[1]//2 - 140, frame.shape[0]//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)

        drowsy_status = True
    else:
        drowsy_status = False
    
    # Show frame
    cv2.imshow("Drowsiness Detection", frame)
    
    # Check for exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
