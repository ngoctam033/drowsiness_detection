import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import time

from scipy.spatial import distance
from imutils import face_utils
import dlib

# Load models
def load_models():
    eye_model = None
    mouth_model = None
    
    # Load eye model
    try:
        eye_model = load_model("drowsiness_eye_cnn_model.h5")
        print("Eye model drowsiness_eye_cnn_model.h5 loaded successfully!")
    except:
        try:
            eye_model = load_model("best_model.keras")
            print("Eye model best_model.keras loaded successfully!")
        except:
            print("Warning: Eye model not found!")
    
    # Load mouth model
    try:
        mouth_model = load_model("mouth_yawn_model.h5")
        print("Mouth model mouth_yawn_model.h5 loaded successfully!")
    except:
        try:
            mouth_model = load_model("mouthYawn_model.keras")
            print("Mouth model mouthYawn_model.keras loaded successfully!")
        except:
            print("Warning: Mouth model not found!")
    
    if eye_model is None and mouth_model is None:
        print("Error: No models found!")
        exit()
    
    return eye_model, mouth_model

# Load models
eye_model, mouth_model = load_models()

# Declare classes
eye_class_names = {0: 'Closed', 1: 'Open'}
mouth_class_names = {0: 'NoYawn', 1: 'Yawn'}

# Dlib face detection and landmark prediction
thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Face landmark indices
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# Initialize normalization layers
eye_normalization_layer = layers.Normalization()
mouth_normalization_layer = layers.Normalization()

# Configure parameters
detection_threshold = 10  # Number of consecutive frames detecting drowsiness to trigger alert
eye_frame_counter = 0
mouth_frame_counter = 0
overall_frame_counter = 0
drowsy_status = False

# Variable for FPS tracking
prev_time = time.time()
fps_counter = 0
fps = 0

# Tracking variables for stable detection
eye_predictions_history = []
mouth_predictions_history = []
history_length = 5  # Number of recent predictions to consider

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    # Calculate vertical distances
    A = distance.euclidean(mouth[2], mouth[10])  # 51, 59
    B = distance.euclidean(mouth[4], mouth[8])   # 53, 57
    
    # Calculate horizontal distance
    C = distance.euclidean(mouth[0], mouth[6])   # 49, 55
    
    # Calculate MAR
    mar = (A + B) / (2.0 * C)
    return mar
    
def extract_eye_region(eye_points, frame):
    # Create bounding box around the eye points
    (x, y, w, h) = cv2.boundingRect(np.array(eye_points))
    
    # Extend the bounding box to include some padding
    padding = 10
    x_start = max(0, x-padding)
    y_start = max(0, y-padding)
    x_end = min(frame.shape[1], x+w+padding)
    y_end = min(frame.shape[0], y+h+padding)
    
    eye_region = frame[y_start:y_end, x_start:x_end]
    
    # Check if the eye region is valid
    if eye_region.size == 0:
        return None, None, None, None, None
    
    return eye_region, x_start, y_start, x_end-x_start, y_end-y_start

def extract_mouth_region(mouth_points, frame):
    # Create bounding box around the mouth points
    (x, y, w, h) = cv2.boundingRect(np.array(mouth_points))
    
    # Extend the bounding box to include some padding
    padding = 15
    x_start = max(0, x-padding)
    y_start = max(0, y-padding)
    x_end = min(frame.shape[1], x+w+padding)
    y_end = min(frame.shape[0], y+h+padding)
    
    mouth_region = frame[y_start:y_end, x_start:x_end]
    
    # Check if the mouth region is valid
    if mouth_region.size == 0:
        return None, None, None, None, None
    
    return mouth_region, x_start, y_start, x_end-x_start, y_end-y_start

def preprocess_eye(eye_region):
    # Check if eye region is valid
    if eye_region is None or eye_region.size == 0:
        return None
        
    # Change color space from BGR to RGB
    eye_rgb = cv2.cvtColor(eye_region, cv2.COLOR_BGR2RGB)
    
    # Resize the eye region to the input size of the model (224x224)
    eye_resized = cv2.resize(eye_rgb, (224, 224))
    
    # Normalize the eye region
    eye_normalization_layer.adapt(eye_resized)
    normalized_eye = eye_normalization_layer(eye_resized)
    
    # Add batch dimension
    return np.expand_dims(normalized_eye, axis=0)

def preprocess_mouth(mouth_region):
    # Check if mouth region is valid
    if mouth_region is None or mouth_region.size == 0:
        return None
        
    # Change color space from BGR to RGB
    mouth_rgb = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2RGB)
    
    # Resize the mouth region to the input size of the model (224x224)
    mouth_resized = cv2.resize(mouth_rgb, (224, 224))
    
    # Normalize the mouth region
    mouth_normalization_layer.adapt(mouth_resized)
    normalized_mouth = mouth_normalization_layer(mouth_resized)
    
    # Add batch dimension
    return np.expand_dims(normalized_mouth, axis=0)

def get_eye_prediction(preprocessed_eye):
    if preprocessed_eye is None or eye_model is None:
        return "Unknown", 0.0, -1
    
    # Predict on preprocessed eye
    prediction = eye_model.predict(preprocessed_eye, verbose=0)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx]
    class_name = eye_class_names[class_idx]
    return class_name, confidence, class_idx

def get_mouth_prediction(preprocessed_mouth):
    if preprocessed_mouth is None or mouth_model is None:
        return "Unknown", 0.0, -1
    
    # Predict on preprocessed mouth
    prediction = mouth_model.predict(preprocessed_mouth, verbose=0)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx]
    class_name = mouth_class_names[class_idx]
    return class_name, confidence, class_idx

def get_stable_prediction(predictions_history, current_prediction):
    """Get stable prediction based on recent history"""
    predictions_history.append(current_prediction)
    
    # Keep only recent predictions
    if len(predictions_history) > history_length:
        predictions_history.pop(0)
    
    # Count occurrences of each prediction
    if len(predictions_history) >= 3:
        from collections import Counter
        counter = Counter(predictions_history)
        most_common = counter.most_common(1)[0]
        return most_common[0]
    else:
        return current_prediction

def determine_drowsiness_state(eye_class_idx, mouth_class_idx, eye_confidence, mouth_confidence):
    """
    Determine overall drowsiness state based on eye and mouth predictions
    """
    eyes_closed = (eye_class_idx == 0)
    mouth_yawning = (mouth_class_idx == 1)
    
    if eyes_closed and mouth_yawning:
        return "Very Drowsy", max(eye_confidence, mouth_confidence), 2
    elif eyes_closed:
        return "Drowsy (Eyes)", eye_confidence, 1
    elif mouth_yawning:
        return "Drowsy (Yawn)", mouth_confidence, 1
    else:
        return "Alert", min(eye_confidence, mouth_confidence) if eye_confidence > 0 and mouth_confidence > 0 else max(eye_confidence, mouth_confidence), 0

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()
else:
    print("Camera opened successfully")

# Main detection loop
while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame from camera")
        break
        
    # Flip frame horizontally for easier front camera usage
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    
    # Initialize default values
    eye_class_name, eye_confidence, eye_class_idx = "Unknown", 0.0, -1
    mouth_class_name, mouth_confidence, mouth_class_idx = "Unknown", 0.0, -1
    
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        
        # Extract eye regions
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        
        # Extract mouth region
        mouth = shape[mStart:mEnd]
        mar = mouth_aspect_ratio(mouth)
        
        # Draw eye and mouth contours
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (255, 0, 0), 1)
        
        # Extract and process eye regions
        left_eye_region, left_x, left_y, left_w, left_h = extract_eye_region(leftEye, frame)
        right_eye_region, right_x, right_y, right_w, right_h = extract_eye_region(rightEye, frame)
        
        # Extract and process mouth region
        mouth_region, mouth_x, mouth_y, mouth_w, mouth_h = extract_mouth_region(mouth, frame)
        
        # Display eye and mouth regions (for debugging)
        if left_eye_region is not None and right_eye_region is not None:
            # Resize and display eye regions
            display_left_eye = cv2.resize(left_eye_region, (80, 60))
            display_right_eye = cv2.resize(right_eye_region, (80, 60))
            
            # Display eye regions on the frame
            frame[10:70, frame.shape[1]-90:frame.shape[1]-10] = display_right_eye
            frame[10:70, frame.shape[1]-180:frame.shape[1]-100] = display_left_eye
            
            # Process and predict for both eyes
            processed_left_eye = preprocess_eye(left_eye_region)
            processed_right_eye = preprocess_eye(right_eye_region)
            
            if processed_left_eye is not None and processed_right_eye is not None:
                left_class_name, left_confidence, left_class_idx = get_eye_prediction(processed_left_eye)
                right_class_name, right_confidence, right_class_idx = get_eye_prediction(processed_right_eye)
                
                # Determine combined eye state (if either eye is closed, consider closed)
                if left_class_idx == 0 or right_class_idx == 0:
                    eye_class_idx = 0
                    eye_class_name = "Closed"
                    eye_confidence = max(left_confidence, right_confidence)
                else:
                    eye_class_idx = 1
                    eye_class_name = "Open" 
                    eye_confidence = (left_confidence + right_confidence) / 2.0
                
                # Get stable eye prediction
                stable_eye_prediction = get_stable_prediction(eye_predictions_history, eye_class_idx)
                if stable_eye_prediction != eye_class_idx:
                    eye_class_idx = stable_eye_prediction
                    eye_class_name = eye_class_names[eye_class_idx]
        
        # Display mouth region and process
        if mouth_region is not None:
            # Resize and display mouth region
            display_mouth = cv2.resize(mouth_region, (120, 80))
            frame[80:160, frame.shape[1]-130:frame.shape[1]-10] = display_mouth
            
            # Process and predict for mouth
            processed_mouth = preprocess_mouth(mouth_region)
            
            if processed_mouth is not None:
                mouth_class_name, mouth_confidence, mouth_class_idx = get_mouth_prediction(processed_mouth)
                
                # Get stable mouth prediction
                stable_mouth_prediction = get_stable_prediction(mouth_predictions_history, mouth_class_idx)
                if stable_mouth_prediction != mouth_class_idx:
                    mouth_class_idx = stable_mouth_prediction
                    mouth_class_name = mouth_class_names[mouth_class_idx]
    
    # Calculate FPS
    fps_counter += 1
    if (time.time() - prev_time) > 1.0:
        fps = fps_counter
        fps_counter = 0
        prev_time = time.time()
    
    # Determine overall drowsiness state
    drowsiness_state, overall_confidence, drowsiness_level = determine_drowsiness_state(
        eye_class_idx, mouth_class_idx, eye_confidence, mouth_confidence
    )
    
    # Update frame counters based on individual detections
    if eye_class_idx == 0:  # Eyes closed
        eye_frame_counter += 1
    else:
        eye_frame_counter = max(0, eye_frame_counter - 1)
    
    if mouth_class_idx == 1:  # Yawning
        mouth_frame_counter += 1
    else:
        mouth_frame_counter = max(0, mouth_frame_counter - 1)
    
    # Overall drowsiness counter
    if drowsiness_level > 0:
        overall_frame_counter += 1
    else:
        overall_frame_counter = max(0, overall_frame_counter - 1)
    
    # Display information on frame
    cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display eye status
    if eye_model is not None:
        cv2.putText(frame, f"Eyes: {eye_class_name} ({eye_confidence:.2f})", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"L: {left_class_name if 'left_class_name' in locals() else 'N/A'}", 
                    (frame.shape[1]-180, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(frame, f"R: {right_class_name if 'right_class_name' in locals() else 'N/A'}", 
                    (frame.shape[1]-90, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    # Display mouth status
    if mouth_model is not None:
        cv2.putText(frame, f"Mouth: {mouth_class_name} ({mouth_confidence:.2f})", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"Mouth: {mouth_class_name}", (frame.shape[1]-130, 170), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    
    # Display overall status
    cv2.putText(frame, f"Status: {drowsiness_state}", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Display drowsiness progress bars
    bar_width = 200
    bar_height = 15
    bar_y_start = 140
    
    # Eye drowsiness bar
    if eye_model is not None:
        eye_bar_length = int((eye_frame_counter / detection_threshold) * bar_width)
        eye_bar_length = min(bar_width, eye_bar_length)
        cv2.rectangle(frame, (10, bar_y_start), (10 + bar_width, bar_y_start + bar_height), (128, 128, 128), 1)
        cv2.rectangle(frame, (10, bar_y_start), (10 + eye_bar_length, bar_y_start + bar_height), (0, 0, 255), -1)
        cv2.putText(frame, "Eye Alert", (10, bar_y_start - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Mouth drowsiness bar
    if mouth_model is not None:
        mouth_bar_y = bar_y_start + 25
        mouth_bar_length = int((mouth_frame_counter / detection_threshold) * bar_width)
        mouth_bar_length = min(bar_width, mouth_bar_length)
        cv2.rectangle(frame, (10, mouth_bar_y), (10 + bar_width, mouth_bar_y + bar_height), (128, 128, 128), 1)
        cv2.rectangle(frame, (10, mouth_bar_y), (10 + mouth_bar_length, mouth_bar_y + bar_height), (0, 165, 255), -1)
        cv2.putText(frame, "Yawn Alert", (10, mouth_bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Overall drowsiness bar
    overall_bar_y = bar_y_start + 50
    overall_bar_length = int((overall_frame_counter / detection_threshold) * bar_width)
    overall_bar_length = min(bar_width, overall_bar_length)
    cv2.rectangle(frame, (10, overall_bar_y), (10 + bar_width, overall_bar_y + bar_height), (128, 128, 128), 1)
    cv2.rectangle(frame, (10, overall_bar_y), (10 + overall_bar_length, overall_bar_y + bar_height), (255, 0, 255), -1)
    cv2.putText(frame, "Overall Alert", (10, overall_bar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Trigger alerts
    alert_triggered = False
    alert_color = (0, 0, 255)  # Red
    alert_text = ""
    
    if overall_frame_counter >= detection_threshold:
        if drowsiness_level == 2:  # Very drowsy
            alert_text = "EXTREME DROWSINESS ALERT!"
            alert_color = (0, 0, 255)  # Red
        elif eye_frame_counter >= detection_threshold and mouth_frame_counter >= detection_threshold:
            alert_text = "MULTIPLE DROWSINESS SIGNS!"
            alert_color = (0, 100, 255)  # Orange-red
        elif eye_frame_counter >= detection_threshold:
            alert_text = "EYES CLOSED ALERT!"
            alert_color = (0, 0, 255)  # Red
        elif mouth_frame_counter >= detection_threshold:
            alert_text = "YAWNING DETECTED!"
            alert_color = (0, 165, 255)  # Orange
        else:
            alert_text = "DROWSINESS ALERT!"
            alert_color = (0, 0, 255)  # Red
        
        alert_triggered = True
    
    # Display alert
    if alert_triggered:
        # Flash border
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), alert_color, 8)
        
        # Display alert text
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = frame.shape[0] // 2
        
        # Background rectangle for text
        cv2.rectangle(frame, (text_x - 10, text_y - 30), (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
        cv2.putText(frame, alert_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, alert_color, 3)
        
        drowsy_status = True
    else:
        drowsy_status = False
    
    # Display additional information
    info_y = frame.shape[0] - 60
    cv2.putText(frame, f"Eye Counter: {eye_frame_counter}/{detection_threshold}", (10, info_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Mouth Counter: {mouth_frame_counter}/{detection_threshold}", (10, info_y + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Overall Counter: {overall_frame_counter}/{detection_threshold}", (10, info_y + 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Show frame
    cv2.imshow("Integrated Drowsiness Detection", frame)
    
    # Check for exit key
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):  # Reset counters
        eye_frame_counter = 0
        mouth_frame_counter = 0
        overall_frame_counter = 0
        eye_predictions_history.clear()
        mouth_predictions_history.clear()
        print("Counters reset!")

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Detection stopped.")