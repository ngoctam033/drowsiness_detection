import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import time

from scipy.spatial import distance
from imutils import face_utils
import dlib

# Load model
try:
    eye_model = load_model("drowsiness_eye_cnn_model.h5")
    print("Model drowsiness_eye_cnn_model loaded successfully!")
except:
    try:
        eye_model = load_model("best_model.keras")
        print("Model best_model.keras loaded successfully!")
    except:
        print("Error: Eye Model not found!")
        exit()

try:
    mouth_model = load_model("mouth_yawn_model.h5")
    print("Model mouth_yawn_model loaded successfully!")
except:
    try:
        mouth_model = load_model("best_model.keras")
        print("Model best_model.keras loaded successfully!")
    except:
        print("Error: Mouth Model not found!")
        exit()

# Declare classes
class_names_eye = {0: 'Closed', 1: 'Open'}
class_names_mouth = {0: 'Yawn', 1: 'NoYawn'}

thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

(lMStart, lMEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# Initialize normalization layer
normalization_layer = layers.Normalization()

# Configure parameters
detection_threshold = 10  # Number of consecutive frames detecting drowsiness to trigger alert
frame_counter = 0
drowsy_status = False

# Variable for FPS tracking
prev_time = time.time()
fps_counter = 0
fps = 0
    
def extract_eye_region(eye_points, frame):
    # Create pounding box around the eye points
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

def preprocess_eye_and_mouth(eye_region):
    # Check if eye region is valid
    if eye_region is None or eye_region.size == 0:
        return None
        
    # Change color space from BGR to Gray
    eye_gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    
    # Resize the eye region to the input size of the model (224x224)
    eye_resized = cv2.resize(eye_gray, (224, 224))
    
    # Normalize the eye region
    normalization_layer.adapt(eye_resized)
    normalized_eye = normalization_layer(eye_resized)
    
    # Add batch dimension
    return np.expand_dims(normalized_eye, axis=0)

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_array = cv2.resize(frame, (224,224))
    normalization_layer.adapt(frame_array)
    frame_array = normalization_layer(frame_array)
    
    return np.expand_dims(frame_array, axis=0)

def get_prediction(preprocessed_frame, model, mouth="eye"):
    # Predict on preprocessed frame
    prediction = model.predict(preprocessed_frame, verbose=0)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx]
    if mouth == "eye":
        class_name = class_names_eye[class_idx]
    else:
        if confidence < 0.75:
            class_idx = 0
        class_name = class_names_mouth[class_idx]
    # class_name = class_names_eye[class_idx]
    return class_name, confidence, class_idx

def is_drowsy(class_idx):
    # Determine drowsy state: closed eyes (0) are considered drowsy
    return class_idx == 0

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open camera")
    exit()

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame from camera")
        break
    class_idx = 1  # Default to 'Open'
    class_name = "Open"   
    # Flip frame horizontally for easier front camera usage
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        mouth = shape[lMStart:lMEnd]
        
        # Cut and display the eye regions
        left_eye_region, left_x, left_y, left_w, left_h = extract_eye_region(leftEye, frame)
        right_eye_region, right_x, right_y, right_w, right_h = extract_eye_region(rightEye, frame)

        # Cut and display the mouth region
        mouth_region, mouth_x, mouth_y, mouth_w, mouth_h = extract_mouth_region(mouth, frame)
        
        # Display eye regions (for debugging)
        if left_eye_region is not None and right_eye_region is not None and mouth_region is not None:
            # Resize
            display_left_eye = cv2.resize(left_eye_region, (100, 100))
            display_right_eye = cv2.resize(right_eye_region, (100, 100))

            display_mouth = cv2.resize(mouth_region, (100, 100))    

            # Display eye regions on the frame
            frame[10:110, frame.shape[1]-110:frame.shape[1]-10] = display_right_eye
            frame[10:110, frame.shape[1]-220:frame.shape[1]-120] = display_left_eye

            # Resize and display mouth region
            frame[170:270, frame.shape[1]-220:frame.shape[1]-120] = display_mouth  

            # Process and predict for left eye
            processed_left_eye = preprocess_eye_and_mouth(left_eye_region)
            if processed_left_eye is not None:
                left_class_name, left_confidence, left_class_idx = get_prediction(processed_left_eye, eye_model)
                
                # Display result for left eye
                cv2.putText(frame, f"Left: {left_class_name}", (frame.shape[1]-220, 130), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Process and predict for right eye
            processed_right_eye = preprocess_eye_and_mouth(right_eye_region)
            if processed_right_eye is not None:
                right_class_name, right_confidence, right_class_idx = get_prediction(processed_right_eye, eye_model)
                
                # Display result for right eye
                cv2.putText(frame, f"Right: {right_class_name}", (frame.shape[1]-110, 130), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
            # Process and predict for mouth
            processed_mouth = preprocess_eye_and_mouth(mouth_region)            
            if processed_mouth is not None:
                mouth_class_name, mouth_confidence, mouth_class_idx = get_prediction(processed_mouth, mouth_model, mouth="mouth")
                
                # Display result for left eye
                cv2.putText(frame, f"Mouth: {mouth_class_name}", (frame.shape[1]-220, 285), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)            
            # Determine drowsiness state based on both eyes
            if processed_left_eye is not None and processed_right_eye is not None and processed_mouth is not None:
                eye_confidence = max(left_confidence, right_confidence)
                # Check if both eyes are closed
                if left_class_idx == 0 or right_class_idx == 0 or mouth_class_idx == 0:
                    class_idx = 0
                    class_name = "Closed"
                    confidence = max(left_confidence, right_confidence, mouth_confidence)
                else:
                    class_idx = 1
                    class_name = "Open" 
                    confidence = (left_confidence + right_confidence + mouth_confidence) / 3.0

    # Calculate FPS
    fps_counter += 1
    if (time.time() - prev_time) > 1.0:
        fps = fps_counter
        fps_counter = 0
        prev_time = time.time()
        
    # Display FPS
    cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Check drowsy state
    if is_drowsy(class_idx):
        frame_counter += 1
    else:
        frame_counter = max(0, frame_counter - 1)  # Decrease counter gradually
        
    # Display status information
    cv2.putText(frame, f"Eye: {class_name} ({eye_confidence:.2f})", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.putText(frame, f"Mouth: {mouth_class_name} ({mouth_confidence:.2f})", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)

    cv2.putText(frame, f"Status: {class_name} ({confidence:.2f})", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if is_drowsy(class_idx) else (255, 0, 0), 2)

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
