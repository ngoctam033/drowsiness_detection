import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
import pandas as pd
import time
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Root directory to search for images
BASE_DIR = "dataset/test"  # Base directory containing both eye and mouth test data

# Load the trained models
def load_models():
    eye_model = None
    mouth_model = None
    
    # Load eye model
    try:
        eye_model = load_model("drowsiness_eye_cnn_model.h5")
        print("Successfully loaded eye model: drowsiness_eye_cnn_model.h5!")
    except:
        try:
            eye_model = load_model("best_model.keras")
            print("Successfully loaded eye model: best_model.keras!")
        except:
            print("Warning: Eye model not found!")
    
    # Load mouth model
    try:
        mouth_model = load_model("mouth_yawn_model.h5")
        print("Successfully loaded mouth model: mouth_yawn_model.h5!")
    except:
        try:
            mouth_model = load_model("mouthYawn_model.keras")
            print("Successfully loaded mouth model: mouthYawn_model.keras!")
        except:
            print("Warning: Mouth model not found!")
    
    if eye_model is None and mouth_model is None:
        print("Error: No models found!")
        exit()
    
    return eye_model, mouth_model

# Declare class names for both models
eye_class_names = {0: 'Closed', 1: 'Open'}
mouth_class_names = {0: 'NoYawn', 1: 'Yawn'}

def find_all_images(base_dir):
    """Find all images in eye and mouth test directories"""
    image_paths = []
    
    # Eye test directories
    eye_dirs = ['Open', 'Closed']
    # Mouth test directories  
    mouth_dirs = ['Yawn', 'NoYawn']
    
    for root, dirs, files in os.walk(base_dir):
        current_dir = os.path.basename(root)
        parent_dir = os.path.basename(os.path.dirname(root))
        
        # Check if current directory is a valid test directory
        is_eye_dir = current_dir in eye_dirs
        is_mouth_dir = current_dir in mouth_dirs
        
        if is_eye_dir or is_mouth_dir:
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    full_path = os.path.join(root, file)
                    image_paths.append({
                        'path': full_path,
                        'type': 'eye' if is_eye_dir else 'mouth',
                        'true_class': current_dir,
                        'parent_dir': parent_dir
                    })
    
    print(f"Found {len(image_paths)} images to analyze")
    print(f"  Eye images: {len([x for x in image_paths if x['type'] == 'eye'])}")
    print(f"  Mouth images: {len([x for x in image_paths if x['type'] == 'mouth'])}")
    
    return image_paths

def preprocess_frame(frame):
    """Preprocess frame for model prediction"""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_array = cv2.resize(frame, (224, 224))
    normalization_layer = layers.Normalization()
    normalization_layer.adapt(frame_array)
    frame_array = normalization_layer(frame_array)
    return np.expand_dims(frame_array, axis=0)

def get_eye_prediction(preprocessed_frame, eye_model):
    """Get eye state prediction"""
    if eye_model is None:
        return None, 0, -1, []
    
    prediction = eye_model.predict(preprocessed_frame, verbose=0)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx]
    class_name = eye_class_names[class_idx]
    
    return class_name, confidence, class_idx, prediction[0]

def get_mouth_prediction(preprocessed_frame, mouth_model):
    """Get mouth state prediction"""
    if mouth_model is None:
        return None, 0, -1, []
    
    prediction = mouth_model.predict(preprocessed_frame, verbose=0)
    class_idx = np.argmax(prediction)
    confidence = prediction[0][class_idx]
    class_name = mouth_class_names[class_idx]
    
    return class_name, confidence, class_idx, prediction[0]

def determine_drowsiness_state(eye_class_idx, mouth_class_idx, eye_confidence, mouth_confidence):
    """
    Determine overall drowsiness state based on eye and mouth predictions
    
    Drowsiness rules:
    1. If eyes are closed (class_idx = 0) -> Drowsy
    2. If mouth is yawning (class_idx = 1) -> Drowsy  
    3. If both eyes closed AND yawning -> Very Drowsy
    4. Otherwise -> Alert
    """
    
    eyes_closed = (eye_class_idx == 0) if eye_class_idx != -1 else False
    mouth_yawning = (mouth_class_idx == 1) if mouth_class_idx != -1 else False
    
    if eyes_closed and mouth_yawning:
        return "Very Drowsy", max(eye_confidence, mouth_confidence)
    elif eyes_closed:
        return "Drowsy (Eyes)", eye_confidence
    elif mouth_yawning:
        return "Drowsy (Yawn)", mouth_confidence
    else:
        return "Alert", min(eye_confidence, mouth_confidence) if eye_confidence > 0 and mouth_confidence > 0 else max(eye_confidence, mouth_confidence)

def main():
    # Load models
    eye_model, mouth_model = load_models()
    
    # Find all images
    image_data = find_all_images(BASE_DIR)
    
    if not image_data:
        print("No images found to analyze")
        return
    
    # Prepare to save results
    results = []
    start_time = time.time()
    
    # Process each image
    for i, img_info in enumerate(image_data):
        try:
            img_path = img_info['path']
            img_type = img_info['type']
            true_class = img_info['true_class']
            
            # Read image
            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Error: Cannot read image {img_path}")
                continue
                
            # Preprocess
            processed_frame = preprocess_frame(frame)
            
            # Get predictions based on image type
            eye_class_name, eye_confidence, eye_class_idx, eye_probs = None, 0, -1, []
            mouth_class_name, mouth_confidence, mouth_class_idx, mouth_probs = None, 0, -1, []
            
            if img_type == 'eye':
                eye_class_name, eye_confidence, eye_class_idx, eye_probs = get_eye_prediction(processed_frame, eye_model)
            elif img_type == 'mouth':
                mouth_class_name, mouth_confidence, mouth_class_idx, mouth_probs = get_mouth_prediction(processed_frame, mouth_model)
            
            # Determine overall drowsiness state
            drowsiness_state, overall_confidence = determine_drowsiness_state(
                eye_class_idx, mouth_class_idx, eye_confidence, mouth_confidence
            )
            
            # Determine if prediction is correct
            if img_type == 'eye':
                predicted_class = eye_class_name
                is_correct = (true_class.lower() == eye_class_name.lower()) if eye_class_name else False
            else:
                predicted_class = mouth_class_name
                is_correct = (true_class.lower() == mouth_class_name.lower()) if mouth_class_name else False
            
            # Save result
            result = {
                'image_path': img_path,
                'image_type': img_type,
                'true_class': true_class,
                'predicted_class': predicted_class,
                'confidence': eye_confidence if img_type == 'eye' else mouth_confidence,
                'drowsiness_state': drowsiness_state,
                'overall_confidence': overall_confidence,
                'is_correct': is_correct,
                'parent_dir': img_info['parent_dir']
            }
            
            # Add model-specific details
            if img_type == 'eye':
                result.update({
                    'eye_class': eye_class_name,
                    'eye_confidence': eye_confidence,
                    'eye_probabilities': eye_probs
                })
            else:
                result.update({
                    'mouth_class': mouth_class_name,
                    'mouth_confidence': mouth_confidence,
                    'mouth_probabilities': mouth_probs
                })
            
            results.append(result)
            
            # Update progress
            if (i + 1) % 10 == 0 or (i + 1) == len(image_data):
                elapsed_time = time.time() - start_time
                images_per_sec = (i + 1) / elapsed_time
                print(f"Processed {i + 1}/{len(image_data)} images ({images_per_sec:.1f} images/sec)")
                
        except Exception as e:
            print(f"Error processing {img_info['path']}: {str(e)}")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Print comprehensive results
    print_comprehensive_results(df)
    
    # Create visualizations
    create_visualizations(df)
    
    # Save results to files
    save_results_to_files(df)

def print_comprehensive_results(df):
    """Print comprehensive analysis results"""
    print("\n" + "="*60)
    print("COMPREHENSIVE DROWSINESS DETECTION RESULTS")
    print("="*60)
    
    # Overall statistics
    print(f"Total images analyzed: {len(df)}")
    
    # Statistics by image type
    print("\nImages by type:")
    type_counts = df['image_type'].value_counts()
    for img_type, count in type_counts.items():
        percentage = count / len(df) * 100
        print(f"  {img_type.capitalize()}: {count} ({percentage:.1f}%)")
    
    # Overall accuracy
    overall_accuracy = df['is_correct'].mean()
    print(f"\nOverall accuracy: {overall_accuracy:.4f} ({df['is_correct'].sum()}/{len(df)})")
    
    # Accuracy by image type
    print("\nAccuracy by image type:")
    for img_type in df['image_type'].unique():
        type_df = df[df['image_type'] == img_type]
        accuracy = type_df['is_correct'].mean()
        correct = type_df['is_correct'].sum()
        total = len(type_df)
        print(f"  {img_type.capitalize()}: {accuracy:.4f} ({correct}/{total})")
    
    # Drowsiness state distribution
    print("\nDrowsiness state distribution:")
    drowsiness_counts = df['drowsiness_state'].value_counts()
    for state, count in drowsiness_counts.items():
        percentage = count / len(df) * 100
        print(f"  {state}: {count} ({percentage:.1f}%)")
    
    # Detailed statistics by true class
    print("\nDetailed statistics by true class:")
    for true_class in df['true_class'].unique():
        class_df = df[df['true_class'] == true_class]
        total = len(class_df)
        correct = class_df['is_correct'].sum()
        accuracy = correct / total if total > 0 else 0
        avg_confidence = class_df['confidence'].mean()
        
        print(f"\n  {true_class}:")
        print(f"    Total images: {total}")
        print(f"    Correct predictions: {correct} ({accuracy*100:.1f}%)")
        print(f"    Average confidence: {avg_confidence:.4f}")
        
        # Show misclassifications
        incorrect = class_df[class_df['is_correct'] == False]
        if len(incorrect) > 0:
            print(f"    Misclassifications:")
            misclass_counts = incorrect['predicted_class'].value_counts()
            for pred_class, count in misclass_counts.items():
                percentage = count / len(incorrect) * 100
                print(f"      - As '{pred_class}': {count} ({percentage:.1f}%)")

def create_visualizations(df):
    """Create comprehensive visualizations"""
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Drowsiness Detection Analysis', fontsize=16, fontweight='bold')
    
    # 1. Accuracy by image type
    accuracy_by_type = df.groupby('image_type')['is_correct'].mean()
    accuracy_by_type.plot(kind='bar', ax=axes[0,0], color=['skyblue', 'lightcoral'])
    axes[0,0].set_title('Accuracy by Image Type')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].set_xlabel('Image Type')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Drowsiness state distribution
    drowsiness_counts = df['drowsiness_state'].value_counts()
    axes[0,1].pie(drowsiness_counts.values, labels=drowsiness_counts.index, autopct='%1.1f%%')
    axes[0,1].set_title('Drowsiness State Distribution')
    
    # 3. Confidence distribution by image type
    for img_type in df['image_type'].unique():
        type_data = df[df['image_type'] == img_type]['confidence']
        axes[0,2].hist(type_data, alpha=0.7, label=f'{img_type.capitalize()}', bins=20)
    axes[0,2].set_title('Confidence Distribution by Image Type')
    axes[0,2].set_xlabel('Confidence')
    axes[0,2].set_ylabel('Frequency')
    axes[0,2].legend()
    
    # 4. Confusion matrix for eye images
    eye_df = df[df['image_type'] == 'eye']
    if len(eye_df) > 0:
        eye_cm = confusion_matrix(eye_df['true_class'], eye_df['predicted_class'])
        eye_classes = sorted(eye_df['true_class'].unique())
        sns.heatmap(eye_cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=eye_classes, yticklabels=eye_classes, ax=axes[1,0])
        axes[1,0].set_title('Eye Detection Confusion Matrix')
        axes[1,0].set_xlabel('Predicted')
        axes[1,0].set_ylabel('Actual')
    
    # 5. Confusion matrix for mouth images
    mouth_df = df[df['image_type'] == 'mouth']
    if len(mouth_df) > 0:
        mouth_cm = confusion_matrix(mouth_df['true_class'], mouth_df['predicted_class'])
        mouth_classes = sorted(mouth_df['true_class'].unique())
        sns.heatmap(mouth_cm, annot=True, fmt='d', cmap='Reds',
                   xticklabels=mouth_classes, yticklabels=mouth_classes, ax=axes[1,1])
        axes[1,1].set_title('Mouth Detection Confusion Matrix')
        axes[1,1].set_xlabel('Predicted')
        axes[1,1].set_ylabel('Actual')
    
    # 6. Overall accuracy comparison
    overall_stats = pd.DataFrame({
        'Metric': ['Eye Accuracy', 'Mouth Accuracy', 'Overall Accuracy'],
        'Value': [
            df[df['image_type'] == 'eye']['is_correct'].mean() if len(eye_df) > 0 else 0,
            df[df['image_type'] == 'mouth']['is_correct'].mean() if len(mouth_df) > 0 else 0,
            df['is_correct'].mean()
        ]
    })
    
    bars = axes[1,2].bar(overall_stats['Metric'], overall_stats['Value'], 
                        color=['skyblue', 'lightcoral', 'lightgreen'])
    axes[1,2].set_title('Overall Performance Comparison')
    axes[1,2].set_ylabel('Accuracy')
    axes[1,2].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, overall_stats['Value']):
        axes[1,2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                      f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved comprehensive analysis to: comprehensive_analysis.png")
    plt.show()

def save_results_to_files(df):
    """Save results to various files"""
    # Save detailed results
    df.to_csv('integrated_drowsiness_results.csv', index=False)
    print("Saved detailed results to: integrated_drowsiness_results.csv")
    
    # Save incorrect predictions
    incorrect_df = df[df['is_correct'] == False]
    if len(incorrect_df) > 0:
        incorrect_df.to_csv('incorrect_predictions_integrated.csv', index=False)
        print(f"Saved {len(incorrect_df)} incorrect predictions to: incorrect_predictions_integrated.csv")
    
    # Save summary statistics
    summary_stats = []
    
    # Overall stats
    summary_stats.append({
        'Category': 'Overall',
        'Metric': 'Total Images',
        'Value': len(df)
    })
    summary_stats.append({
        'Category': 'Overall', 
        'Metric': 'Overall Accuracy',
        'Value': f"{df['is_correct'].mean():.4f}"
    })
    
    # Stats by image type
    for img_type in df['image_type'].unique():
        type_df = df[df['image_type'] == img_type]
        summary_stats.append({
            'Category': f'{img_type.capitalize()}',
            'Metric': 'Count',
            'Value': len(type_df)
        })
        summary_stats.append({
            'Category': f'{img_type.capitalize()}',
            'Metric': 'Accuracy', 
            'Value': f"{type_df['is_correct'].mean():.4f}"
        })
        summary_stats.append({
            'Category': f'{img_type.capitalize()}',
            'Metric': 'Avg Confidence',
            'Value': f"{type_df['confidence'].mean():.4f}"
        })
    
    # Stats by drowsiness state
    drowsiness_counts = df['drowsiness_state'].value_counts()
    for state, count in drowsiness_counts.items():
        summary_stats.append({
            'Category': 'Drowsiness State',
            'Metric': state,
            'Value': f"{count} ({count/len(df)*100:.1f}%)"
        })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv('summary_statistics.csv', index=False)
    print("Saved summary statistics to: summary_statistics.csv")

if __name__ == "__main__":
    main()