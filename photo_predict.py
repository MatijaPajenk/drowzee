import dlib
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import os
import argparse
from common import (
    DrowsinessNet, eye_aspect_ratio, get_head_tilt_features,
    LEFT_EYE_POINTS, RIGHT_EYE_POINTS,
    LEFT_EYE_LEFT_CORNER, RIGHT_EYE_RIGHT_CORNER, NOSE_TIP, CHIN,
    draw_text_with_background
)

# --- Dlib Model Paths and Landmark Definitions (Copied from training script) ---
predictor_path = "shape_predictor_68_face_landmarks.dat"

# Ensure dlib models are loaded
detector = None
predictor = None
if not os.path.exists(predictor_path):
    print(f"Shape predictor model not found at {predictor_path}")
    print("Please download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    print("Unzip it, and place it in the current directory or update 'predictor_path'.")
else:
    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(predictor_path)
        print("Dlib face detector and shape predictor loaded successfully.")
    except Exception as e:
        print(f"Error loading Dlib models: {e}")
        print("Ensure the predictor_path is correct and the file is not corrupted.")
        detector = None
        predictor = None # Set to None to prevent further execution if dlib fails



# --- Main Prediction Logic ---
def predict_drowsiness(image_path, model, detector, predictor, device):
    if detector is None or predictor is None:
        print("Dlib models are not loaded. Cannot perform prediction.")
        return None, None

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None
    
    original_image = image.copy() # Keep a copy for drawing results
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1) # Detect faces in the grayscale image

    if len(rects) == 0:
        print("No faces detected in the image.")
        # Draw "No Face Detected" on the image
        # Increased font scale and thickness
        cv2.putText(original_image, "No Face Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        return original_image, "No Face"

    # Assume only one face for simplicity, or pick the largest one
    rect = rects[0] 
    
    # Draw bounding box around the detected face
    # Increased thickness
    x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
    cv2.rectangle(original_image, (x1, y1), (x2, y2), (255, 0, 0), 3)

    shape = predictor(gray, rect)
    landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

    # Extract features
    left_eye_lm = landmarks[LEFT_EYE_POINTS]
    right_eye_lm = landmarks[RIGHT_EYE_POINTS]
    left_ear = eye_aspect_ratio(left_eye_lm)
    right_ear = eye_aspect_ratio(right_eye_lm)
    avg_ear = (left_ear + right_ear) / 2.0

    roll_angle, nod_feature = get_head_tilt_features(shape, (image.shape[1], image.shape[0]))

    # Prepare features for the model
    features_np = np.array([avg_ear, roll_angle, nod_feature], dtype=np.float32)
    features_tensor = torch.tensor(features_np).unsqueeze(0).to(device) # Add batch dimension and move to device

    # Make prediction
    model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculations
        outputs_logits = model(features_tensor)
        probabilities = torch.sigmoid(outputs_logits).item() # Get probability (0-1)

    # Determine prediction based on a threshold (e.g., 0.5)
    prediction_label = "Drowsy" if probabilities < 0.5 else "Alert"
    color = (0, 0, 255) if prediction_label == "Drowsy" else (0, 255, 0) # Red for drowsy, Green for alert

    # Draw landmarks for visualization
    # Increased thickness for landmark circles
    for (x, y) in landmarks:
        cv2.circle(original_image, (x, y), 2, (0, 255, 0), -1) # Increased radius to 2

    # Draw features used for tilt visualization (optional)
    # Increased thickness for lines
    cv2.line(original_image, tuple(landmarks[LEFT_EYE_LEFT_CORNER]), tuple(landmarks[RIGHT_EYE_RIGHT_CORNER]), (255, 0, 0), 3) # Increased thickness to 3
    cv2.line(original_image, tuple(landmarks[NOSE_TIP]), tuple(landmarks[CHIN]), (0, 0, 255), 3) # Increased thickness to 3


    # Display prediction and feature values on the image
    text_ear = f"EAR: {avg_ear:.2f}"
    text_roll = f"Roll: {roll_angle:.1f}"
    text_nod = f"Nod: {nod_feature:.2f}"
    text_pred = f"Status: {prediction_label} ({probabilities:.2f})"

    # Adjust text position based on face detection
    y_offset = y2 + 30 if len(rects) > 0 else 50
    
    font_size = 13
    font_weight = 16

    # Increased font scale and thickness for all text
    # cv2.putText(original_image, text_ear, (x1, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), font_weight)
    # cv2.putText(original_image, text_roll, (x1, y_offset + 200), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), font_weight)
    # cv2.putText(original_image, text_nod, (x1, y_offset + 400), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), font_weight)
    # Significantly increased font scale for the main prediction text
    draw_text_with_background(
        image=original_image,
        text=text_pred,
        org=(x1 - 700, y_offset + 500),
        font_scale=font_size,
        text_color=(0, 0, 0),
        bg_color=(255, 255, 255),
        thickness=font_weight
    )

    return original_image, prediction_label

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict drowsiness from a single image using a trained PyTorch model.")
    parser.add_argument("image_path", type=str, help="Path to the input image file.")
    parser.add_argument("--model_path", type=str, default="best_drowsiness_model.pth", 
                        help="Path to the trained PyTorch model (.pth file).")
    parser.add_argument("--output_path", type=str, default="prediction_output.jpg", 
                        help="Path to save the output image with predictions.")
    args = parser.parse_args()

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    # The input_features_count must match what the model was trained with (EAR, Roll, Nod = 3)
    input_features_count = 3 
    model = DrowsinessNet(input_features_count=input_features_count)
    model.to(device)

    # Load trained weights
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print("Please ensure 'best_drowsiness_model.pth' is in the current directory or specify its path.")
        exit()

    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        print(f"Model loaded successfully from {args.model_path}")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("Ensure the model architecture (DrowsinessNet) matches the saved model, and the file is not corrupted.")
        exit()

    # Perform prediction
    output_image, prediction = predict_drowsiness(args.image_path, model, detector, predictor, device)

    if output_image is not None:
        cv2.imwrite(args.output_path, output_image)
        print(f"Prediction saved to {args.output_path}")

        output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(8, 6))
        plt.imshow(output_image_rgb)
        plt.axis('off')  # Hide axes
        plt.show()
    else:
        print("Prediction could not be performed.")