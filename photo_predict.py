import cv2
import os
import argparse
import numpy as np
import torch
import dlib
import matplotlib.pyplot as plt
from common import (
    DrowsinessNet, eye_aspect_ratio, get_head_tilt_features,
    LEFT_EYE_POINTS, RIGHT_EYE_POINTS,
    draw_text_with_background
)

# Constants
predictor_path = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(predictor_path):
    raise FileNotFoundError("Dlib shape predictor file not found. Download it from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")

# Dlib initialization
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def predict_drowsiness(image_path, model, detector, predictor, device):
    if detector is None or predictor is None:
        print("Dlib models are not loaded. Cannot perform prediction.")
        return None, None

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None

    original_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    if len(rects) == 0:
        print("No faces detected in the image.")
        draw_text_with_background(
            image=original_image,
            text="No Face Detected",
            org=(50, 50),
            font_scale=1.2,
            text_color=(0, 0, 255),
            bg_color=(255, 255, 255),
            thickness=2
        )
        return original_image, "No Face"

    rect = rects[0]
    shape = predictor(gray, rect)
    landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

    left_eye_lm = landmarks[LEFT_EYE_POINTS]
    right_eye_lm = landmarks[RIGHT_EYE_POINTS]
    left_ear = eye_aspect_ratio(left_eye_lm)
    right_ear = eye_aspect_ratio(right_eye_lm)
    avg_ear = (left_ear + right_ear) / 2.0

    roll_angle, nod_feature = get_head_tilt_features(shape, (image.shape[1], image.shape[0]))

    features_np = np.array([avg_ear, roll_angle, nod_feature], dtype=np.float32)
    features_tensor = torch.tensor(features_np).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(features_tensor)
        probability = torch.sigmoid(output).item()

    label = "Drowsy" if probability < 0.5 else "Alert"
    color = (0, 0, 255) if label == "Drowsy" else (0, 255, 0)

    # Draw face rectangle based on label
    x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
    cv2.rectangle(original_image, (x1, y1), (x2, y2), color, 20)

    # Draw facial landmarks
    for (x, y) in landmarks:
        cv2.circle(original_image, (x, y), 2, (255, 0, 255), 15)

    # Draw label with probability
    text_pred = f"{label} ({probability:.2f})"
    draw_text_with_background(
        image=original_image,
        text=text_pred,
        org=(x1, y1 - 10),
        font_scale=12.0,
        text_color=(0, 0, 0),
        bg_color=(255, 255, 255),
        thickness=12
    )

    return original_image, label


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict drowsiness from a single image using a trained PyTorch model.")
    parser.add_argument("image_path", type=str, help="Path to the input image file.")
    parser.add_argument("--model_path", type=str, default="best_drowsiness_model.pth", help="Path to the trained PyTorch model.")
    parser.add_argument("--output_path", type=str, default="prediction_output.jpg", help="Path to save the output image.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = DrowsinessNet(input_features_count=3)
    model.to(device)

    if not os.path.exists(args.model_path):
        print(f"Model not found at {args.model_path}")
        exit()

    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        print(f"Model loaded from {args.model_path}")
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit()

    output_image, prediction = predict_drowsiness(args.image_path, model, detector, predictor, device)

    if output_image is not None:
        cv2.imwrite(args.output_path, output_image)
        print(f"Saved prediction image to {args.output_path}")
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"Prediction: {prediction}")
        plt.show()
    else:
        print("No output generated.")
