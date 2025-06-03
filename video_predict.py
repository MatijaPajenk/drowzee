import cv2
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import distance as dist
import dlib
from common import (
    DrowsinessNet, eye_aspect_ratio, get_head_tilt_features,
    LEFT_EYE_POINTS, RIGHT_EYE_POINTS,
    LEFT_EYE_LEFT_CORNER, RIGHT_EYE_RIGHT_CORNER, NOSE_TIP, CHIN,
    draw_text_with_background
)

# Initialize dlib
predictor_path = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(predictor_path):
    raise FileNotFoundError("Dlib shape predictor file not found. Please download from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def process_frame(frame, model, device):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    if len(rects) == 0:
        draw_text_with_background(
            image=frame,
            text="No Face Detected",
            org=(50, 50),
            font_scale=1.2,
            text_color=(0, 0, 255),
            bg_color=(255, 255, 255),
            thickness=2
        )
        return frame

    rect = rects[0]
    shape = predictor(gray, rect)
    landmarks = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])

    left_eye_lm = landmarks[LEFT_EYE_POINTS]
    right_eye_lm = landmarks[RIGHT_EYE_POINTS]
    left_ear = eye_aspect_ratio(left_eye_lm)
    right_ear = eye_aspect_ratio(right_eye_lm)
    avg_ear = (left_ear + right_ear) / 2.0

    roll_angle, nod_feature = get_head_tilt_features(shape, (frame.shape[1], frame.shape[0]))
    features_np = np.array([avg_ear, roll_angle, nod_feature], dtype=np.float32)
    features_tensor = torch.tensor(features_np).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(features_tensor)
        probability = torch.sigmoid(output).item()

    label = "Drowsy" if probability < 0.5 else "Alert"
    color = (0, 0, 255) if label == "Drowsy" else (0, 255, 0)

    for (x, y) in landmarks:
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    cv2.line(frame, tuple(landmarks[LEFT_EYE_LEFT_CORNER]), tuple(landmarks[RIGHT_EYE_RIGHT_CORNER]), (255, 0, 0), 2)
    cv2.line(frame, tuple(landmarks[NOSE_TIP]), tuple(landmarks[CHIN]), (0, 0, 255), 2)

    text_pred = f"{label} ({probability:.2f})"
    draw_text_with_background(
        image=frame,
        text=text_pred,
        org=(30, 50),
        font_scale=1.2,
        text_color=(0, 0, 0),
        bg_color=(255, 255, 255),
        thickness=2
    )

    return frame

def main():
    parser = argparse.ArgumentParser(description="Real-time Drowsiness Detection on Video or Webcam")
    parser.add_argument("--video", type=str, help="Path to video file (leave blank to use webcam)")
    parser.add_argument("--model_path", type=str, default="best_drowsiness_model.pth", help="Path to trained PyTorch model")
    parser.add_argument("--output", type=str, default="output_drowsiness_video.avi", help="Path to save output video")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = DrowsinessNet(input_features_count=3)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.to(device)
    print("[INFO] Model loaded successfully.")

    cap = cv2.VideoCapture(0 if args.video is None else args.video)

    if not cap.isOpened():
        print("[ERROR] Could not open video source.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    print(f"[INFO] Output video will be saved to {args.output}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, model, device)
        out.write(frame)

        cv2.imshow("Drowsiness Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("[INFO] Video saved and resources released.")

if __name__ == "__main__":
    main()
