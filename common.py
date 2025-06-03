import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import distance as dist


# Define landmark indices for eyes and other relevant points
JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))

# Points for head pose estimation (simplified)
NOSE_TIP = 30
CHIN = 8
LEFT_EYE_LEFT_CORNER = 36
RIGHT_EYE_RIGHT_CORNER = 45
LEFT_MOUTH_CORNER = 48
RIGHT_MOUTH_CORNER = 54

# --- Feature Extraction Functions (Copied from training script) ---
def eye_aspect_ratio(eye_landmarks_np):
    # eye_landmarks_np should be a NumPy array of 6 (x, y)-coordinates
    # Vertical eye landmarks
    A = dist.euclidean(eye_landmarks_np[1], eye_landmarks_np[5])
    B = dist.euclidean(eye_landmarks_np[2], eye_landmarks_np[4])
    # Horizontal eye landmark
    C = dist.euclidean(eye_landmarks_np[0], eye_landmarks_np[3])

    if C == 0: return 0.3 # Avoid division by zero, return a neutral EAR
    ear = (A + B) / (2.0 * C)
    return ear

def get_head_tilt_features(shape, image_size_wh):
    # shape: dlib full_object_detection
    # image_size_wh: (width, height) of the image
    
    # For Roll angle: using the line connecting the outer eye corners
    left_eye_outer_corner = (shape.part(LEFT_EYE_POINTS[0]).x, shape.part(LEFT_EYE_POINTS[0]).y)
    right_eye_outer_corner = (shape.part(RIGHT_EYE_POINTS[3]).x, shape.part(RIGHT_EYE_POINTS[3]).y)

    delta_y_eyes = right_eye_outer_corner[1] - left_eye_outer_corner[1]
    delta_x_eyes = right_eye_outer_corner[0] - left_eye_outer_corner[0]

    roll_angle_deg = 0
    if delta_x_eyes != 0:
        roll_angle_rad = np.arctan(delta_y_eyes / delta_x_eyes)
        roll_angle_deg = np.degrees(roll_angle_rad)
    elif delta_y_eyes != 0: # Eyes are perfectly vertical relative to each other
        roll_angle_deg = 90 if delta_y_eyes > 0 else -90

    # For a simplified Pitch/Nod feature:
    # Ratio of vertical distance (nose tip to chin) to face height (eyebrow to chin)
    # This is a heuristic. More advanced methods use solvePnP.
    try:
        nose_tip_y = shape.part(NOSE_TIP).y
        chin_y = shape.part(CHIN).y
        
        # Estimate face top (e.g., midpoint of eyebrows)
        eyebrow_left_y = min(shape.part(p).y for p in LEFT_EYEBROW_POINTS)
        eyebrow_right_y = min(shape.part(p).y for p in RIGHT_EYEBROW_POINTS)
        face_top_y = (eyebrow_left_y + eyebrow_right_y) / 2.0
        
        face_height = chin_y - face_top_y
        nose_chin_dist = chin_y - nose_tip_y

        nod_feature = 0.5 # Default neutral value
        if face_height > 10: # Avoid division by zero or instability with tiny faces
            nod_feature = nose_chin_dist / face_height
            nod_feature = np.clip(nod_feature, 0, 1) # Keep it in a reasonable range
            
    except Exception: # If any landmark is not found (shouldn't happen with dlib's 68 points if face is found)
        nod_feature = 0.5 # Default if calculation fails

    return roll_angle_deg, nod_feature

# --- DrowsinessNet Model Definition (Copied from training script) ---
class DrowsinessNet(nn.Module):
    def __init__(self, input_features_count):
        super(DrowsinessNet, self).__init__()
        self.fc1 = nn.Linear(input_features_count, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.4)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.3)
        
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        return x
    
def draw_text_with_background(image, text, org, font=cv2.FONT_HERSHEY_SIMPLEX, 
                              font_scale=1.0, text_color=(0, 0, 0), 
                              bg_color=(255, 255, 255), thickness=2, padding=25):
    """
    Draw text with a background rectangle on an image.
    
    :param image: Image to draw on
    :param text: Text string
    :param org: Bottom-left corner of the text string (x, y)
    :param font: Font type
    :param font_scale: Font scale (float)
    :param text_color: Text color (B, G, R)
    :param bg_color: Background color (B, G, R)
    :param thickness: Thickness of text
    :param padding: Padding around the text
    """
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    top_left = (x - padding, y - text_h - padding)
    bottom_right = (x + text_w + padding, y + baseline + padding)

    # Draw rectangle
    cv2.rectangle(image, top_left, bottom_right, bg_color, -1)
    # Put text
    cv2.putText(image, text, org, font, font_scale, text_color, thickness, cv2.LINE_AA)