import cv2
import numpy as np
import tensorflow as tf
import argparse

# --- Configuration ---
# Model and Image settings
IMG_HEIGHT = 224
IMG_WIDTH = 224
LABELS = ["Not Drowsy", "Drowsy"]

# Drowsiness detection settings
DROWSY_FRAMES_THRESHOLD = 20 # Number of consecutive frames to trigger an alert
alert_color = (0, 0, 255) # Red for the alert text

def main():
    # --- Command-Line Argument Parsing ---
    parser = argparse.ArgumentParser(description="Real-time drowsiness detection from video.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained .h5 model file.")
    parser.add_argument("--source", type=str, default="0", help="Video source. '0' for webcam, or path to a video file.")
    args = parser.parse_args()
    
    # --- Load Resources ---
    try:
        model = tf.keras.models.load_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # --- Initialize Video Capture ---
    video_source = 0 if args.source == "0" else args.source
    try:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise IOError(f"Cannot open video source: {video_source}")
    except Exception as e:
        print(e)
        return

    # --- Main Loop ---
    drowsy_frames_counter = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame.")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        # Assume the largest face is the driver
        if len(faces) > 0:
            (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            
            # Crop and preprocess the face
            face_roi = frame[y:y+h, x:x+w]
            
            if face_roi.size > 0:
                resized_face = cv2.resize(face_roi, (IMG_WIDTH, IMG_HEIGHT))
                normalized_face = resized_face / 255.0
                input_tensor = np.expand_dims(normalized_face, axis=0)
                
                # Make a prediction
                prediction = model.predict(input_tensor)[0][0]
                
                # Interpret the prediction
                if prediction > 0.5: # Drowsy
                    drowsy_frames_counter += 1
                    label = LABELS[1]
                    color = (0, 165, 255) # Orange for warning
                else: # Not Drowsy
                    drowsy_frames_counter = 0 # Reset counter
                    label = LABELS[0]
                    color = (0, 255, 0) # Green for alert

                result_text = f"{label} ({prediction:.2f})"
                cv2.putText(frame, result_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                # --- Trigger Alert if Threshold is Met ---
                if drowsy_frames_counter >= DROWSY_FRAMES_THRESHOLD:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, alert_color, 2)
            
        # Display the resulting frame
        cv2.imshow('Drowsiness Detection', frame)
        
        # Exit loop by pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()