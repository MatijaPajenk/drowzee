import cv2
import numpy as np
import tensorflow as tf
import argparse
import os

# --- Model and Image Configuration ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
LABELS = ["Not Drowsy", "Drowsy"]

def preprocess_image(image, face_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    if len(faces) == 0:
        return None, None

    (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    face_roi = image[y:y+h, x:x+w]
    if face_roi.size == 0:
        return None, None

    # Resize to model's expected input shape: 128x128
    resized_face = cv2.resize(face_roi, (128, 128))
    normalized_face = resized_face / 255.0

    # Repeat this frame 5 times
    sequence = np.stack([normalized_face] * 5, axis=0)  # shape (5, 128, 128, 3)

    # Add batch dimension: (1, 5, 128, 128, 3)
    input_tensor = np.expand_dims(sequence, axis=0)

    return input_tensor, (x, y, w, h)


def main():
    parser = argparse.ArgumentParser(description="Drowsiness detection from a single photo.")
    parser.add_argument("--model", type=str, default="drowsiness_detection_model.keras",
                        help="Path to the trained .keras model file.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image.")
    args = parser.parse_args()

    # Load model
    try:
        model = tf.keras.models.load_model(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Use absolute path to Haar cascade to avoid dependency on cv2.data
    haar_path = os.path.join("haarcascade_frontalface_default.xml")
    if not os.path.exists(haar_path):
        print("Error: Haar cascade file not found.")
        return

    face_cascade = cv2.CascadeClassifier(haar_path)

    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not read image at {args.image}")
        return

    # Add padding above image to avoid text clipping
    padding = 60  # pixels of blank space
    output_image = cv2.copyMakeBorder(
        image,
        top=padding, bottom=0, left=0, right=0,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)  # black padding
    )

    


    processed_face, face_coords = preprocess_image(image, face_cascade)

    if processed_face is None:
        print("No face detected in the image.")
        cv2.imshow("Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    prediction = model.predict(processed_face)[0][0]
    print(f"Raw model output: {prediction:.4f}")

    # Interpretation
    if prediction < 0.5:
        label = LABELS[1]  # Drowsy
        color = (0, 0, 255)  # Red
    else:
        label = LABELS[0]  # Not Drowsy
        color = (0, 255, 0)  # Green

    result_text = f"{label} ({prediction:.2f})"

    # Adjust y coordinate because of top padding
    (x, y, w, h) = face_coords
    y += padding

    # Draw rectangle and label
    cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(output_image, result_text, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, color, 2, lineType=cv2.LINE_AA)
    
    # Resize for display if too large
    max_height = 400
    if output_image.shape[0] > max_height:
        scale = max_height / output_image.shape[0]
        output_image = cv2.resize(output_image, (int(output_image.shape[1] * scale), max_height))

    

    cv2.imshow("Drowsiness Detection Result", output_image)
    print("Prediction complete. Press any key to exit.")
    cv2.waitKey(2000)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
