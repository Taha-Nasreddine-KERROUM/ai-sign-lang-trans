import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

DATASET_PATH = r"C:\Users\PC\Desktop\data\processed_combine_asl_dataset"  # Folder where images are stored
OUTPUT_FILE = "sign_keypoints.npy"  # File to save extracted keypoints

data = []
labels = []

# Loop through each class folder (A, B, C, ..., Z)
for sign_class in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, sign_class)

    # Check if it's a directory
    if not os.path.isdir(class_path):
        continue

    print(f"Processing class: {sign_class}")

    # Loop through images in the folder
    for img_name in os.listdir(class_path):
        img_path = os.path.join(class_path, img_name)

        # Ensure it's an image
        if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            print(f"Skipping non-image file: {img_name}")
            continue

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process image with MediaPipe
        result = hands.process(img_rgb)

        # Extract keypoints if a hand is detected
        keypoints = []
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    keypoints.append(lm.x)
                    keypoints.append(lm.y)
                    keypoints.append(lm.z)
        else:
            # No hand detected, append empty keypoints (optional)
            keypoints = [0] * (21 * 3)

        data.append(keypoints)
        labels.append(sign_class)  # Save the class label

# Convert to NumPy arrays and Save
data = np.array(data)
labels = np.array(labels)

np.save("sign_keypoints.npy", data)
np.save("sign_labels.npy", labels)

print(f"✅ Successfully saved keypoints to sign_keypoints.npy")
print(f"✅ Successfully saved labels to sign_labels.npy")