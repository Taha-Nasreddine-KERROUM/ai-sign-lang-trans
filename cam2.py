import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Load trained model
model = tf.keras.models.load_model("asl_sign_language_model.h5")


# Load label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load("sign_labels.npy", allow_pickle=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame (mirror effect)
    frame = cv2.flip(frame, 1)

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand landmarks
    result = hands.process(img_rgb)

    keypoints = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract keypoints (x, y, z)
            for lm in hand_landmarks.landmark:
                keypoints.append(lm.x)  # X-coordinate
                keypoints.append(lm.y)  # Y-coordinate
                keypoints.append(lm.z)  # Z-coordinate

    # If we detected a hand, make a prediction
    if len(keypoints) == 63:
        # Reshape for LSTM model input
        keypoints = np.array(keypoints).reshape(1, 1, -1)  # (1, sequence_length, features)

        # Predict sign
        prediction = model.predict(keypoints)
        sign_index = np.argmax(prediction)
        predicted_sign = label_encoder.inverse_transform([sign_index])[0]

        # Display the predicted sign on the frame
        cv2.putText(frame, f"Prediction: {predicted_sign}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Sign Language Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
