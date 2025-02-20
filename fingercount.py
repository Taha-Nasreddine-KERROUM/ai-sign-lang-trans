import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Finger tip landmarks
            fingertips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
            finger_count = 0

            for tip in fingertips:
                # Get fingertip and lower joint (knuckle) positions
                tip_y = hand_landmarks.landmark[tip].y
                base_y = hand_landmarks.landmark[tip - 2].y  # Lower joint

                # If fingertip is above the base joint, count it as raised
                if tip_y < base_y:
                    finger_count += 1

            # Display count
            cv2.putText(frame, f"Fingers: {finger_count}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Finger Counting", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
