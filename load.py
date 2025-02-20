import cv2
import torch
import mediapipe as mp
import numpy as np
from torchvision import transforms

# Load MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load Pre-trained PyTorch Model (Assuming a model is available)
class SignLanguageModel(torch.nn.Module):
    def __init__(self):
        super(SignLanguageModel, self).__init__()
        self.fc = torch.nn.Linear(42, 26)  # Example: 21 keypoints * 2 coords
    
    def forward(self, x):
        return self.fc(x)

model = SignLanguageModel()
model.load_state_dict(torch.load("sign_language_model.pth"))
model.eval()
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Transform function
def preprocess_keypoints(landmarks):
    keypoints = np.array([[lm.x, lm.y] for lm in landmarks.landmark]).flatten()
    return torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

# OpenCV Webcam Capture
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Preprocess landmarks
            input_tensor = preprocess_keypoints(hand_landmarks)
            
            # Predict sign language letter
            with torch.no_grad():
                output = model(input_tensor)
                pred_label = chr(torch.argmax(output).item() + 65)  # Assuming A-Z classification
                
            cv2.putText(frame, pred_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Sign Language Translator", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()