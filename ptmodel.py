import cv2
import mediapipe as mp
import torch
import torchvision.transforms as transforms
from PIL import Image

# Load trained model
class FingerCountCNN(torch.nn.Module):
    def __init__(self):
        super(FingerCountCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.relu = torch.nn.ReLU()

        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 128, 128)
            dummy_output = self.pool(self.relu(self.conv1(dummy_input)))
            dummy_output = self.pool(self.relu(self.conv2(dummy_output)))
            self.flatten_size = dummy_output.view(1, -1).size(1)

        self.fc1 = torch.nn.Linear(self.flatten_size, 128)
        self.fc2 = torch.nn.Linear(128, 8)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FingerCountCNN().to(device)
model.load_state_dict(torch.load("finger_count_model.pth", map_location=device))
model.eval()

# Transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Gesture Labels
gesture_labels = {0: "Palm", 1: "L", 2: "Fist", 3: "Fist Moved", 4: "Thumb",
                  5: "Index", 6: "OK", 7: "Palm Moved"}

# Open Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box of the hand
            h, w, c = frame.shape
            x_min, y_min, x_max, y_max = w, h, 0, 0

            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x, x_min), min(y, y_min)
                x_max, y_max = max(x, x_max), max(y, y_max)

            # Extract hand ROI and classify it
            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size > 0:
                pil_img = Image.fromarray(cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB))
                input_img = transform(pil_img).unsqueeze(0).to(device)
                output = model(input_img)
                pred_class = torch.argmax(output, dim=1).item()
                label = gesture_labels.get(pred_class, "Unknown")

                # Draw the label
                cv2.putText(frame, f"Gesture: {label}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw colored circles on each fingertip
            fingertip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255)]

            for i, fingertip_id in enumerate(fingertip_ids):
                x, y = int(hand_landmarks.landmark[fingertip_id].x * w), int(hand_landmarks.landmark[fingertip_id].y * h)
                cv2.circle(frame, (x, y), 10, colors[i], -1)  # Draw filled circles

    cv2.imshow("Hand Gesture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()