import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


# Load the trained model
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
        self.fc2 = torch.nn.Linear(128, 8)  # 8 classes

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
model.load_state_dict(torch.load("finger_count_model.pth"))
model.eval()

# Class mapping
class_mapping = {
    0: "01_palm",
    1: "02_l",
    2: "03_fist",
    3: "04_fist_moved",
    4: "05_thumb",
    5: "06_index",
    6: "07_ok",
    7: "08_palm_moved"
}

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert OpenCV frame to PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Preprocess the image
    image = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(image)
        predicted_class = torch.argmax(output, dim=1).item()

    predicted_label = class_mapping[predicted_class]

    # Display prediction
    cv2.putText(frame, f"Prediction: {predicted_label}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Hand Gesture Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
