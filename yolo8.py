from ultralytics import YOLO
import cv2

# Load pre-trained YOLO model (or your custom-trained model)
model = YOLO("yolov8n.pt")  # Replace with your trained model (e.g., "best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame)

    # Show results
    results.show()

cap.release()
cv2.destroyAllWindows()
