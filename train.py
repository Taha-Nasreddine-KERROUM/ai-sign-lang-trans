import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split

# ✅ Correct dataset path
dataset_path = r"C:\Users\PC\Desktop\data\hand_gestures"

# ✅ Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ✅ Load dataset
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# ✅ Print class-to-index mapping
print("Class-to-Index Mapping:", dataset.class_to_idx)

# ✅ Dynamically determine number of classes
num_classes = len(dataset.classes)
print(f"Detected {num_classes} classes.")

# ✅ Ensure at least 2 classes exist
assert num_classes > 1, "Error: Not enough classes in dataset!"

# ✅ Split dataset into training (80%) and validation (20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# ✅ Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ✅ Define CNN model
class FingerCountCNN(nn.Module):
    def __init__(self, num_classes):
        super(FingerCountCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # ✅ Compute dynamically the size of the flattened layer
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 128, 128)
            dummy_output = self.pool(self.relu(self.conv1(dummy_input)))
            dummy_output = self.pool(self.relu(self.conv2(dummy_output)))
            self.flatten_size = dummy_output.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, num_classes)  # ✅ Adjust to actual class count

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ✅ Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Initialize model with correct class count
model = FingerCountCNN(num_classes).to(device)

# ✅ Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ✅ Train model
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # ✅ Debug: Check label range
        assert labels.min() >= 0, "Error: Negative labels found!"
        assert labels.max() < num_classes, f"Error: Label {labels.max()} is out of range!"

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

# ✅ Save trained model
torch.save(model.state_dict(), 'finger_count_model.pth')
print("Model saved successfully!")

# ✅ Validate model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total * 100
print(f"Validation Accuracy: {accuracy:.2f}%")
