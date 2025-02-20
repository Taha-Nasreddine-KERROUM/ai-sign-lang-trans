import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler

# 🔹 Define the number of classes based on detected labels
num_classes = 8

# ✅ Smarter CNN Model
class FingerCountCNN(nn.Module):
    def __init__(self):
        super(FingerCountCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Added batch normalization
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Added extra conv layer
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3)

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 128, 128)
            dummy_output = self.pool(F.relu(self.bn1(self.conv1(dummy_input))))
            dummy_output = self.pool(F.relu(self.bn2(self.conv2(dummy_output))))
            dummy_output = self.pool(F.relu(self.bn3(self.conv3(dummy_output))))
            self.flatten_size = dummy_output.view(1, -1).size(1)

        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 🔹 Dataset Path
dataset_path = r"C:\Users\PC\Desktop\data\processed_combine_asl_dataset"

# 🔹 Smarter Data Augmentation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),  # Flip images
    transforms.RandomRotation(15),  # Rotate slightly
    transforms.ColorJitter(brightness=0.3, contrast=0.3),  # Adjust colors
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize
])

# 🔹 Load Dataset
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

# ✅ Debugging: Check class mapping
print(f"Class-to-Index Mapping: {dataset.class_to_idx}")
num_classes = len(dataset.class_to_idx)  # Auto-detect number of classes

# 🔹 Split into Train (80%) and Validation (20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

# 🔹 Model, Loss, Optimizer, Scheduler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FingerCountCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # AdamW for better performance
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# 🔥 Mixed Precision Training
scaler = GradScaler()

# 🔹 Training Loop
epochs = 15
best_val_loss = float("inf")

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with autocast():  # Mixed precision training
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total * 100
    avg_loss = total_loss / len(train_loader)

    # 🔥 Validation Loop
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total * 100
    val_loss /= len(val_loader)

    # 🔹 Reduce learning rate if no improvement
    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

    # 🔹 Save Best Model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'finger_count_best.pth')
        print("✅ Best model saved!")

# 🔥 Final Validation Accuracy
print(f"✅ Final Validation Accuracy: {val_acc:.2f}%")

# ✅ Save Model
torch.save(model.state_dict(), 'finger_count_final.pth')
print("✅ Final model saved!")
