import numpy as np

# Load keypoints and labels
X = np.load("sign_keypoints.npy")  # Features (hand keypoints)
y = np.load("sign_labels.npy")  # Labels (sign classes)

print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features each.")
print(f"Unique classes: {np.unique(y)}")
