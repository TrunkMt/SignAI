
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

# SETTINGS
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "sign_model.pth"
LABELS_PATH = BASE_DIR / "labels.json"

IMAGE_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 5

torch.set_num_threads(4)
DEVICE = torch.device("cpu")

# LOAD DATASET
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)
labels = dataset.classes

print("Classes:", labels)
print("Total images:", len(dataset))

# WARN ABOUT SMALL CLASSES
counts = {}
for _, class_index in dataset.samples:
    counts[class_index] = counts.get(class_index, 0) + 1

for i, label in enumerate(labels):
    print(f"{label}: {counts.get(i, 0)} images")

# BALANCED SAMPLING
sample_weights = [
    1.0 / counts[class_index]
    for _, class_index in dataset.samples
]

sampler = WeightedRandomSampler(
    sample_weights,
    num_samples=min(len(dataset), 15000),
    replacement=True
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0
)

# LIGHTWEIGHT CNN
class SignAIModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)

model = SignAIModel(len(labels)).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# SAVE LABELS
with open(LABELS_PATH, "w") as f:
    json.dump(labels, f)

# TRAIN
print("\nStarting fast SignAI training...")
start = time.time()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == targets).sum().item()
        total += targets.size(0)

    accuracy = 100 * correct / total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {total_loss / len(loader):.4f} | "
        f"Training accuracy: {accuracy:.2f}% | "
        f"Elapsed: {(time.time() - start) / 60:.1f} min"
    )

# SAVE MODEL
torch.save(model.state_dict(), MODEL_PATH)

print("\nTraining completed!")
print("Model:", MODEL_PATH)
print("Labels:", LABELS_PATH)
print(f"Total time: {(time.time() - start) / 60:.2f} minutes")
