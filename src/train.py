from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from model import WasteClassifier


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "dataset-resized"
MODEL_DIR = PROJECT_ROOT / "models"


# --------------------------------------------------
# Training settings
# --------------------------------------------------

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE = 0.0001

MODEL_NAME = "waste_classifier_resnet18_finetuned.pth"


# --------------------------------------------------
# Image transformations
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# Dataset
# --------------------------------------------------

dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform
)

print("Classes:", dataset.classes)
print("Total images:", len(dataset))


# --------------------------------------------------
# Train / validation split
# --------------------------------------------------

indices = list(range(len(dataset)))

labels = [
    label
    for _, label in dataset.samples
]

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=labels
)


train_dataset = Subset(
    dataset,
    train_indices
)

val_dataset = Subset(
    dataset,
    val_indices
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# --------------------------------------------------
# DataLoaders
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = WasteClassifier()


# --------------------------------------------------
# Loss and optimizer
# --------------------------------------------------

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# Best model tracking
# --------------------------------------------------

best_val_accuracy = 0.0

MODEL_DIR.mkdir(
    exist_ok=True
)

model_path = MODEL_DIR / MODEL_NAME


# --------------------------------------------------
# Training
# --------------------------------------------------

for epoch in range(EPOCHS):

    # ------------------------------
    # Training
    # ------------------------------

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        outputs = model(images)

        loss = loss_fn(
            outputs,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        running_loss += loss.item()


    train_loss = (
        running_loss / len(train_loader)
    )


    # ------------------------------
    # Validation
    # ------------------------------

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)


    val_accuracy = correct / total


    # ------------------------------
    # Print results
    # ------------------------------

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f}"
    )


    # ------------------------------
    # Save best model
    # ------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            model_path
        )

        print(
            f"  ✓ Best model saved "
            f"(Val Accuracy: {val_accuracy:.4f})"
        )


# --------------------------------------------------
# Final result
# --------------------------------------------------

print("\nTraining complete.")

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.4f}"
)

print(
    f"Best model saved to: "
    f"{model_path}"
)