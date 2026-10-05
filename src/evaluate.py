from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from model import WasteClassifier


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "dataset-resized"
MODEL_PATH = PROJECT_ROOT / "models" / "waste_classifier_resnet18_finetuned.pth"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

RESULTS_PATH = OUTPUT_DIR / "evaluation_results.txt"


# --------------------------------------------------
# Settings
# --------------------------------------------------

IMAGE_SIZE = 224
BATCH_SIZE = 32


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
# Recreate the same validation split
# --------------------------------------------------

indices = list(range(len(dataset)))

labels = [
    label
    for _, label in dataset.samples
]

_, val_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=labels
)


val_dataset = Subset(
    dataset,
    val_indices
)

print("Validation images:", len(val_dataset))


# --------------------------------------------------
# DataLoader
# --------------------------------------------------

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = WasteClassifier()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()


# --------------------------------------------------
# Make predictions
# --------------------------------------------------

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in val_loader:

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# --------------------------------------------------
# Calculate metrics
# --------------------------------------------------

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

report = classification_report(
    all_labels,
    all_predictions,
    target_names=dataset.classes
)

matrix = confusion_matrix(
    all_labels,
    all_predictions
)


# --------------------------------------------------
# Display results
# --------------------------------------------------

print(
    f"\nValidation Accuracy: {accuracy:.4f}"
)

print("\nClassification Report:")
print(report)

print("Confusion Matrix:")
print(matrix)


# --------------------------------------------------
# Save results
# --------------------------------------------------

results_text = f"""Waste Image Classification - Evaluation Results

Model: ResNet18 Fine-Tuned
Validation Images: {len(val_dataset)}
Validation Accuracy: {accuracy:.4f}

Classification Report:
{report}

Confusion Matrix:
{matrix}
"""

RESULTS_PATH.write_text(
    results_text,
    encoding="utf-8"
)

print(f"\nEvaluation results saved to:")
print(RESULTS_PATH)