from pathlib import Path
import argparse

import torch
from PIL import Image
from torchvision import transforms

from model import WasteClassifier


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "waste_classifier_resnet18_finetuned.pth"
)


# --------------------------------------------------
# Settings
# --------------------------------------------------

IMAGE_SIZE = 224

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]


# --------------------------------------------------
# Image transformation
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
# Prediction function
# --------------------------------------------------

def predict_image(image_path):

    image = Image.open(
        image_path
    ).convert("RGB")

    image_tensor = transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

    predicted_class = torch.argmax(
        probabilities
    ).item()

    predicted_name = CLASS_NAMES[
        predicted_class
    ]

    return predicted_name, probabilities


# --------------------------------------------------
# Command-line arguments
# --------------------------------------------------

parser = argparse.ArgumentParser(
    description="Predict the waste category of an image."
)

parser.add_argument(
    "--image",
    type=str,
    required=True,
    help="Path to the image"
)

args = parser.parse_args()


# --------------------------------------------------
# Validate image path
# --------------------------------------------------

image_path = Path(
    args.image
)

if not image_path.exists():

    print(
        f"Error: Image not found: {image_path}"
    )

    raise SystemExit(1)


# --------------------------------------------------
# Predict
# --------------------------------------------------

predicted_class, probabilities = predict_image(
    image_path
)


print("\nPrediction:")
print(
    f"  {predicted_class}"
)


print("\nClass probabilities:")

sorted_predictions = sorted(
    zip(CLASS_NAMES, probabilities.tolist()),
    key=lambda x: x[1],
    reverse=True
)

for class_name, probability in sorted_predictions:

    print(
        f"  {class_name:<10} "
        f"{probability * 100:6.2f}%"
    )