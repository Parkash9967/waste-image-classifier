import torch.nn as nn
from torchvision import models


class WasteClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        # Freeze all pretrained layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the final ResNet block
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace the final classifier
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            6
        )

    def forward(self, x):
        return self.model(x)