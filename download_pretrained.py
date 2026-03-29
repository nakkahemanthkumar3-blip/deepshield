import torch
import torch.nn as nn
from torchvision import models
import os

print("Building EfficientNet deepfake detection model...")

model = models.efficientnet_b0(weights='IMAGENET1K_V1')

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features, 2
)

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/deepfake_model.pth")

print("Model saved to models/deepfake_model.pth")
print("Done!")