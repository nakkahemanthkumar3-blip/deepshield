import torch
import torch.nn as nn
from torchvision import models
import os

print("Building deepfake detection model...")

# Load MobileNetV2 as base model
model = models.mobilenet_v2(pretrained=True)

# Replace the last layer for deepfake detection (2 classes: REAL or FAKE)
model.classifier[1] = nn.Linear(model.last_channel, 2)

# Save the model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/deepfake_model.pth")

print("Model saved to models/deepfake_model.pth")
print("Done!")