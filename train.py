import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import os

# ── Configuration ──────────────────────────────────────────────────
DATA_DIR   = "real_and_fake_face"
MODEL_PATH = "models/deepfake_model.pth"
EPOCHS     = 5
BATCH_SIZE = 16
LR         = 0.001

# ── Data transforms ────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ── Load dataset ───────────────────────────────────────────────────
print("Loading dataset...")
dataset    = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=0)

print(f"Classes: {dataset.classes}")
print(f"Total images: {len(dataset)}")

# ── Build model ────────────────────────────────────────────────────
print("Building model...")
model = models.efficientnet_b0(weights='IMAGENET1K_V1')
model.classifier[1] = nn.Linear(
    model.classifier[1].in_features, 2
)

# ── Training ───────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss   = 0
    correct      = 0
    total        = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} "
                  f"Batch {batch_idx} "
                  f"Loss: {loss.item():.4f} "
                  f"Acc: {100.*correct/total:.1f}%")

    print(f"Epoch {epoch+1} done! "
          f"Accuracy: {100.*correct/total:.1f}%")

# ── Save trained model ─────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"Trained model saved to {MODEL_PATH}")
print("Training complete!")