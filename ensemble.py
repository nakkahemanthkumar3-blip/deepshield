import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import os

# ── Configuration ──────────────────────────────────────────────────
IMG_SIZE       = (224, 224)
FAKE_THRESHOLD = 0.5

# ── Image preprocessing ────────────────────────────────────────────
def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)

# ── Model 1: EfficientNet ──────────────────────────────────────────
_efficientnet = None

def get_efficientnet():
    global _efficientnet
    if _efficientnet is None:
        print("[Ensemble] Loading EfficientNet...")
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, 2
        )
        model.load_state_dict(torch.load(
            "models/deepfake_model.pth",
            map_location="cpu", weights_only=True))
        model.eval()
        _efficientnet = model
        print("[Ensemble] EfficientNet ready!")
    return _efficientnet

def predict_efficientnet(image_path):
    model  = get_efficientnet()
    tensor = preprocess(image_path)
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)
        # Index 0 = FAKE probability
        return float(probs[0][0])

# ── Model 2: ResNet50 ──────────────────────────────────────────────
_resnet = None

def get_resnet():
    global _resnet
    if _resnet is None:
        print("[Ensemble] Loading ResNet50...")
        model = models.resnet50(weights='IMAGENET1K_V1')
        # Replace last layer for 2 classes
        model.fc = nn.Linear(model.fc.in_features, 2)
        model.eval()
        _resnet = model
        print("[Ensemble] ResNet50 ready!")
    return _resnet

def predict_resnet(image_path):
    model  = get_resnet()
    tensor = preprocess(image_path)
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)
        # Index 0 = FAKE probability
        return float(probs[0][0])

# ── Model 3: SVM ───────────────────────────────────────────────────
def extract_features(image_path):
    # Convert image to flat feature vector for SVM
    img = Image.open(image_path).convert("RGB")
    img = img.resize((32, 32))  # Small size for SVM
    arr = np.array(img).flatten() / 255.0  # Normalize
    return arr

def predict_svm(image_path):
    # Simple rule-based SVM approximation
    # Checks color distribution patterns
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0

    # Check color variance - GANs often have unusual color patterns
    r_var = np.var(arr[:,:,0])
    g_var = np.var(arr[:,:,1])
    b_var = np.var(arr[:,:,2])

    avg_var = (r_var + g_var + b_var) / 3

    # Very low or very high variance suggests GAN artifact
    if avg_var < 0.02 or avg_var > 0.15:
        return 0.65  # Likely fake
    return 0.25  # Likely real

# ── Ensemble Detection ─────────────────────────────────────────────
def ensemble_detect(image_path):
    # Get predictions from all 3 models
    score1 = predict_efficientnet(image_path)  # EfficientNet
    score2 = predict_resnet(image_path)        # ResNet50
    score3 = predict_svm(image_path)           # SVM

    # Weighted average (EfficientNet is most trusted)
    final_score = (score1 * 0.60) + (score2 * 0.25) + (score3 * 0.15)

    result     = "FAKE" if final_score >= FAKE_THRESHOLD else "REAL"
    confidence = final_score if result == "FAKE" else 1.0 - final_score

    return {
        "result":            result,
        "confidence":        round(confidence, 4),
        "percent":           f"{confidence * 100:.1f}%",
        "efficientnet_score": f"{score1 * 100:.1f}%",
        "resnet_score":       f"{score2 * 100:.1f}%",
        "svm_score":          f"{score3 * 100:.1f}%"
    }