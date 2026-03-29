import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from biometric import run_biometric_checks
from ensemble import ensemble_detect

MODEL_PATH     = "models/deepfake_model.pth"
IMG_SIZE       = (224, 224)
FAKE_THRESHOLD = 0.5

_model = None

def get_model():
    global _model
    if _model is None:
        print("[Detector] Loading model...")
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, 2
        )
        model.load_state_dict(torch.load(
            MODEL_PATH, map_location="cpu", weights_only=True))
        model.eval()
        _model = model
        print("[Detector] Model ready!")
    return _model

def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)

def detect_image(image_path):
    # Step 1 — Ensemble model detection
    ensemble = ensemble_detect(image_path)

    # Step 2 — Biometric checkpoints
    biometric = run_biometric_checks(image_path)
    bio_score = biometric["biometric_score"]

    # Step 3 — Combine Ensemble + Biometric
    # Convert ensemble confidence to fake score
    if ensemble["result"] == "FAKE":
        ensemble_fake_score = ensemble["confidence"]
    else:
        ensemble_fake_score = 1.0 - ensemble["confidence"]

    # Final score: 80% ensemble + 20% biometric
    final_score = (ensemble_fake_score * 0.80) + (bio_score * 0.20)

    result     = "FAKE" if final_score >= FAKE_THRESHOLD else "REAL"
    confidence = final_score if result == "FAKE" else 1.0 - final_score

    return {
        "result":             result,
        "confidence":         round(confidence, 4),
        "percent":            f"{confidence * 100:.1f}%",
        "eye_analysis":       biometric["eye_analysis"],
        "skin_analysis":      biometric["skin_analysis"],
        "face_geometry":      biometric["face_geometry"],
        "frequency":          biometric["frequency"],
        "efficientnet_score": ensemble["efficientnet_score"],
        "resnet_score":       ensemble["resnet_score"],
        "svm_score":          ensemble["svm_score"]
    }

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: py detector.py <image_path>")
    else:
        output = detect_image(sys.argv[1])
        print(f"Result:       {output['result']}")
        print(f"Confidence:   {output['percent']}")
        print(f"EfficientNet: {output['efficientnet_score']}")
        print(f"ResNet50:     {output['resnet_score']}")
        print(f"SVM:          {output['svm_score']}")