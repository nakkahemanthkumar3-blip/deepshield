import cv2
import numpy as np
from PIL import Image

def analyze_eyes(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml'
    )
    eyes = eye_cascade.detectMultiScale(gray, 1.1, 4)

    if len(eyes) == 0:
        return {"score": 0.75, "detail": "⚠ No eyes detected — suspicious"}
    if len(eyes) >= 2:
        eye1, eye2 = eyes[0], eyes[1]
        height_diff = abs(int(eye1[1]) - int(eye2[1]))
        width_diff  = abs(int(eye1[2]) - int(eye2[2]))
        if height_diff > 15 or width_diff > 20:
            return {"score": 0.70, "detail": "⚠ Asymmetric eyes — possible fake"}
        else:
            return {"score": 0.15, "detail": "✅ Eyes look symmetric and natural"}
    return {"score": 0.30, "detail": "⚠ Only one eye detected"}

def analyze_skin_texture(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()

    if variance < 100:
        return {"score": 0.80, "detail": "⚠ Skin too smooth — GAN artifact"}
    elif variance < 300:
        return {"score": 0.55, "detail": "⚠ Slightly unnatural texture"}
    elif variance > 1500:
        return {"score": 0.60, "detail": "⚠ Unnatural sharpness detected"}
    else:
        return {"score": 0.15, "detail": "✅ Skin texture looks natural"}

def analyze_face_geometry(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return {"score": 0.85, "detail": "⚠ No face detected — suspicious"}
    if len(faces) > 1:
        return {"score": 0.65, "detail": "⚠ Multiple faces — suspicious"}

    x, y, w, h = faces[0]
    ratio = w / h

    if ratio < 0.55:
        return {"score": 0.70, "detail": "⚠ Face too narrow — unusual"}
    elif ratio > 0.95:
        return {"score": 0.70, "detail": "⚠ Face too wide — unusual"}
    elif 0.65 <= ratio <= 0.85:
        return {"score": 0.10, "detail": "✅ Face geometry looks normal"}
    else:
        return {"score": 0.40, "detail": "⚠ Slightly unusual proportions"}

def analyze_frequency(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_float = np.float32(img)
    dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])
    magnitude_log = np.log(magnitude + 1)

    mean_freq = np.mean(magnitude_log)
    std_freq  = np.std(magnitude_log)

    if std_freq > 4.0:
        return {"score": 0.75, "detail": "⚠ GAN frequency artifacts detected"}
    elif std_freq > 3.0:
        return {"score": 0.50, "detail": "⚠ Slight frequency anomaly"}
    else:
        return {"score": 0.15, "detail": "✅ Frequency pattern looks natural"}

def run_biometric_checks(image_path):
    eye_result  = analyze_eyes(image_path)
    skin_result = analyze_skin_texture(image_path)
    face_result = analyze_face_geometry(image_path)
    freq_result = analyze_frequency(image_path)

    biometric_score = (
        eye_result["score"]  * 0.30 +
        skin_result["score"] * 0.25 +
        face_result["score"] * 0.25 +
        freq_result["score"] * 0.20
    )

    return {
        "biometric_score": round(biometric_score, 4),
        "eye_analysis":    eye_result["detail"],
        "skin_analysis":   skin_result["detail"],
        "face_geometry":   face_result["detail"],
        "frequency":       freq_result["detail"]
    }  