import cv2
import os
import tempfile
from detector import detect_image

def detect_video(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step         = max(1, total_frames // num_frames)
    frame_indices= list(range(0, total_frames, step))[:num_frames]

    scores  = []
    tmp_dir = tempfile.mkdtemp()

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        tmp_path = os.path.join(tmp_dir, f"frame_{idx}.jpg")
        cv2.imwrite(tmp_path, frame)

        result = detect_image(tmp_path)
        scores.append(
            result["confidence"] if result["result"] == "FAKE"
            else 1.0 - result["confidence"]
        )
        os.remove(tmp_path)

    cap.release()

    if not scores:
        return {"result": "UNKNOWN", "confidence": 0.0,
                "percent": "0%", "frames_fake": 0, "frames_total": 0}

    avg_score   = sum(scores) / len(scores)
    frames_fake = sum(1 for s in scores if s >= 0.5)
    result      = "FAKE" if avg_score >= 0.5 else "REAL"
    confidence  = avg_score if result == "FAKE" else 1.0 - avg_score

    return {
        "result":       result,
        "confidence":   round(confidence, 4),
        "percent":      f"{confidence * 100:.1f}%",
        "frames_fake":  frames_fake,
        "frames_total": len(scores)
    }