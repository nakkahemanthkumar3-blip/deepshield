import os
from detector import detect_image

# ── Test Configuration ─────────────────────────────────────────────
REAL_FOLDER = "real_and_fake_face/training_real"
FAKE_FOLDER = "real_and_fake_face/training_fake"
NUM_IMAGES  = 10  # Test 10 real + 10 fake

print("=" * 50)
print("DeepShield — Accuracy Test")
print("=" * 50)

results = []

# ── Test Real Images ───────────────────────────────────────────────
print("\nTesting REAL images...")
real_files = os.listdir(REAL_FOLDER)[:NUM_IMAGES]

for filename in real_files:
    path = os.path.join(REAL_FOLDER, filename)
    try:
        detection = detect_image(path)
        correct   = detection['result'] == 'REAL'
        results.append({
            'file':     filename,
            'actual':   'REAL',
            'predicted':detection['result'],
            'correct':  correct,
            'confidence': detection['percent']
        })
        status = "✓" if correct else "✗"
        print(f"  {status} {filename[:30]} → {detection['result']} ({detection['percent']})")
    except Exception as e:
        print(f"  Error: {filename} — {e}")

# ── Test Fake Images ───────────────────────────────────────────────
print("\nTesting FAKE images...")
fake_files = os.listdir(FAKE_FOLDER)[:NUM_IMAGES]

for filename in fake_files:
    path = os.path.join(FAKE_FOLDER, filename)
    try:
        detection = detect_image(path)
        correct   = detection['result'] == 'FAKE'
        results.append({
            'file':     filename,
            'actual':   'FAKE',
            'predicted':detection['result'],
            'correct':  correct,
            'confidence': detection['percent']
        })
        status = "✓" if correct else "✗"
        print(f"  {status} {filename[:30]} → {detection['result']} ({detection['percent']})")
    except Exception as e:
        print(f"  Error: {filename} — {e}")

# ── Calculate Results ──────────────────────────────────────────────
total   = len(results)
correct = sum(1 for r in results if r['correct'])
accuracy = (correct / total * 100) if total > 0 else 0

# Count specific results
real_correct = sum(1 for r in results if r['actual'] == 'REAL' and r['correct'])
fake_correct = sum(1 for r in results if r['actual'] == 'FAKE' and r['correct'])

print("\n" + "=" * 50)
print("RESULTS SUMMARY")
print("=" * 50)
print(f"Total images tested : {total}")
print(f"Correct predictions : {correct}")
print(f"Wrong predictions   : {total - correct}")
print(f"Real images correct : {real_correct}/{NUM_IMAGES}")
print(f"Fake images correct : {fake_correct}/{NUM_IMAGES}")
print(f"\nFINAL ACCURACY: {accuracy:.1f}%")
print("=" * 50)

if accuracy >= 85:
    print("EXCELLENT! Above industry standard (85%)")
elif accuracy >= 70:
    print("GOOD! Acceptable accuracy for FYP")
else:
    print("Needs improvement")