
import cv2
from pathlib import Path
import time

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"

LETTERS = list("ABCDE")
TARGET_PER_CLASS = 500

for letter in LETTERS:
    (DATASET_DIR / letter).mkdir(parents=True, exist_ok=True)

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not access webcam.")

current_letter = "A"
last_saved = 0
saved_count = 0

print("SignAI - A to E data collection")
print("Press A-E to select a class.")
print("Press SPACE to save an image.")
print("Press Q to quit.")

while True:
    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)
    display = frame.copy()

    existing = len(list((DATASET_DIR / current_letter).glob("*.jpg")))

    cv2.putText(display, f"Class: {current_letter}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    cv2.putText(display, f"Images: {existing}/{TARGET_PER_CLASS}",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)

    cv2.putText(display, "A-E: Select | SPACE: Save | Q: Quit",
                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 200, 255), 2)

    cv2.imshow("SignAI Data Collection", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key in [ord(x.lower()) for x in LETTERS]:
        current_letter = chr(key).upper()

    elif key == 32:
        now = time.time()

        if now - last_saved >= 0.3:
            folder = DATASET_DIR / current_letter
            existing_files = list(folder.glob("*.jpg"))

            if len(existing_files) < TARGET_PER_CLASS:
                filename = f"{current_letter}_extra_{int(now * 1000)}.jpg"
                cv2.imwrite(str(folder / filename), frame)
                saved_count += 1
                last_saved = now
                print(f"Saved {current_letter}: {len(existing_files) + 1}")

            else:
                print(f"{current_letter} already has enough images.")

camera.release()
cv2.destroyAllWindows()

print(f"Collection finished. New images saved: {saved_count}")
