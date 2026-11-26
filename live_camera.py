
import json
import time
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "sign_model.pth"
LABELS_PATH = BASE_DIR / "labels.json"

IMAGE_SIZE = 64
CONFIDENCE_THRESHOLD = 0.0
STABLE_SECONDS = 0.8
DEVICE = torch.device("cpu")


class SignAIModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("sign_model.pth not found.")

    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)

    model = SignAIModel(len(labels))
    state = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
    model.load_state_dict(state)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    sentence = ""
    candidate = None
    candidate_since = time.time()
    last_added = None

    print("SignAI live camera started!")
    print("Show your hand to the camera.")
    print("Q = Quit | C = Clear sentence")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame.")
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)

            tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                output = model(tensor)
                probabilities = torch.softmax(output, dim=1)[0]

            index = int(probabilities.argmax())
            prediction = labels[index]
            confidence = float(probabilities[index])

            if confidence < CONFIDENCE_THRESHOLD:
                prediction = "uncertain"

            now = time.time()

            if prediction != candidate:
                candidate = prediction
                candidate_since = now

            stable = (
                prediction != "uncertain"
                and now - candidate_since >= STABLE_SECONDS
            )

            if stable and prediction != last_added:
                if prediction.lower() == "space":
                    sentence += " "
                elif prediction.lower() == "del":
                    sentence = sentence[:-1]
                elif prediction.lower() != "nothing":
                    sentence += prediction

                last_added = prediction

            # Re-arm after a different or uncertain sign
            if prediction != last_added and (
                prediction == "uncertain"
                or prediction.lower() == "nothing"
            ):
                if prediction == "uncertain" or prediction.lower() == "nothing":
                    last_added = None

            cv2.putText(
                frame,
                f"Sign: {prediction}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Confidence: {confidence * 100:.1f}%",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Sentence: " + sentence[-45:],
                (20, frame.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            cv2.imshow("SignAI - Live ASL Recognition", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("c"):
                sentence = ""
                last_added = None

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("\nYour sentence:", sentence)


if __name__ == "__main__":
    main()
