
import json
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "sign_model.pth"
LABELS_PATH = BASE_DIR / "labels.json"

IMAGE_SIZE = 64
CONFIDENCE_THRESHOLD = 0.70
DEVICE = torch.device("cpu")

st.set_page_config(
    page_title="SignAI - ASL Translator",
    page_icon="🤟",
    layout="wide"
)

# SAME MODEL ARCHITECTURE AS train_model.py
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


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        raise FileNotFoundError(
            "Model missing. Run python train_model.py first."
        )

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

    return model, labels


def predict(image, model, labels):
    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    index = int(probabilities.argmax())
    return (
        labels[index],
        float(probabilities[index]),
        probabilities.tolist()
    )


def show_result(image, model, labels, source):
    letter, confidence, predictions = predict(
        image, model, labels
    )

    st.image(image, caption=source, width=300)

    col1, col2 = st.columns(2)
    col1.metric("Predicted Letter", letter)
    col2.metric("Confidence", f"{confidence * 100:.1f}%")

    if confidence >= CONFIDENCE_THRESHOLD:
        st.success(f"SignAI recognizes: {letter}")

        if st.button(
            f"Add {letter} to sentence",
            key=f"add_{source}_{letter}"
        ):
            st.session_state.sentence += (
                " " if letter in ["space", "SPACE"] else letter
            )
            st.rerun()
    else:
        st.warning("Uncertain prediction. Try a clearer image.")

    with st.expander("See all predictions"):
        for name, score in sorted(
            zip(labels, predictions),
            key=lambda item: item[1],
            reverse=True
        ):
            st.write(f"{name}: {score * 100:.1f}%")


if "sentence" not in st.session_state:
    st.session_state.sentence = ""

st.title("🤟 SignAI")
st.subheader("American Sign Language Recognition System")
st.write("Upload a hand gesture or use your webcam to recognize ASL letters.")

try:
    model, labels = load_model()
except Exception as error:
    st.error(f"Could not load AI model: {error}")
    st.info("Run python train_model.py after saving this file.")
    st.stop()

tab1, tab2 = st.tabs(["📷 Upload Image", "🎥 Webcam"])

with tab1:
    st.header("Upload an ASL hand gesture")
    uploaded_file = st.file_uploader(
        "Choose a hand gesture picture",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        show_result(image, model, labels, "Uploaded image")

with tab2:
    st.header("Recognize using your webcam")
    camera_image = st.camera_input("Take a picture of your hand gesture")

    if camera_image:
        image = Image.open(camera_image)
        show_result(image, model, labels, "Webcam image")

st.divider()

st.header("📝 Sentence Builder")
st.text_area(
    "Your recognized sentence",
    key="sentence",
    height=120
)

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Clear sentence"):
        st.session_state.sentence = ""
        st.rerun()

with col2:
    if st.button("Delete last character"):
        st.session_state.sentence = st.session_state.sentence[:-1]
        st.rerun()

with col3:
    st.download_button(
        "Download sentence",
        data=st.session_state.sentence,
        file_name="signai_sentence.txt"
    )
