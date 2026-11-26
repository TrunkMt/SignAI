
# SignAI 🤟

An AI-powered American Sign Language alphabet recognition project.

## Features

- Webcam image capture
- ASL letter recognition
- TensorFlow image classification
- Prediction confidence
- Text generation
- Space and delete functions

## Technologies

- Python
- TensorFlow
- Streamlit
- OpenCV
- NumPy
- Pillow

## Installation

Install the requirements:

```bash
python -m pip install -r requirements.txt
```

## Step 1: Collect images

```bash
python collect_data.py
```

Press the corresponding keyboard letter to save an image of that sign.

## Step 2: Train the AI

```bash
python train_model.py
```

This creates:

- sign_model.keras
- labels.json

## Step 3: Run SignAI

```bash
streamlit run App.py
```

Open http://localhost:8501

## Limitations

The initial model recognizes 24 static ASL letters.
J and Z require movement recognition.

This is an educational prototype, not a complete
ASL translation system.
