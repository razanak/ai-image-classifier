# 🤖 AI Image Classifier

My first AI project! A web app that classifies uploaded images using a
pretrained deep learning model (MobileNetV2, trained on ImageNet — 1000
everyday object categories).

Upload a photo of pretty much anything, and the app returns its top 5
guesses with confidence scores.

## Demo

![screenshot](screenshot.png)
*(add your own screenshot here after running the app)*

## How it works

1. The app loads **MobileNetV2**, a convolutional neural network already
   trained on 1.4 million images (the ImageNet dataset).
2. When you upload an image, it's resized to 224x224 pixels and
   preprocessed to match what the model expects.
3. The model outputs probabilities for 1000 possible classes.
4. The app displays the top 5 most likely classes with their confidence
   scores.

This is called **transfer learning / inference with a pretrained model** —
no training required, which makes it a great starting point for a first AI
project.

## Getting started

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/ai-image-classifier.git
cd ai-image-classifier
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

Your browser should open automatically to `http://localhost:8501`. If not,
open that URL manually.

## Tech stack

- **Python**
- **TensorFlow / Keras** — pretrained MobileNetV2 model
- **Streamlit** — web interface
- **Pillow** — image handling

## Possible next steps

- Deploy it publicly with [Streamlit Community Cloud](https://streamlit.io/cloud) (free)
- Swap MobileNetV2 for a different pretrained model (e.g. ResNet50, EfficientNet)
- Fine-tune the model on your own custom image categories
- Add webcam support so you can classify live photos

## License

MIT — feel free to use this project however you like.
