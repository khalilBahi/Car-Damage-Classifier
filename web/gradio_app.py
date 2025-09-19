"""Gradio interface for car damage classifier."""
import os
import json
import numpy as np
import tensorflow as tf
import gradio as gr
from keras.applications.efficientnet import preprocess_input
from PIL import Image

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH_CANDIDATES = [
    os.path.join(MODELS_DIR, 'best_model.keras'),
    os.path.join(MODELS_DIR, 'final_model.keras'),
]
CLASS_INDICES_JSON = os.path.join(MODELS_DIR, 'class_indices.json')
IMG_SIZE = (224, 224)

def load_class_labels():
    if os.path.exists(CLASS_INDICES_JSON):
        with open(CLASS_INDICES_JSON) as f:
            class_indices = json.load(f)  # {"class_name": index}
        # Sort by the numeric index to preserve training order
        labels = [name for name, idx in sorted(class_indices.items(), key=lambda kv: kv[1])]
        return labels
    return ["crack", "dent", "glass_shatter", "lamp_broken", "scratch", "tire_flat"]

CLASS_LABELS = load_class_labels()

MODEL_PATH = next((p for p in MODEL_PATH_CANDIDATES if os.path.exists(p)), None)
if MODEL_PATH is None:
    raise FileNotFoundError(f"No model file found in: {MODEL_PATH_CANDIDATES}")

print(f"[INFO] Loading model: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

def _prepare(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)[None, ...]
    arr = preprocess_input(arr)
    return arr

def predict(img: Image.Image):
    if img is None:
        return {}
    x = _prepare(img)
    probs = model.predict(x, verbose=0)[0]
    # Ensure Python floats
    result = {cls: float(p) for cls, p in zip(CLASS_LABELS, probs)}
    # Sort descending for label output (gr.Label handles dict)
    return dict(sorted(result.items(), key=lambda kv: kv[1], reverse=True))

DESC = "Upload an image of vehicle damage. The model outputs class probabilities."
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil', label="Car Image"),
    outputs=gr.Label(num_top_classes=3, label="Top Predictions"),
    title='Car Damage Classifier',
    description=DESC,
    allow_flagging="never",
)

if __name__ == '__main__':
    # share=True if local host access blocked; server_name lets you hit from browser on host
    iface.launch(server_name="0.0.0.0", server_port=7860, share=True)