## civic_classifier/model_utils.py
import tensorflow as tf
import numpy as np
import json
import os
from tensorflow.keras.preprocessing import image

# Load model only once
def load_model(model_path="models/civicfix_mobilenetv2.h5"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model

# Load class labels
def load_labels(label_path="civic_classifier/labels.json"):
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Labels file not found: {label_path}")
    with open(label_path, "r") as f:
        labels = json.load(f)
    return labels

# Preprocess a single image for prediction
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array