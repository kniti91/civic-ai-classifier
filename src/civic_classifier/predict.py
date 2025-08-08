## civic_classifier/predict.py
from .model_utils import load_model, load_labels, preprocess_image
import numpy as np

def predict_image(img_path):
    model = load_model()
    labels = load_labels()
    img = preprocess_image(img_path)
    
    preds = model.predict(img)[0]  # Get first batch item
    top_index = np.argmax(preds)
    confidence = float(preds[top_index])
    predicted_class = labels[str(top_index)]
    
    return predicted_class, confidence

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m civic_classifier.predict <path_to_image>")
    else:
        path = sys.argv[1]
        label, score = predict_image(path)
        print(f"Prediction: {label} ({score * 100:.2f}% confidence)")