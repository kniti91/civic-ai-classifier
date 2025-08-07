from civic_classifier.predict import predict_image
import os

def test_prediction_runs():
    image_path = "examples/pothole_1.jpg"
    assert os.path.exists(image_path)
    label, confidence = predict_image(image_path)
    assert isinstance(label, str)
    assert 0.0 <= confidence <= 1.0
