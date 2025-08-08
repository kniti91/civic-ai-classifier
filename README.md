# Civic AI Classifier

🚧 An open-source image classification library for identifying civic infrastructure issues like potholes, broken roads, garbage dumps, and more — built using TensorFlow and MobileNetV2.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build: Passing](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/yourusername/civic-ai-classifier/actions)

---

## 📦 Features
- Pretrained MobileNetV2-based classifier
- Custom civic issue categories (trained on scraped images)
- CLI-based prediction tool
- Easy to extend and retrain

---

## 📁 Folder Structure
```
civic-ai-classifier/
├── civic_classifier/        # Core module: model loading and prediction
│   ├── __init__.py
│   ├── model_utils.py
│   ├── predict.py
│   └── labels.json
│
├── models/                  # Trained model(s)
│   └── civicfix_mobilenetv2.h5
│
├── scripts/                 # Tools to train, scrape, and preprocess
│   ├── train_mobilenet.py
│   ├── scrape_images_multi.py
│   ├── clean_dataset.py
│   └── split_dataset.py
│
├── dataset/                 # Training and validation dataset (optional)
│
├── examples/                # Sample images to test prediction
│   └── pothole_1.jpg
│
├── tests/                   # Unit tests
│   └── test_predict.py
│
├── requirements.txt         # Dependencies
├── .gitignore               # Ignore venv, models, cache, etc.
└── README.md                # You're here
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/civic-ai-classifier.git
cd civic-ai-classifier
```

### 2. Set up environment
```bash
python -m venv venv
source venv/bin/activate   # Or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Run prediction
```bash
python -m civic_classifier.predict examples/pothole_1.jpg
```
Output:
```
Prediction: pothole (86.73% confidence)
```

---

## 🧠 How It Works
- Uses `MobileNetV2` as the backbone
- Trained on civic issue categories using scraped images from Google
- Labels stored in `labels.json`
- Model trained using `scripts/train_mobilenet.py`

---

## 🧪 Running Tests
```bash
pytest tests/
```

---

## 🛠️ Future Plans
- TensorFlow Lite export for mobile apps
- Flask/FastAPI wrapper for backend APIs
- Auto-retraining pipeline
- Web UI for uploading and testing images

---

## 👥 Authors
- [Nitish Pandey](https://github.com/nitishpandey) — Lead Developer
- Open to contributors!

---

## 📄 License
MIT License. Free for personal or commercial use.

---

> Want to improve civic life using AI? This is your base. Fork it. Extend it. Ship it.
