import os
from PIL import Image

dataset_path = "dataset/train"

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            os.remove(os.path.join(root, file))
            continue
        try:
            img_path = os.path.join(root, file)
            img = Image.open(img_path)
            img.verify()
        except Exception as e:
            print(f"Removing corrupted file: {img_path}")
            os.remove(img_path)
