import os
import shutil
from sklearn.model_selection import train_test_split

source_path = "dataset/train"
val_path = "dataset/validation"

os.makedirs(val_path, exist_ok=True)

for category in os.listdir(source_path):
    cat_path = os.path.join(source_path, category)
    if not os.path.isdir(cat_path):
        continue
    images = os.listdir(cat_path)
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

    val_cat_path = os.path.join(val_path, category)
    os.makedirs(val_cat_path, exist_ok=True)

    for img in val_imgs:
        shutil.move(os.path.join(cat_path, img), os.path.join(val_cat_path, img))
