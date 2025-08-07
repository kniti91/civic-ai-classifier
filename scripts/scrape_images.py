from icrawler.builtin import GoogleImageCrawler
import os

categories = [
    "pothole road",
    "garbage dump",
    "open manhole",
    "road crack",
    "waterlogging",
    "broken road"
]

dataset_path = "dataset/train"

for category in categories:
    folder = os.path.join(dataset_path, category)
    os.makedirs(folder, exist_ok=True)
    print(f"Downloading images for: {category}")
    
    crawler = GoogleImageCrawler(storage={"root_dir": folder})
    crawler.crawl(keyword=category, max_num=5000)
