from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
import os

# Define your categories and multiple search terms for each
search_terms = {
    "pothole": [
        "pothole road",
        "damaged road pothole",
        "street pothole",
        "cracked asphalt pothole"
    ],
    "garbage dump": [
        "garbage dump",
        "waste dump site",
        "trash heap",
        "pile of garbage on road"
    ],
    "open manhole": [
        "open manhole",
        "open sewer cover",
        "uncovered manhole",
        "dangerous open manhole"
    ],
    "road crack": [
        "road crack",
        "asphalt crack",
        "damaged cracked road",
        "street surface crack"
    ],
    "waterlogging": [
        "waterlogging",
        "flooded road",
        "rainwater road flood",
        "street water flooding"
    ],
    "broken road": [
        "broken road",
        "damaged asphalt road",
        "uneven broken street",
        "collapsed road surface"
    ]
}

dataset_path = "dataset/train"
max_images_per_query = 300  # per search term

def download_from_google(folder, query):
    crawler = GoogleImageCrawler(storage={"root_dir": folder})
    crawler.crawl(keyword=query, max_num=max_images_per_query)

def download_from_bing(folder, query):
    crawler = BingImageCrawler(storage={"root_dir": folder})
    crawler.crawl(keyword=query, max_num=max_images_per_query)

for category, queries in search_terms.items():
    category_folder = os.path.join(dataset_path, category)
    os.makedirs(category_folder, exist_ok=True)

    for query in queries:
        print(f"\n[Google] Downloading '{query}' for category '{category}'...")
        download_from_google(category_folder, query)

        print(f"[Bing] Downloading '{query}' for category '{category}'...")
        download_from_bing(category_folder, query)

print("\n✅ Download complete. Check the dataset/train/ folder.")
