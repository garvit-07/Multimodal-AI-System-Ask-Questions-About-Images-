"""
download_data.py — Download VQA v2 official dataset files
Official source: https://visualqa.org/download.html

Run:
    python download_data.py
"""

import os
import urllib.request
import zipfile

DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")

FILES = [
    # Questions
    (
        "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
        "v2_Questions_Train_mscoco.zip",
    ),
    (
        "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
        "v2_Questions_Val_mscoco.zip",
    ),
    # Annotations
    (
        "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
        "v2_Annotations_Train_mscoco.zip",
    ),
    (
        "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
        "v2_Annotations_Val_mscoco.zip",
    ),
]

COCO_IMAGES = [
    
    (
        "http://images.cocodataset.org/zips/train2014.zip",
        "train2014.zip",
        "images/train2014",
    ),
    (
        "http://images.cocodataset.org/zips/val2014.zip",
        "val2014.zip",
        "images/val2014",
    ),
]


def download_file(url: str, dest: str):
    if os.path.exists(dest):
        print(f"  [Skip] Already exists: {dest}")
        return
    print(f"  [Download] {url}")
    print(f"          → {dest}")

    def progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            print(f"\r    {pct:5.1f}%  ({downloaded/1e6:.1f} MB / {total_size/1e6:.1f} MB)", end="")

    urllib.request.urlretrieve(url, dest, reporthook=progress)
    print()


def extract_zip(zip_path: str, extract_to: str):
    print(f"  [Extract] {zip_path} → {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    print("=" * 60)
    print("  VQA v2 Dataset Downloader")
    print("  Official: https://visualqa.org/download.html")
    print("=" * 60)

    
    print("\n[Step 1] Downloading questions and annotations …")
    for url, filename in FILES:
        dest = os.path.join(DATA_DIR, filename)
        download_file(url, dest)
        extract_zip(dest, DATA_DIR)

    
    print("\n[Step 2] COCO Images")
    print("  train2014 ≈ 13 GB | val2014 ≈ 6 GB")
    ans = input("  Download COCO images? [y/N]: ").strip().lower()

    if ans == "y":
        for url, filename, subfolder in COCO_IMAGES:
            dest = os.path.join(DATA_DIR, filename)
            out_dir = os.path.join(DATA_DIR, subfolder.split("/")[0])
            download_file(url, dest)
            extract_zip(dest, out_dir)
    else:
        print("  Skipped. You can download manually later:")
        for url, _, _ in COCO_IMAGES:
            print(f"    wget {url}")
        print(f"  and extract into {IMAGE_DIR}/")

    print("\n[Done] Dataset ready in:", DATA_DIR)
    print("  → Now run: python train.py --debug")


if __name__ == "__main__":
    main()
