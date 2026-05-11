import os
import urllib.request
import zipfile
from pathlib import Path
from dotenv import dotenv_values

env = dotenv_values('.env')


def download_and_prepare_dataset():
    if Path("images").exists() and Path("embed_data").exists():
        print("Dataset already exists. Skipping download.")
        return env  # Trả về env đã load

    url = env.get("DATASET_ZIP_URL")
    if not url:
        raise ValueError("DATASET_ZIP_URL is missing in .env")

    zip_path = Path("DeepFashion.zip")
    if not zip_path.exists():
        print("Downloading dataset from Hugging Face...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"Downloaded: {zip_path}")

    print("Extracting zip...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
    zip_path.unlink()
    print("Extraction complete.")

    # Xác định đường dẫn
    base_folder = Path("DeepFashion")
    images_path = base_folder / "images" if base_folder.exists() else Path("images")
    embed_path = base_folder / "embed_data" if base_folder.exists() else Path("embed_data")

    if not images_path.exists():
        raise FileNotFoundError(f"'images/' not found at {images_path}")
    if not embed_path.exists():
        raise FileNotFoundError(f"'embed_data/' not found at {embed_path}")

    env["DEFAULT_IMAGES_PATH"] = str(images_path)
    env["INDEX_PATH"] = str(embed_path)

    print(f"Using images path: {images_path}")
    print(f"Using embed path: {embed_path}")
    return env
