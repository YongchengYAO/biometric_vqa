import os
import shutil
import argparse
import zipfile
import urllib.request
import py7zr
from biometric_vqa.utils.preprocess_utils import match_and_clean_files, move_folder


# ====================================
# Dataset Info [!]
# ====================================
# Dataset: AbdomenCT-1K
# Website: https://github.com/JunMa11/AbdomenCT-1K
# Data: https://zenodo.org/records/5903099; https://zenodo.org/records/5903846; https://zenodo.org/records/5903769
# Format: nii.gz
# ====================================


def download_and_extract(dataset_dir, dataset_name):
    # Download files
    current_dir = os.getcwd()
    os.chdir(dataset_dir)
    tmp_dir = os.path.join(dataset_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    os.chdir(tmp_dir)
    print(f"Downloading {dataset_name} dataset to {dataset_dir}...")

    # ====================================
    # Add download logic here [!]
    # ====================================
    # Downloading dataset
    urls = [
        "https://zenodo.org/records/5903099/files/AbdomenCT-1K-ImagePart1.zip?download=1",
        "https://zenodo.org/records/5903846/files/AbdomenCT-1K-ImagePart2.zip?download=1",
        "https://zenodo.org/records/5903769/files/AbdomenCT-1K-ImagePart3.zip?download=1",
        "https://zenodo.org/records/5903769/files/Mask.7z?download=1",
    ]
    filenames = [
        "AbdomenCT-1K-ImagePart1.zip",
        "AbdomenCT-1K-ImagePart2.zip",
        "AbdomenCT-1K-ImagePart3.zip",
        "Mask.7z",
    ]
    # Download files using urllib
    for url, filename in zip(urls, filenames):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
    # Extract zip archives
    for filename in filenames[:3]:
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(filename, "r") as zip_ref:
            zip_ref.extractall()
    # Extract 7z archive
    print(f"Extracting Mask.7z...")
    with py7zr.SevenZipFile("Mask.7z", "r") as archive:
        archive.extractall(path="Masks")

    # Combine data
    for part in ["AbdomenCT-1K-ImagePart2", "AbdomenCT-1K-ImagePart3"]:
        for file in os.listdir(part):
            if file.endswith(".nii.gz"):
                shutil.move(os.path.join(part, file), "AbdomenCT-1K-ImagePart1")
    shutil.rmtree("Images") if os.path.exists("Images") else None
    shutil.move("AbdomenCT-1K-ImagePart1", "Images")

    # Removing image files without corresponding mask files
    match_and_clean_files("Images", "Masks")

    # Move folder to dataset_dir
    folders_to_move = [
        "Images",
        "Masks",
    ]
    for folder in folders_to_move:
        move_folder(
            os.path.join(tmp_dir, folder),
            os.path.join(dataset_dir, folder),
            create_dest=True,
        )
    # ====================================

    print(f"Download and extraction completed for {dataset_name}")
    os.chdir(dataset_dir)
    shutil.rmtree(tmp_dir)
    os.chdir(current_dir)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Download and extract dataset")
    parser.add_argument(
        "-d",
        "--dir_datasets_data",
        help="Directory path where datasets will be stored",
        required=True,
    )
    parser.add_argument(
        "-n",
        "--dataset_name",
        help="Name of the dataset",
        required=True,
    )
    args = parser.parse_args()

    # Create dataset directory
    dataset_dir = os.path.join(args.dir_datasets_data, args.dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Change to dataset directory
    os.chdir(dataset_dir)

    # Download and extract dataset
    download_and_extract(dataset_dir, args.dataset_name)
