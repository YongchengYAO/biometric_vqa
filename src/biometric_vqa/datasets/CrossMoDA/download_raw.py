import os
import shutil
import glob
import argparse
import zipfile
import urllib.request
from biometric_vqa.utils.preprocess_utils import process_dataset, move_folder


# ====================================
# Dataset Info [!]
# ====================================
# [CrossMoDA]
# Challenge: https://crossmoda-challenge.ml
# Data: https://zenodo.org/records/4662239
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
    # Download dataset from Zenodo
    url = "https://zenodo.org/records/4662239/files/crossmoda_training.zip?download=1"
    print(f"Downloading from {url}...")
    urllib.request.urlretrieve(url, "crossmoda_training.zip")

    # Extract downloaded zip file
    print("Extracting zip file...")
    with zipfile.ZipFile("crossmoda_training.zip", "r") as zip_ref:
        zip_ref.extractall("crossmoda_training")

    os.makedirs("Images", exist_ok=True)
    os.makedirs("Masks", exist_ok=True)
    # Move ceT1 files to Images folder
    for file in glob.glob(
        os.path.join("crossmoda_training", "source_training", "*_ceT1.nii.gz")
    ):
        shutil.move(file, os.path.join("Images", os.path.basename(file)))

    # Move Label files to Masks folder
    for file in glob.glob(
        os.path.join("crossmoda_training", "source_training", "*_Label.nii.gz")
    ):
        shutil.move(file, os.path.join("Masks", os.path.basename(file)))

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
