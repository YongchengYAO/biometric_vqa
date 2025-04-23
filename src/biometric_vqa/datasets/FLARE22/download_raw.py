import os
import shutil
import argparse
import gdown
import zipfile
from biometric_vqa.utils.preprocess_utils import move_folder


# ====================================
# Dataset Info [!]
# ====================================
# Dataset: FLARE22
# Challenge: https://flare22.grand-challenge.org
# Data: https://drive.google.com/drive/folders/1x0l-bxte46QFn5K_ZJzBxp8HsscF8v6t
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
    # Download folder from Google Drive
    gdown.download_folder(
        "https://drive.google.com/drive/folders/130DVWqCFALnpHIqnT6aucNIwYygemNjV?usp=share_link"
    )

    # Extract images
    with zipfile.ZipFile(
        os.path.join("FLARE22_LabeledCase50", "images.zip"), "r"
    ) as zip_ref:
        zip_ref.extractall("Images")

    # Extract masks
    with zipfile.ZipFile(
        os.path.join("FLARE22_LabeledCase50", "labels.zip"), "r"
    ) as zip_ref:
        zip_ref.extractall("Masks")

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
