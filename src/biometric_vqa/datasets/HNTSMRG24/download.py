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
# [HNTSMRG24]
# Challenge: https://hntsmrg24.grand-challenge.org
# Data: https://zenodo.org/records/11199559
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
    print("Downloading dataset from Zenodo...")
    url = "https://zenodo.org/records/11199559/files/HNTSMRG24_train.zip?download=1"
    zip_path = "HNTSMRG24_train.zip"
    urllib.request.urlretrieve(url, zip_path)

    # Extract downloaded zip file
    print("Extracting zip file...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall()

    # Create directory structure for both timepoints
    for timepoint in ["HNTSMRG24-midRT", "HNTSMRG24-preRT"]:
        os.makedirs(os.path.join(timepoint, "Images"), exist_ok=True)
        os.makedirs(os.path.join(timepoint, "Masks"), exist_ok=True)

    # Process dataset for pre-RT timepoint
    process_dataset(
        data_dirs=["HNTSMRG24_train"],
        seg_pattern="*_preRT_mask.nii.gz",
        base_suffix="_mask.nii.gz",
        img_suffix="_T2.nii.gz",
        out_dir="HNTSMRG24-preRT",
    )

    # Process dataset for mid-RT timepoint
    process_dataset(
        data_dirs=["HNTSMRG24_train"],
        seg_pattern="*_midRT_mask.nii.gz",
        base_suffix="_mask.nii.gz",
        img_suffix="_T2.nii.gz",
        out_dir="HNTSMRG24-midRT",
    )

    # Move folder to dataset_dir
    folders_to_move = [
        "HNTSMRG24-midRT",
        "HNTSMRG24-preRT",
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
