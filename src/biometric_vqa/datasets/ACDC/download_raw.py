import os
import shutil
import argparse
import zipfile
import urllib.request
from biometric_vqa.utils.preprocess_utils import process_dataset, move_folder


# ====================================
# Dataset Info [!]
# ====================================
# Dataset: ACDC
# Data: https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html
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
    # Download dataset using pure Python
    print("Downloading ACDC.zip...")
    urllib.request.urlretrieve(
        "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/collection/637218c173e9f0047faa00fb/download",
        "ACDC.zip",
    )

    # Extract and cleanup
    print("Extracting ACDC.zip...")
    with zipfile.ZipFile("ACDC.zip", "r") as zip_ref:
        zip_ref.extractall()
    os.remove("ACDC.zip")

    # Reorganize directory structure
    shutil.move(os.path.join("ACDC", "database"), ".")
    shutil.rmtree("ACDC")

    # Remove documentation and consolidate data
    shutil.move(os.path.join("database", "testing"), ".")
    shutil.move(os.path.join("database", "training"), ".")
    shutil.rmtree("database")

    # Move testing contents to training
    for item in os.listdir("testing"):
        if item.startswith("patient"):
            shutil.move(os.path.join("testing", item), "training")
    shutil.rmtree("testing")

    # Create directories
    os.makedirs("Images", exist_ok=True)
    os.makedirs("Masks", exist_ok=True)

    # Process and organize files
    process_dataset(["training"], "*_gt.nii.gz", "_gt.nii.gz", replace=False)

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
