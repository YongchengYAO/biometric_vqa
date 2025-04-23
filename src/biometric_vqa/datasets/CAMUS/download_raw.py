import os
import zipfile
import urllib.request
import shutil
import argparse
from biometric_vqa.utils.preprocess_utils import process_dataset, move_folder


# ====================================
# Dataset Info [!]
# ====================================
# Dataset: CAMUS
# Challenge: https://www.creatis.insa-lyon.fr/Challenge/camus/
# Data: https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8
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
    # Download dataset
    urllib.request.urlretrieve(
        "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1/collection/6373703d73e9f0047faa1bc8/download",
        "CAMUS.zip",
    )

    # Extract zip using zipfile
    with zipfile.ZipFile("CAMUS.zip", "r") as zip_ref:
        zip_ref.extractall()
    # Move nifti database
    shutil.move(os.path.join("CAMUS_public", "database_nifti"), ".")

    # Create directories
    os.makedirs("Images", exist_ok=True)
    os.makedirs("Masks", exist_ok=True)

    # Process dataset
    process_dataset(
        data_dirs=["database_nifti"],
        seg_pattern="*_gt.nii.gz",
        base_suffix="_gt.nii.gz",
    )

    # Remove 2D images and masks
    for file in os.listdir("Images"):
        if file.endswith("_ED.nii.gz") or file.endswith("_ES.nii.gz"):
            os.remove(os.path.join("Images", file))
    for file in os.listdir("Masks"):
        if file.endswith("_ED_gt.nii.gz") or file.endswith("_ES_gt.nii.gz"):
            os.remove(os.path.join("Masks", file))

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
