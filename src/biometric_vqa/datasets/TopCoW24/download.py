import os
import shutil
import argparse
import glob
import zipfile
from huggingface_hub import hf_hub_download
from biometric_vqa.utils.preprocess_utils import move_folder


# ====================================
# Dataset Info [!]
# ====================================
# Dataset: TopCoW24
# Challenge: https://topcow24.grand-challenge.org
# Data (official): https://drive.switch.ch/index.php/s/rkqOO3adjmJVlMz
# HF Data: https://huggingface.co/datasets/YongchengYAO/TopCoW24-Seg
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
    # Download dataset from Hugging Face Hub
    hf_hub_download(
        repo_id="YongchengYAO/TopCoW24-Seg",
        filename="TopCoW24.zip",
        repo_type="dataset",
        revision="53469cf5998bd051d29803c6660d4cd2210214bc",  # commit hash on 2025-02-20
        local_dir=".",
    )

    # Extract the downloaded zip file
    with zipfile.ZipFile("TopCoW24.zip", 'r') as zip_ref:
        zip_ref.extractall()

    # Rename directories to standard format
    shutil.rmtree("Images") if os.path.exists("Images") else None
    shutil.rmtree("Masks") if os.path.exists("Masks") else None 
    shutil.move(os.path.join("TopCoW24", "cow_seg_labelsTr"), "Masks")
    shutil.move(os.path.join("TopCoW24", "imagesTr"), "Images")
    # Create directories for CT and MR images and masks
    for modality in ["CT", "MR"]:
        for folder in ["Images", "Masks"]:
            os.makedirs(os.path.join(f"TopCoW24-{modality}", folder), exist_ok=True)

    # Move files based on modality (CT/MR) and type (Images/Masks)
    for folder in ["Images", "Masks"]:
        ct_files = glob.glob(os.path.join(folder, "topcow_ct*.nii.gz"))
        mr_files = glob.glob(os.path.join(folder, "topcow_mr*.nii.gz"))
        for file in ct_files:
            shutil.move(file, os.path.join("TopCoW24-CT", folder))
        for file in mr_files:
            shutil.move(file, os.path.join("TopCoW24-MR", folder))

    # Move folder to dataset_dir
    folders_to_move = [
        "TopCoW24-CT",
        "TopCoW24-MR",
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
