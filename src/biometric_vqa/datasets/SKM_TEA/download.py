import os
import shutil
import argparse
import zipfile
from huggingface_hub import hf_hub_download
from biometric_vqa.utils.preprocess_utils import move_folder


# ====================================
# Dataset Info [!]
# ====================================
# Dataset: SKM-TEA
# Data: https://aimi.stanford.edu/datasets/skm-tea-knee-mri
# Format: nii.gz (converted from raw data)
# ====================================


# Define HuggingFace dataset ID
BiometricVQA_SKMTEA_HF_ID = os.environ.get(
    "BiometricVQA_SKMTEA_HF_ID", "YongchengYAO/SKM-TEA-nii"
)


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
        repo_id=BiometricVQA_SKMTEA_HF_ID,
        filename="SKM-TEA-nii.zip",
        repo_type="dataset",
        local_dir=".",
    )

    # Extract the downloaded zip file
    with zipfile.ZipFile("SKM-TEA-nii.zip", "r") as zip_ref:
        zip_ref.extractall()

    # Rename directories to match standard format
    shutil.rmtree("Images-E1") if os.path.exists("Images-E1") else None
    shutil.rmtree("Images-E2") if os.path.exists("Images-E2") else None
    shutil.rmtree("Masks") if os.path.exists("Masks") else None
    shutil.move(os.path.join("SKM-TEA-nii", "img_nii_E1"), "Images-E1")
    shutil.move(os.path.join("SKM-TEA-nii", "img_nii_E2"), "Images-E2")
    shutil.move(os.path.join("SKM-TEA-nii", "seg-nii"), "Masks")

    # Move folder to dataset_dir
    folders_to_move = [
        "Images-E1",
        "Images-E2",
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
