import os
import shutil
import argparse
import zipfile
from huggingface_hub import hf_hub_download
from biometric_vqa.utils.data_conversion import convert_mha_to_nifti
from biometric_vqa.utils.preprocess_utils import move_folder


# ====================================
# Dataset Info [!]
# ====================================
# Dataset: ToothFairy2
# Data: https://ditto.ing.unimore.it/toothfairy2/
# Format: mha
# ====================================


# Define HuggingFace dataset ID
BiometricVQA_ToothFairy2_HF_ID = os.environ.get(
    "BiometricVQA_ToothFairy2_HF_ID", "YongchengYAO/ToothFairy2"
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
        repo_id=BiometricVQA_ToothFairy2_HF_ID,
        filename="ToothFairy2.zip",
        repo_type="dataset",
        local_dir=".",
    )

    # Extract the downloaded zip file
    with zipfile.ZipFile("ToothFairy2.zip", "r") as zip_ref:
        zip_ref.extractall()

    # Rename directories to match standard format using Windows path conventions
    convert_mha_to_nifti(os.path.join("ToothFairy2", "imagesTr"), "Images")
    convert_mha_to_nifti(os.path.join("ToothFairy2", "labelsTr"), "Masks")

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
