import os
import zipfile
import shutil
import argparse
from huggingface_hub import hf_hub_download
from biometric_vqa.utils.preprocess_utils import move_folder


# ====================================
# Dataset Info [!]
# ====================================
# Dataset: Ceph-Biometrics-400
# Data (original): https://figshare.com/s/37ec464af8e81ae6ebbf
# Data (HF): https://huggingface.co/datasets/YongchengYAO/Ceph-Biometrics-400
# Format (original): bm
# Format (HF): nii.gz
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
    # Download and extract dataset
    for file in ["Images.zip", "Landmarks.zip", "Landmarks-fig.zip"]:
        # Download and extract dataset
        hf_hub_download(
            repo_id="YongchengYAO/Ceph-Biometrics-400",
            filename=file,
            repo_type="dataset",
            revision="8cd93443d4ba6d327c74dde39184d846034d920a", # commit hash on 2025-05-09
            local_dir=".",
        )
        print(f"Extracting {file}... This may take some time.") 
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall()
        os.remove(file)

    # Move folder to dataset_dir
    folders_to_move = [
        "Images",
        "Landmarks",
        "Landmarks-fig",
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
