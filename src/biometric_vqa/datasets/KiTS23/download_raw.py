import os
import subprocess
import shutil
import argparse
import glob
from biometric_vqa.utils.preprocess_utils import move_folder


# ====================================
# Dataset Info [!]
# ====================================
# Dataset: KiTS23
# Challenge: https://kits-challenge.org/kits23/
# Official Release: https://github.com/neheller/kits23#data-download
# HF Release: https://huggingface.co/datasets/YongchengYAO/KiTS23-Lite 
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
    # Clone and setup KiTS23
    subprocess.run(["git", "clone", "https://github.com/neheller/kits23"])
    os.chdir("kits23")
    subprocess.run(["pip3", "install", "-e", "."])
    subprocess.run(["kits23_download_data"])
    shutil.move("dataset", "..")
    os.chdir("..")

    # Create output directories
    os.makedirs("Images", exist_ok=True)
    os.makedirs("Masks", exist_ok=True)

    # Process each directory in dataset folder
    for dir_path in glob.glob(os.path.join("dataset", "*", "")):
        if os.path.isdir(dir_path):
            # Get directory name without trailing slash
            id = os.path.basename(os.path.normpath(dir_path))

            # Process imaging file
            img_path = os.path.join(dir_path, "imaging.nii.gz")
            if os.path.exists(img_path):
                shutil.move(img_path, os.path.join("Images", f"{id}.nii.gz"))

            # Process segmentation file
            seg_path = os.path.join(dir_path, "segmentation.nii.gz")
            if os.path.exists(seg_path):
                shutil.move(seg_path, os.path.join("Masks", f"{id}.nii.gz"))

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
