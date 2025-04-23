import os
import shutil
import argparse
import glob
import zipfile
import urllib.request
from biometric_vqa.utils.preprocess_utils import move_folder

# ====================================
# Dataset Info [!]
# ====================================
# Dataset: AMOS22
# Challenge: https://amos22.grand-challenge.org
# Data: https://zenodo.org/records/7262581
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
    print("Downloading amos22.zip...")
    urllib.request.urlretrieve(
        "https://zenodo.org/records/7155725/files/amos22.zip?download=1", "amos22.zip"
    )

    # Extract zip
    print("Extracting amos22.zip...")
    with zipfile.ZipFile("amos22.zip", "r") as zip_ref:
        zip_ref.extractall()

    # Move contents
    for item in os.listdir("amos22"):
        shutil.move(os.path.join("amos22", item), ".")

    # Create directories
    for modality in ["CT", "MRI"]:
        for subdir in ["Images", "Masks"]:
            os.makedirs(os.path.join(f"AMOS22-{modality}", subdir), exist_ok=True)

    # Move image files
    for folder in ["imagesTr", "imagesVa"]:
        if os.path.exists(folder):
            for f in glob.glob(os.path.join(folder, "amos_????.nii.gz")):
                # Extract the number from filename
                num = int(os.path.basename(f)[5:9])
                if num < 507:
                    shutil.move(f, os.path.join("AMOS22-CT", "Images"))
                else:
                    shutil.move(f, os.path.join("AMOS22-MRI", "Images"))

    # Move mask files
    for folder in ["labelsTr", "labelsVa"]:
        if os.path.exists(folder):
            for f in glob.glob(os.path.join(folder, "amos_????.nii.gz")):
                # Extract the number from filename
                num = int(os.path.basename(f)[5:9])
                if num < 507:
                    shutil.move(f, os.path.join("AMOS22-CT", "Masks"))
                else:
                    shutil.move(f, os.path.join("AMOS22-MRI", "Masks"))

    # Move folder to dataset_dir
    folders_to_move = [
        "AMOS22-CT",
        "AMOS22-MRI",
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
