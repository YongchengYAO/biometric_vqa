import os
import shutil
import argparse
import tarfile
import requests
import numpy as np
import nibabel as nib
from biometric_vqa.utils.preprocess_utils import move_folder


# ====================================
# Dataset Info [!]
# ====================================
# Dataset: autoPET-III
# Challenge: https://autopet-iii.grand-challenge.org
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
    url = "https://it-portal.med.uni-muenchen.de/autopet/Autopet_v1.1.tgz"
    filename = "Autopet_v1.1.tgz"
    print(f"Downloading {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Extract tar archive
    try:
        with tarfile.open("Autopet_v1.1.tgz", "r:gz") as tar:
            tar.extractall()
    except tarfile.TarError as e:
        print(f"Error extracting archive: {e}")
    except FileNotFoundError:
        print("Archive file not found")

    # Set write permissions
    extracted_dir = "2024-05-10_Autopet_v1.1"
    for root, dirs, files in os.walk(extracted_dir):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o700)
        for f in files:
            os.chmod(os.path.join(root, f), 0o600)

    # Move files
    shutil.move(os.path.join(extracted_dir, "imagesTr"), ".")
    shutil.move(os.path.join(extracted_dir, "labelsTr"), ".")

    # Create directories
    os.makedirs("Images-CT", exist_ok=True)
    os.makedirs("Images-PET", exist_ok=True)

    # Move files to respective directories
    for file in os.listdir("imagesTr"):
        if file.endswith("_0000.nii.gz"):
            shutil.move(os.path.join("imagesTr", file), os.path.join("Images-CT", file))
        elif file.endswith("_0001.nii.gz"):
            shutil.move(
                os.path.join("imagesTr", file), os.path.join("Images-PET", file)
            )
    os.rename("labelsTr", "Masks")

    # Check and remove empty masks and corresponding images
    mask_files = [f for f in os.listdir("Masks") if f.endswith(".nii.gz")]
    for mask_file in mask_files:
        # Get the ID from mask filename
        patient_id = mask_file.replace(".nii.gz", "")

        # Load and check mask
        mask_path = os.path.join("Masks", mask_file)
        mask_data = nib.load(mask_path).get_fdata()

        # If mask is empty (contains only zeros)
        if np.all(mask_data == 0):
            print(f"Found empty mask for {patient_id}, removing associated files...")

            # Remove mask file
            os.remove(mask_path)

            # Remove corresponding CT image
            ct_file = f"{patient_id}_0000.nii.gz"
            ct_path = os.path.join("Images-CT", ct_file)
            if os.path.exists(ct_path):
                print(f"Removing CT image: {ct_path}")
                os.remove(ct_path)

            # Remove corresponding PET image
            pet_file = f"{patient_id}_0001.nii.gz"
            pet_path = os.path.join("Images-PET", pet_file)
            if os.path.exists(pet_path):
                print(f"Removing PET image: {pet_path}")
                os.remove(pet_path)

    # Move folder to dataset_dir
    folders_to_move = [
        "Images-CT",
        "Images-PET",
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
