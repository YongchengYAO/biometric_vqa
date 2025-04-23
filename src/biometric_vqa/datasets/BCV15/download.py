import os
import shutil
import synapseclient
import zipfile
import argparse
from biometric_vqa.utils.preprocess_utils import move_folder

# ====================================
# Dataset Info [!]
# ====================================
# Dataset: BCV15
# Challenge: https://www.synapse.org/Synapse:syn3193805/wiki/89480
# Data: Abdomen: https://www.synapse.org/Synapse:syn3193805/wiki/217789;
#       Cervix: https://www.synapse.org/Synapse:syn3193805/wiki/217790
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
    # Initialize Synapse client
    syn = synapseclient.Synapse()
    token = os.environ.get("SYNAPSE_TOKEN")
    if not token:
        raise ValueError("SYNAPSE_TOKEN environment variable not set")
    syn.login(authToken=token)

    # Download datasets
    syn.get("syn3379050", downloadLocation="Abdomen")
    syn.get("syn3546986", downloadLocation="Cervix")

    # Extract zip files
    with zipfile.ZipFile(os.path.join("Abdomen", "RawData.zip"), "r") as zip_ref:
        zip_ref.extractall("Abdomen")
    with zipfile.ZipFile(os.path.join("Cervix", "CervixRawData.zip"), "r") as zip_ref:
        zip_ref.extractall("Cervix")

    # Create directories and move files for Abdomen
    os.makedirs(os.path.join("Abdomen", "Images"), exist_ok=True)
    os.makedirs(os.path.join("Abdomen", "Masks"), exist_ok=True)
    for f in os.listdir(os.path.join("Abdomen", "RawData", "Training", "img")):
        if f.endswith(".nii.gz"):
            shutil.move(
                os.path.join("Abdomen", "RawData", "Training", "img", f),
                os.path.join("Abdomen", "Images", f),
            )
    for f in os.listdir(os.path.join("Abdomen", "RawData", "Training", "label")):
        if f.endswith(".nii.gz"):
            shutil.move(
                os.path.join("Abdomen", "RawData", "Training", "label", f),
                os.path.join("Abdomen", "Masks", f),
            )

    # Create directories and move files for Cervix
    os.makedirs(os.path.join("Cervix", "Images"), exist_ok=True)
    os.makedirs(os.path.join("Cervix", "Masks"), exist_ok=True)
    for f in os.listdir(os.path.join("Cervix", "RawData", "Training", "img")):
        if f.endswith(".nii.gz"):
            shutil.move(
                os.path.join("Cervix", "RawData", "Training", "img", f),
                os.path.join("Cervix", "Images", f),
            )
    for f in os.listdir(os.path.join("Cervix", "RawData", "Training", "label")):
        if f.endswith(".nii.gz"):
            shutil.move(
                os.path.join("Cervix", "RawData", "Training", "label", f),
                os.path.join("Cervix", "Masks", f),
            )

    # Rename final directories
    os.rename("Abdomen", "BCV15-Abdomen")
    os.rename("Cervix", "BCV15-Cervix")

    # Clean up raw data folders and zip files
    shutil.rmtree(os.path.join("BCV15-Abdomen", "RawData"), ignore_errors=True)
    os.remove(os.path.join("BCV15-Abdomen", "RawData.zip"))
    shutil.rmtree(os.path.join("BCV15-Cervix", "RawData"), ignore_errors=True)
    os.remove(os.path.join("BCV15-Cervix", "CervixRawData.zip"))

    # Move folder to dataset_dir
    folders_to_move = [
        "BCV15-Cervix",
        "BCV15-Abdomen",
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
