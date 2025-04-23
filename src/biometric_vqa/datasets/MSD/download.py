import os
import shutil
import argparse
import glob
import urllib.request
import tarfile
from biometric_vqa.utils.preprocess_utils import split_4d_nifti, move_folder


# ====================================
# Dataset Info [!]
# ====================================
# Dataset: MSD
# Data: http://medicaldecathlon.com/dataaws/
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
    # MSD task URLs and filenames
    tasks = {
        "BrainTumour": "Task01_BrainTumour.tar",
        "Heart": "Task02_Heart.tar",
        "Liver": "Task03_Liver.tar",
        "Hippocampus": "Task04_Hippocampus.tar",
        "Prostate": "Task05_Prostate.tar",
        "Lung": "Task06_Lung.tar",
        "Pancreas": "Task07_Pancreas.tar",
        "HepaticVessel": "Task08_HepaticVessel.tar",
        "Spleen": "Task09_Spleen.tar",
        "Colon": "Task10_Colon.tar",
    }

    base_url = "https://msd-for-monai.s3-us-west-2.amazonaws.com"

    # Download all tasks
    print("Downloading MSD task datasets...")
    for task_name, filename in tasks.items():
        url = f"{base_url}/{filename}"
        print(f"Downloading {filename} from {url}...")
        urllib.request.urlretrieve(url, filename)
        print(f"Download complete: {filename}")

    # Extract tar files
    print("Extracting downloaded files...")
    for tar_file in glob.glob("*.tar"):
        print(f"Extracting {tar_file}...")
        with tarfile.open(tar_file, "r") as tar:
            tar.extractall()
        os.remove(tar_file)
        print(f"{tar_file} extracted")

    # Rename task directories
    for task_dir in glob.glob("Task*"):
        new_name = f"MSD-{task_dir[7:]}"  # Remove "Task??_" prefix
        os.rename(task_dir, new_name)

    # Rename/remove subdirectories in MSD folders
    for msd_dir in glob.glob("MSD-*"):
        if not os.path.isdir(msd_dir):
            continue
        msd_dir = msd_dir.rstrip("/\\")
        # Rename images and labels directories
        os.rename(os.path.join(msd_dir, "imagesTr"), os.path.join(msd_dir, "Images"))
        os.rename(os.path.join(msd_dir, "labelsTr"), os.path.join(msd_dir, "Masks"))
        # Remove test images directory if it exists
        test_dir = os.path.join(msd_dir, "imagesTs")
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

    # Split 4D Nifti files in the MSD-BrainTumour dataset
    split_4d_nifti(os.path.join("MSD-BrainTumour", "Images"), "MSD-BrainTumour")
    shutil.rmtree(os.path.join("MSD-BrainTumour", "Images"))
    # Rename numbered image folders to modality names in MSD-BrainTumour
    modalities = ["FLAIR", "T1w", "T1gd", "T2w"]
    for i, modality in enumerate(modalities, 1):
        os.rename(
            os.path.join("MSD-BrainTumour", f"Images-{i}"),
            os.path.join("MSD-BrainTumour", f"Images-{modality}"),
        )

    # Split 4D Nifti files in the MSD-Prostate dataset
    split_4d_nifti(os.path.join("MSD-Prostate", "Images"), "MSD-Prostate")
    shutil.rmtree(os.path.join("MSD-Prostate", "Images"))
    # Rename numbered image folders to modality names in MSD-Prostate
    modalities = ["T2w", "ADC"]
    for i, modality in enumerate(modalities, 1):
        os.rename(
            os.path.join("MSD-Prostate", f"Images-{i}"),
            os.path.join("MSD-Prostate", f"Images-{modality}"),
        )

    # Clean up macOS and hidden files
    for msd_dir in glob.glob("MSD-*"):
        if not os.path.isdir(msd_dir):
            continue
        # Remove macOS metadata files
        for pattern in ["._*", "._.DS_Store"]:
            for file_path in glob.glob(
                os.path.join(msd_dir, "**", pattern), recursive=True
            ):
                os.remove(file_path)

        # Remove specific hidden files
        hidden_files = ["._dataset.json", "._imagesTr", "._imagesTs", "._labelsTr"]
        for hf in hidden_files:
            hf_path = os.path.join(msd_dir, hf)
            if os.path.exists(hf_path):
                os.remove(hf_path)

    # Clean up root hidden files
    for msd_dir in glob.glob("MSD-*"):
        json_file = os.path.join(msd_dir, "dataset.json")
        if os.path.exists(json_file):
            os.remove(json_file)
    for f in glob.glob("._*"):
        os.remove(f)

    # Move folder to dataset_dir
    folders_to_move = [
        "MSD-BrainTumour",
        "MSD-Colon",
        "MSD-Heart",
        "MSD-HepaticVessel",
        "MSD-Hippocampus",
        "MSD-Liver",
        "MSD-Lung",
        "MSD-Pancreas",
        "MSD-Prostate",
        "MSD-Spleen",
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
