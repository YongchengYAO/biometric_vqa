import os
import shutil
import argparse
import glob
import zipfile
import rarfile
import synapseclient
from biometric_vqa.utils.preprocess_utils import process_dataset_mm, move_folder


# ====================================
# Dataset Info [!]
# ====================================
# Dataset: BraTS24
# Challenge: https://www.synapse.org/Synapse:syn53708249/wiki/
# Segmentation Task:
#   Adult Glioma Post Treatment (GLI): https://www.synapse.org/Synapse:syn53708249/wiki/627500
#   Brain Metastases (MET): https://www.synapse.org/Synapse:syn53708249/wiki/627504
#   Meningioma Radiotherapy (MEN-RT): https://www.synapse.org/Synapse:syn53708249/wiki/627503
#   Pediatric Tumors (PED): https://www.synapse.org/Synapse:syn53708249/wiki/627505
# Data:
#   Adult Glioma Post Treatment (GLI): https://www.synapse.org/Synapse:syn59059776
#   Brain Metastases (MET): https://www.synapse.org/Synapse:syn59059764
#   Meningioma Radiotherapy (MEN-RT): https://www.synapse.org/Synapse:syn59059779
#   Pediatric Tumors (PED): https://www.synapse.org/Synapse:syn58894466
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

    # Process GLI dataset
    print("Downloading BraTS24-GLI dataset...")
    os.makedirs("BraTS24-GLI", exist_ok=True)
    for id in ["syn64314352", "syn60086071"]:
        file = syn.get(id, downloadLocation="BraTS24-GLI")
        print(f" - Downloaded: {file.path}")
        with zipfile.ZipFile(file.path, "r") as zip_ref:
            zip_ref.extractall("BraTS24-GLI")
    os.chdir("BraTS24-GLI")
    os.makedirs("Masks", exist_ok=True)
    for mod in ["t1c", "t1n", "t2f", "t2w"]:
        os.makedirs(f"Images-{mod}", exist_ok=True)
    # Process GLI files
    process_dataset_mm(
        ["training_data1_v2", "training_data_additional"],
        "*-seg.nii.gz",
        ["t1c", "t1n", "t2f", "t2w"],
        "-seg.nii.gz",
    )
    # Cleanup GLI
    for folder in ["training_data1_v2", "training_data_additional"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    for ext in ["tsv", "bib", "xlsx", "zip"]:
        for f in glob.glob(f"*.{ext}"):
            os.remove(f)
    os.chdir("..")

    # Process MEN-RT dataset
    print("Downloading BraTS24-MEN-RT dataset...")
    os.makedirs("BraTS24-MEN-RT", exist_ok=True)
    for id in ["syn60085033"]:
        file = syn.get(id, downloadLocation="BraTS24-MEN-RT")
        print(f" - Downloaded: {file.path}")
        with zipfile.ZipFile(file.path, "r") as zip_ref:
            zip_ref.extractall("BraTS24-MEN-RT")
    os.chdir("BraTS24-MEN-RT")
    os.makedirs("Masks", exist_ok=True)
    os.makedirs("Images-t1c", exist_ok=True)
    # Process MEN-RT files
    process_dataset_mm(
        ["BraTS-MEN-RT-Train-v2"],
        "*_gtv.nii.gz",
        ["t1c"],
        "_gtv.nii.gz",
    )
    # Cleanup MEN-RT
    if os.path.exists("BraTS-MEN-RT-Train-v2"):
        shutil.rmtree("BraTS-MEN-RT-Train-v2")
    for ext in ["tsv", "bib", "zip"]:
        for f in glob.glob(f"*.{ext}"):
            os.remove(f)
    os.chdir("..")

    # Process MET dataset
    os.makedirs("BraTS24-MET", exist_ok=True)
    # Download training data
    for id in ["syn59407686", "syn59860022", "syn61596964"]:
        file = syn.get(id, downloadLocation="BraTS24-MET")
        print(f" - Downloaded: {file.path}")
    with zipfile.ZipFile(
        os.path.join(
            "BraTS24-MET", "MICCAI-BraTS2024-MET-Challenge-TrainingData_1.zip"
        ),
        "r",
    ) as zip_ref:
        zip_ref.extractall("BraTS24-MET")
    with zipfile.ZipFile(
        os.path.join(
            "BraTS24-MET", "MICCAI-BraTS2024-MET-Challenge-TrainingData_2.zip"
        ),
        "r",
    ) as zip_ref:
        zip_ref.extractall(
            os.path.join("BraTS24-MET", "MICCAI-BraTS2024-MET-Challenge-TrainingData_2")
        )
    with zipfile.ZipFile(
        os.path.join(
            "BraTS24-MET",
            "MICCAI-BraTS2024-MET-Challenge-TrainingData_2-fixed-cases.zip",
        ),
        "r",
    ) as zip_ref:
        zip_ref.extractall("BraTS24-MET")
    syn.get("syn61929632", downloadLocation="BraTS24-MET")
    rar_path = os.path.join("BraTS24-MET", "BraTS-MET-00232-000.rar")
    extract_dir = os.path.join(
        "BraTS24-MET", "MICCAI-BraTS2024-MET-Challenge-TrainingData_2-fixed-cases"
    )
    os.makedirs(extract_dir, exist_ok=True)
    with rarfile.RarFile(rar_path) as rf:
        rf.extractall(path=extract_dir)
    os.chdir("BraTS24-MET")
    # Create directories
    os.makedirs("Masks", exist_ok=True)
    for mod in ["t1c", "t1n", "t2f", "t2w"]:
        os.makedirs(f"Images-{mod}", exist_ok=True)
    # Process main training datasets
    process_dataset_mm(
        [
            "MICCAI-BraTS2024-MET-Challenge-Training_1",
            "MICCAI-BraTS2024-MET-Challenge-TrainingData_2",
        ],
        "*-seg.nii.gz",
        ["t1c", "t1n", "t2f", "t2w"],
        "-seg.nii.gz",
    )
    # Process fixed cases with force overwrite
    process_dataset_mm(
        ["MICCAI-BraTS2024-MET-Challenge-TrainingData_2-fixed-cases"],
        "*-seg.nii.gz",
        ["t1c", "t1n", "t2f", "t2w"],
        "-seg.nii.gz",
        replace=True,  # Enable force overwrite
    )
    # Delete cases where the NIfTI image and mask headers don't match
    cases_to_remove = ["BraTS-MET-00232-000"]
    for case in cases_to_remove:
        for mod in ["t1c", "t1n", "t2f", "t2w"]:
            img_path = os.path.join(
                "BraTS24-MET", f"Images-{mod}", f"{case}-{mod}.nii.gz"
            )
            mask_path = os.path.join("BraTS24-MET", "Masks", f"{case}-seg.nii.gz")
            if os.path.exists(img_path):
                os.remove(img_path)
            if os.path.exists(mask_path):
                os.remove(mask_path)
    # Cleanup
    for folder in [
        "MICCAI-BraTS2024-MET-Challenge-TrainingData_2-fixed-cases",
        "MICCAI-BraTS2024-MET-Challenge-Training_1",
        "MICCAI-BraTS2024-MET-Challenge-TrainingData_2",
    ]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    for ext in ["tsv", "bib", "zip", "rar"]:
        for f in glob.glob(f"*.{ext}"):
            os.remove(f)
    os.chdir("..")

    # Process PED dataset
    os.makedirs("BraTS24-PED", exist_ok=True)
    # Download training data
    for id in ["syn58894928", "syn60140557"]:
        file = syn.get(id, downloadLocation="BraTS24-PED")
        print(f" - Downloaded: {file.path}")
        with zipfile.ZipFile(file.path, "r") as zip_ref:
            zip_ref.extractall("BraTS24-PED")
    os.chdir("BraTS24-PED")
    # Fix broken case
    if os.path.exists(os.path.join("BraTS-PEDs2024_Training", "BraTS-PED-00255-000")):
        shutil.rmtree(os.path.join("BraTS-PEDs2024_Training", "BraTS-PED-00255-000"))
    shutil.move("BraTS-PED-00255-000", "BraTS-PEDs2024_Training")
    # Create directories
    os.makedirs("Masks", exist_ok=True)
    for mod in ["t1c", "t1n", "t2f", "t2w"]:
        os.makedirs(f"Images-{mod}", exist_ok=True)
    # Process training data
    process_dataset_mm(
        ["BraTS-PEDs2024_Training"],
        "*-seg.nii.gz",
        ["t1c", "t1n", "t2f", "t2w"],
        "-seg.nii.gz",
    )
    # Remove any hidden files
    for hidden_file in glob.glob(
        os.path.join("BraTS24-PED", "**", "._*.nii.gz"), recursive=True
    ):
        os.remove(hidden_file)
    # Cleanup
    for folder in ["BraTS-PEDs2024_Training", "__MACOSX"]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    for ext in ["tsv", "bib", "zip"]:
        for f in glob.glob(f"*.{ext}"):
            os.remove(f)
    os.chdir("..")

    # Move folder to dataset_dir
    folders_to_move = [
        "BraTS24-GLI",
        "BraTS24-MEN-RT",
        "BraTS24-MET",
        "BraTS24-PED",
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
