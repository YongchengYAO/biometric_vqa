import os
import shutil
import argparse
import glob
from huggingface_hub import snapshot_download
import nibabel as nib
from biometric_vqa.utils.preprocess_utils import move_folder


# ====================================
# Dataset Info [!]
# ====================================
# Dataset: AbdomenAtlas1.0Mini
# Website: https://github.com/MrGiovanni/AbdomenAtlas
# Data: https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas1.0Mini
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
    dest_dir = "Images-raw"
    snapshot_download(
        repo_id="AbdomenAtlas/AbdomenAtlas1.0Mini",
        repo_type="dataset",
        local_dir=dest_dir,
    )

    # Create Images and Masks directories
    os.makedirs("Images", exist_ok=True)
    os.makedirs("Masks", exist_ok=True)

    # Process each case folder
    for case_dir in glob.glob(os.path.join(dest_dir, "BDMAP*")):
        if os.path.isdir(case_dir):
            case_name = os.path.basename(case_dir)

            # Move CT images
            ct_src = os.path.join(case_dir, "ct.nii.gz")
            if os.path.exists(ct_src):
                shutil.move(ct_src, os.path.join("Images", f"{case_name}.nii.gz"))

            # Move mask files
            mask_src = os.path.join(case_dir, "combined_labels.nii.gz")
            if os.path.exists(mask_src):
                shutil.move(mask_src, os.path.join("Masks", f"{case_name}.nii.gz"))

    # Copy Nifit header of images to masks
    print("Copying Nifti headers from images to masks...")
    nifti_files = list(glob.glob(os.path.join("Images", "*.nii.gz")))
    total_files = len(nifti_files)
    for idx, img_file in enumerate(nifti_files, 1):
        base_name = os.path.basename(img_file)
        mask_file = os.path.join("Masks", base_name)
        if os.path.exists(mask_file):
            print(f"[{idx}/{total_files}] Processing file: {base_name}")
            img = nib.load(img_file)
            mask = nib.load(mask_file)
            new_mask = nib.Nifti1Image(mask.get_fdata(), img.affine, img.header)
            nib.save(new_mask, mask_file)
    print("Finished updating Nifti headers for mask files")

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
