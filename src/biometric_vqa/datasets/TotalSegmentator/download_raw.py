import os
import shutil
import argparse
import zipfile
import urllib.request
import nibabel as nib
import numpy as np
from pathlib import Path


# ====================================
# Dataset Info [!]
# ====================================
# Dataset: TotalSegmentator
# GitHub: https://github.com/wasserth/TotalSegmentator
# Data: CT: https://zenodo.org/records/10047292
#       MR: https://zenodo.org/records/14710732
# Preprocessed Data:
#       CT: https://huggingface.co/datasets/YongchengYAO/TotalSegmentator-CT-Lite
#       MR: https://huggingface.co/datasets/YongchengYAO/TotalSegmentator-MR-Lite
# Format: nii.gz
# ====================================

# --------------------------
# DO NOT CHANGE THE LABELS
# --------------------------
# Define the dictionary mapping index to TotalSegmentator name
labels_CT = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "pancreas",
    8: "adrenal_gland_right",
    9: "adrenal_gland_left",
    10: "lung_upper_lobe_left",
    11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right",
    13: "lung_middle_lobe_right",
    14: "lung_lower_lobe_right",
    15: "esophagus",
    16: "trachea",
    17: "thyroid_gland",
    18: "small_bowel",
    19: "duodenum",
    20: "colon",
    21: "urinary_bladder",
    22: "prostate",
    23: "kidney_cyst_left",
    24: "kidney_cyst_right",
    25: "sacrum",
    26: "vertebrae_S1",
    27: "vertebrae_L5",
    28: "vertebrae_L4",
    29: "vertebrae_L3",
    30: "vertebrae_L2",
    31: "vertebrae_L1",
    32: "vertebrae_T12",
    33: "vertebrae_T11",
    34: "vertebrae_T10",
    35: "vertebrae_T9",
    36: "vertebrae_T8",
    37: "vertebrae_T7",
    38: "vertebrae_T6",
    39: "vertebrae_T5",
    40: "vertebrae_T4",
    41: "vertebrae_T3",
    42: "vertebrae_T2",
    43: "vertebrae_T1",
    44: "vertebrae_C7",
    45: "vertebrae_C6",
    46: "vertebrae_C5",
    47: "vertebrae_C4",
    48: "vertebrae_C3",
    49: "vertebrae_C2",
    50: "vertebrae_C1",
    51: "heart",
    52: "aorta",
    53: "pulmonary_vein",
    54: "brachiocephalic_trunk",
    55: "subclavian_artery_right",
    56: "subclavian_artery_left",
    57: "common_carotid_artery_right",
    58: "common_carotid_artery_left",
    59: "brachiocephalic_vein_left",
    60: "brachiocephalic_vein_right",
    61: "atrial_appendage_left",
    62: "superior_vena_cava",
    63: "inferior_vena_cava",
    64: "portal_vein_and_splenic_vein",
    65: "iliac_artery_left",
    66: "iliac_artery_right",
    67: "iliac_vena_left",
    68: "iliac_vena_right",
    69: "humerus_left",
    70: "humerus_right",
    71: "scapula_left",
    72: "scapula_right",
    73: "clavicula_left",
    74: "clavicula_right",
    75: "femur_left",
    76: "femur_right",
    77: "hip_left",
    78: "hip_right",
    79: "spinal_cord",
    80: "gluteus_maximus_left",
    81: "gluteus_maximus_right",
    82: "gluteus_medius_left",
    83: "gluteus_medius_right",
    84: "gluteus_minimus_left",
    85: "gluteus_minimus_right",
    86: "autochthon_left",
    87: "autochthon_right",
    88: "iliopsoas_left",
    89: "iliopsoas_right",
    90: "brain",
    91: "skull",
    92: "rib_left_1",
    93: "rib_left_2",
    94: "rib_left_3",
    95: "rib_left_4",
    96: "rib_left_5",
    97: "rib_left_6",
    98: "rib_left_7",
    99: "rib_left_8",
    100: "rib_left_9",
    101: "rib_left_10",
    102: "rib_left_11",
    103: "rib_left_12",
    104: "rib_right_1",
    105: "rib_right_2",
    106: "rib_right_3",
    107: "rib_right_4",
    108: "rib_right_5",
    109: "rib_right_6",
    110: "rib_right_7",
    111: "rib_right_8",
    112: "rib_right_9",
    113: "rib_right_10",
    114: "rib_right_11",
    115: "rib_right_12",
    116: "sternum",
    117: "costal_cartilages",
}


labels_MR = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "pancreas",
    8: "adrenal_gland_right",
    9: "adrenal_gland_left",
    10: "lung_left",
    11: "lung_right",
    12: "esophagus",
    13: "small_bowel",
    14: "duodenum",
    15: "colon",
    16: "urinary_bladder",
    17: "prostate",
    18: "sacrum",
    19: "vertebrae",
    20: "intervertebral_discs",
    21: "spinal_cord",
    22: "heart",
    23: "aorta",
    24: "inferior_vena_cava",
    25: "portal_vein_and_splenic_vein",
    26: "iliac_artery_left",
    27: "iliac_artery_right",
    28: "iliac_vena_left",
    29: "iliac_vena_right",
    30: "humerus_left",
    31: "humerus_right",
    32: "scapula_left",
    33: "scapula_right",
    34: "clavicula_left",
    35: "clavicula_right",
    36: "femur_left",
    37: "femur_right",
    38: "hip_left",
    39: "hip_right",
    40: "gluteus_maximus_left",
    41: "gluteus_maximus_right",
    42: "gluteus_medius_left",
    43: "gluteus_medius_right",
    44: "gluteus_minimus_left",
    45: "gluteus_minimus_right",
    46: "autochthon_left",
    47: "autochthon_right",
    48: "iliopsoas_left",
    49: "iliopsoas_right",
    50: "brain",
}
# --------------------------


def process_data_totalsegmentator(subject_path: Path, output_dir: Path, modality: str):
    """Process a single subject folder."""
    print(f"Processing subject: {subject_path.name}")

    # Make dirs
    masks_dir = output_dir / "Masks"
    images_dir = output_dir / "Images"
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Initialize output image with zeros
    seg_dir = subject_path / "segmentations"
    if not seg_dir.exists():
        print(f"No segmentations directory found in {subject_path}")
        return

    # Get image file
    if modality == "CT":
        img_file = subject_path / "ct.nii.gz"
    elif modality == "MR":
        img_file = subject_path / "mri.nii.gz"
    else:
        print(f"Invalid modality: {modality}")
        return

    # Get reference image to copy metadata
    ref_img = nib.load(str(img_file))
    combined_array = np.zeros(ref_img.shape, dtype=np.int16)

    # Get segmentation labels
    if modality == "CT":
        seg_labels = labels_CT
    elif modality == "MR":
        seg_labels = labels_MR
    else:
        print(f"Invalid modality: {modality}")
        return

    # Process each segmentation
    for index, organ in seg_labels.items():
        seg_file = seg_dir / f"{organ}.nii.gz"
        if seg_file.exists():
            print(f"Processing {organ}")
            # Read binary mask
            mask = nib.load(str(seg_file))
            mask_array = mask.get_fdata().astype(np.int16)
            # Find overlapping regions and set them to 0 in the new mask
            overlap = (mask_array == 1) & (combined_array > 0)
            mask_array[overlap] = 0
            # Replace remaining 1s with index
            mask_array[mask_array == 1] = index
            # Add to combined mask
            combined_array += mask_array

    # Save combined mask
    output_file = masks_dir / f"{subject_path.name}.nii.gz"
    combined_nifti = nib.Nifti1Image(combined_array, ref_img.affine, ref_img.header)
    nib.save(combined_nifti, str(output_file))
    print(f"Saved combined mask to {output_file}")

    if img_file.exists():
        shutil.copy2(img_file, images_dir / f"{subject_path.name}.nii.gz")
        print(f"Copied {modality} image for {subject_path.name}")


def download_and_extract(dataset_dir, dataset_name):
    # Download files
    print(f"Downloading {dataset_name} dataset to {dataset_dir}...")

    # ====================================
    # Add download logic here [!]
    # ====================================
    # Download dataset from Zenodo
    # Download CT dataset
    print("Downloading TotalSegmentator CT dataset...")
    urllib.request.urlretrieve(
        "https://zenodo.org/records/10047292/files/Totalsegmentator_dataset_v201.zip",
        "TotalSegmentator-CT.zip"
    )
    # Download MR dataset
    print("Downloading TotalSegmentator MR dataset...")
    urllib.request.urlretrieve(
        "https://zenodo.org/records/14710732/files/TotalsegmentatorMRI_dataset_v200.zip",
        "TotalSegmentator-MR.zip"
    )

    # Extract CT dataset
    print("Extracting TotalSegmentator CT dataset...")
    with zipfile.ZipFile("TotalSegmentator-CT.zip", 'r') as zip_ref:
        zip_ref.extractall("TotalSegmentator-CT-raw")
    
    # Extract MR dataset
    print("Extracting TotalSegmentator MR dataset...")
    with zipfile.ZipFile("TotalSegmentator-MR.zip", 'r') as zip_ref:
        zip_ref.extractall("TotalSegmentator-MR-raw")

    data_dir = Path.cwd()

    # Process CT data
    ct_out_dir = data_dir / "TotalSegmentator-CT"
    ct_working_dir = data_dir / "TotalSegmentator-CT-raw"
    for item in ct_working_dir.iterdir():
        process_data_totalsegmentator(item, ct_out_dir, "CT")

    # Process MR data
    mr_out_dir = data_dir / "TotalSegmentator-MR"
    mr_working_dir = data_dir / "TotalSegmentator-MR-raw"
    for item in mr_working_dir.iterdir():
        process_data_totalsegmentator(item, mr_out_dir, "MR")

    # Clean up
    shutil.rmtree("TotalSegmentator-CT-raw")
    shutil.rmtree("TotalSegmentator-MR-raw")
    os.remove("TotalSegmentator-CT.zip")
    os.remove("TotalSegmentator-MR.zip")
    # ====================================

    print(f"Download and extraction completed for {dataset_name}")


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
