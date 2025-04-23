import os
import argparse
from biometric_vqa.utils.benchmark_planner import BiometricVQA_BenchmarkPlannerDetection


# ====================================
# Dataset Info [!]
# Do not change keys in
#  - benchmark_plan
# ====================================
dataset_info = {
    "dataset": "",
    "dataset_website": "",
    "dataset_data": [""],
    "license": [""],
    "paper": [""],
}

labels_map = {}

# ------------
# Task-specific benchmark planning configuration
# ------------
# - dataset_info: Dictionary containing dataset metadata
# - tasks: List of task configurations where each task contains:
#   - image_modality: Type of medical imaging (e.g., "CT", "MRI")
#   - image_description: Description of image, used in text prompts
#   - image_folder: Directory for .nii.gz image files
#   - mask_folder: Directory for mask files
#   - image_prefix: Filename part before case ID for images
#   - image_suffix: Filename part after case ID for images
#   - mask_prefix: Filename part before case ID for masks
#   - mask_suffix: Filename part after case ID for masks
#   - labels_map: Dictionary mapping mask values to class labels
# ------------
benchmark_plan = {
    "dataset_info": dataset_info,
    "tasks": [
        {
            "image_modality": "",
            "image_description": "",
            "image_folder": "Images",
            "mask_folder": "Masks",
            "image_prefix": "",
            "image_suffix": "_0000.nii.gz",
            "mask_prefix": "",
            "mask_suffix": ".nii.gz",
            "labels_map": labels_map,
        },
    ],
}
# ====================================


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Download and extract dataset")
    parser.add_argument(
        "-d",
        "--dir_datasets_data",
        type=str,
        help="Directory path where datasets will be stored",
        required=True,
    )
    parser.add_argument(
        "-n",
        "--dataset_name",
        type=str,
        help="Name of the dataset",
        required=True,
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=1024,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.7,
        help="Train/test split ratio (0-1)",
    )
    args = parser.parse_args()

    # Create dataset directory
    dataset_dir = os.path.join(args.dir_datasets_data, args.dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Change to dataset directory
    os.chdir(dataset_dir)

    # Process dataset for segmentation task
    planner_segmentation = BiometricVQA_BenchmarkPlannerDetection(
        dataset_dir,
        benchmark_plan,
        args.dataset_name,
        seed=args.random_seed,
        split_ratio=args.split_ratio,
        force_uint16_mask=False,
        reorient2RAS=False,
    )
    planner_segmentation.process()
