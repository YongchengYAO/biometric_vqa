import os
import argparse
from biometric_vqa.utils.benchmark_planner import BiometricVQA_BenchmarkPlannerBiometry_fromSeg


# ====================================
# Dataset Info [!]
# Do not change keys in
#  - benchmark_plan
# ====================================
CLUSTER_SIZE_THRESHOLD = 200

dataset_info = {
    "dataset": "",
    "dataset_website": "",
    "dataset_data": [""],
    "license": [""],
    "paper": [""],
}

labels_map = {
    "1": "<segmentation label 1 (use descriptive name)>",
}
# ====================================


# ===============
# DO NOT CHANGE
# ===============
landmarks_map = {
    "P1": "most right/anterior/superior endpoint of the major axis",
    "P2": "most left/superior/inferior endpoint of the major axis",
    "P3": "most right/anterior/superior endpoint of the minor axis",
    "P4": "most left/superior/inferior endpoint of the minor axis",
}

lines_map = {
    "L-1-2": {
        "name": "marjor axis of the fitted ellipse",
        "element_keys": ["P1", "P2"],
        "element_map_name": "landmarks_map",
    },
    "L-3-4": {
        "name": "minor axis of the fitted ellipse",
        "element_keys": ["P3", "P4"],
        "element_map_name": "landmarks_map",
    },
}

angles_map = {}

biometrics_map = [
    {
        "metric_type": "distance",
        "metric_map_name": "lines_map",
        "metric_key": "L-1-2",
    },
    {
        "metric_type": "distance",
        "metric_map_name": "lines_map",
        "metric_key": "L-3-4",
    },
]
# ===============


# ------------
# Task-specific benchmark planning configuration
# ------------
# - dataset_info: Dictionary containing dataset metadata
# - tasks: List of task configurations where each task contains:
#   - image_modality: Type of medical imaging (e.g., "CT", "MRI")
#   - image_description: Description of image, used in text prompts
#   - image_folder: Directory for .nii.gz image files
#   - landmark_folder: Directory for landmark files
#   - image_prefix: Filename part before case ID for images
#   - image_suffix: Filename part after case ID for images
#   - landmark_prefix: Filename part before case ID for landmarks
#   - landmark_suffix: Filename part after case ID for landmarks
#   - landmarks_map: Dictionary mapping landmarks to their descriptions
# NOTE:
# - These keys should match the variable names:
#        "landmarks_map": landmarks_map,
#         "lines_map": lines_map,
#         "angles_map": angles_map,
#         "biometrics_map": biometrics_map,
# ------------
benchmark_plan = {
    "dataset_info": dataset_info,
    "tasks": [
        {
            "image_modality": "MRI",
            "image_description": "",
            "image_folder": "",
            "mask_folder": "",
            "image_prefix": "",
            "landmark_folder": "",
            "landmark_figure_folder": "",
            "image_suffix": "_T2.nii.gz",
            "mask_prefix": "",
            "mask_suffix": "_mask.nii.gz",
            "landmark_prefix": "",
            "landmark_suffix": ".json.gz",
            "labels_map": labels_map,
            "landmarks_map": landmarks_map,
            "lines_map": lines_map,
            "angles_map": angles_map,
            "biometrics_map": biometrics_map,
            "target_label": 1,
            "cluster_size_threshold": CLUSTER_SIZE_THRESHOLD,
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
    planner_segmentation = BiometricVQA_BenchmarkPlannerBiometry_fromSeg(
        dataset_dir,
        benchmark_plan,
        args.dataset_name,
        seed=args.random_seed,
        split_ratio=args.split_ratio,
        shrunk_bbox_scale=0.9,
        enlarged_bbox_scale=1.1,
        force_uint16_mask=False, # set to True if the mask files have not been processed
        reorient2RAS=False, # set to True if the mask and image files have not been processed
        visualization=True, # set to True to visualize the biometric annotations
    )
    planner_segmentation.process()
