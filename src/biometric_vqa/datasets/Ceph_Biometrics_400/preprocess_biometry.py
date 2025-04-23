import os
import argparse
from biometric_vqa.utils.benchmark_planner import BiometricVQA_BenchmarkPlannerBiometry


# ====================================
# Dataset Info [!]
# Do not change keys in
#  - benchmark_plan
# Do not change the dictionray names
#  - dataset_info, landmarks_map, lines_map, angles_map, biometrics_map
# ====================================
dataset_info = {
    "dataset": "Ceph-Biometrics-400",
    "dataset_website": "",
    "dataset_data": [
        "https://figshare.com/s/37ec464af8e81ae6ebbf",
        "https://huggingface.co/datasets/YongchengYAO/Ceph-Biometrics-400",
    ],
    "license": ["N/A", "CC BY-NC 4.0"],
    "paper": ["https://doi.org/10.1038/srep33581"],
}

landmarks_map = {
    "P1": "sella",
    "P2": "nasion",
    "P3": "orbitale",
    "P4": "porion",
    "P5": "subspinale",
    "P6": "supramentale",
    "P7": "pogonion",
    "P8": "menton",
    "P9": "gnathion",
    "P10": "gonion",
    "P11": "incision inferius",
    "P12": "incision superius",
    "P13": "upper lip",
    "P14": "lower lip",
    "P15": "subnasale",
    "P16": "soft tissue pogonion",
    "P17": "posterior nasal spine",
    "P18": "anterior nasal spine",
    "P19": "articulare",
}

lines_map = {
    "L-1-2": {
        "name":"",
        "element_keys": ["P1", "P2"],
        "element_map_name": "landmarks_map",
    },
    "L-1-10": {
        "name":"",
        "element_keys": ["P1", "P10"],
        "element_map_name": "landmarks_map",
    },
    "L-2-5": {
        "name":"",
        "element_keys": ["P2", "P5"],
        "element_map_name": "landmarks_map",
    },
    "L-2-6": {
        "name":"",
        "element_keys": ["P2", "P6"],
        "element_map_name": "landmarks_map",
    },
    "L-2-7": {
        "name":"",
        "element_keys": ["P2", "P7"],
        "element_map_name": "landmarks_map",
    },
    "L-2-8": {
        "name":"",
        "element_keys": ["P2", "P8"],
        "element_map_name": "landmarks_map",
    },
    "L-3-4": {
        "name":"",
        "element_keys": ["P3", "P4"],
        "element_map_name": "landmarks_map",
    },
    "L-5-6": {
        "name":"",
        "element_keys": ["P5", "P6"],
        "element_map_name": "landmarks_map",
    },
    "L-8-10": {
        "name":"",
        "element_keys": ["P8", "P10"],
        "element_map_name": "landmarks_map",
    },
    "L-9-10": {
        "name":"",
        "element_keys": ["P9", "P10"],
        "element_map_name": "landmarks_map",
    },
    "L-17-18": {
        "name":"",
        "element_keys": ["P17", "P18"],
        "element_map_name": "landmarks_map",
    },
}

angles_map = {
    "A-L_2_5-L_2_6": {
        "name":"",
        "element_keys": ["L-2-5", "L-2-6"],
        "element_map_name": "lines_map",
    },
    "A-L_1_2-L_2_6": {
        "name":"",
        "element_keys": ["L-1-2", "L-2-6"],
        "element_map_name": "lines_map",
    },
    "A-L_1_2-L_2_5": {
        "name":"",
        "element_keys": ["L-1-2", "L-2-5"],
        "element_map_name": "lines_map",
    },
    "A-L_5_6-L_8_10": {
        "name":"",
        "element_keys": ["L-5-6", "L-8-10"],
        "element_map_name": "lines_map",
    },
    "A-L_3_4-L_2_7": {
        "name":"",
        "element_keys": ["L-3-4", "L-2-7"],
        "element_map_name": "lines_map",
    },
    "A-L_3_4-L_17_18": {
        "name":"",
        "element_keys": ["L-3-4", "L-17-18"],
        "element_map_name": "lines_map",
    },
    "A-L_2_7-L_5_6": {
        "name":"",
        "element_keys": ["L-2-7", "L-5-6"],
        "element_map_name": "lines_map",
    },
    "A-L_1_2-L_9_10": {
        "name":"",
        "element_keys": ["L-1-2", "L-9-10"],
        "element_map_name": "lines_map",
    },
}

biometrics_map = [
    {
        "metric_type": "angle",
        "metric_map_name": "angles_map",
        "metric_key": "A-L_2_5-L_2_6",
        "slice_dim": 0,
    },
    {
        "metric_type": "angle",
        "metric_map_name": "angles_map",
        "metric_key": "A-L_1_2-L_2_6",
        "slice_dim": 0,
    },
    {
        "metric_type": "angle",
        "metric_map_name": "angles_map",
        "metric_key": "A-L_1_2-L_2_5",
        "slice_dim": 0,
    },
    {
        "metric_type": "angle",
        "metric_map_name": "angles_map",
        "metric_key": "A-L_5_6-L_8_10",
        "slice_dim": 0,
    },
    {
        "metric_type": "angle",
        "metric_map_name": "angles_map",
        "metric_key": "A-L_3_4-L_2_7",
        "slice_dim": 0,
    },
    {
        "metric_type": "angle",
        "metric_map_name": "angles_map",
        "metric_key": "A-L_3_4-L_17_18",
        "slice_dim": 0,
    },
    {
        "metric_type": "angle",
        "metric_map_name": "angles_map",
        "metric_key": "A-L_2_7-L_5_6",
        "slice_dim": 0,
    },
    {
        "metric_type": "angle",
        "metric_map_name": "angles_map",
        "metric_key": "A-L_1_2-L_9_10",
        "slice_dim": 0,
    },
    {
        "metric_type": "distance",
        "metric_map_name": "lines_map",
        "metric_key": "L-1-10",
        "slice_dim": 0,
    },
    {
        "metric_type": "distance",
        "metric_map_name": "lines_map",
        "metric_key": "L-2-8",
        "slice_dim": 0,
    },
]


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
            "image_modality": "X Ray",
            "image_description": "cephalogram (head and neck X-ray)",
            "image_folder": "Images",
            "landmark_folder": "Landmarks",
            "image_prefix": "",
            "image_suffix": ".nii.gz",
            "landmark_prefix": "",
            "landmark_suffix": ".json.gz",
            "landmarks_map": landmarks_map,
            "lines_map": lines_map,
            "angles_map": angles_map,
            "biometrics_map": biometrics_map,
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
    planner_segmentation = BiometricVQA_BenchmarkPlannerBiometry(
        dataset_dir,
        benchmark_plan,
        args.dataset_name,
        seed=args.random_seed,
        split_ratio=args.split_ratio,
    )
    planner_segmentation.process()
