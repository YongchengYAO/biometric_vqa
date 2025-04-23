import os
import argparse
from biometric_vqa.utils.benchmark_planner import BiometricVQA_BenchmarkPlannerBiometry_fromSeg


# ====================================
# Dataset Info [!]
# Do not change keys in
#  - benchmark_plan
#  - labels_map
# ====================================
CLUSTER_SIZE_THRESHOLD = 200

dataset_info = {
    "dataset": "autoPET-III",
    "dataset_website": "https://autopet-iii.grand-challenge.org",
    "dataset_data": [
        "https://it-portal.med.uni-muenchen.de/autopet/Autopet_v1.1.tgz",
    ],
    "license": ["CC-BY-NC 4.0"],
    "paper": ["https://doi.org/10.1038/s41597-022-01718-3"],
}

labels_map = {"1": "tumor"}

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

benchmark_plan = {
    "dataset_info": dataset_info,
    "tasks": [
        {
            "image_modality": "CT",
            "image_description": "whole-body computed tomography (CT) scan",
            "image_folder": "Images-CT",
            "mask_folder": "Masks",
            "landmark_folder": "Landmarks-Label1",
            "landmark_figure_folder": "Landmarks-Label1-fig",
            "image_prefix": "",
            "image_suffix": "_0000.nii.gz",
            "mask_prefix": "",
            "mask_suffix": ".nii.gz",
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
        force_uint16_mask=False,
        reorient2RAS=False,
        visualization=True,
    )
    planner_segmentation.process()
