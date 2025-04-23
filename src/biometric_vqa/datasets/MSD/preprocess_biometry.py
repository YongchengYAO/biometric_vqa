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
    "dataset": "MSD",
    "dataset_website": "http://medicaldecathlon.com/dataaws/",
    "dataset_data": [
        "http://medicaldecathlon.com/dataaws/",
    ],
    "license": ["CC BY-SA 4.0"],
    "paper": ["https://doi.org/10.1038/s41467-022-30695-9"],
}

labels_map_BrainTumour = {
    "1": "edema",
    "2": "non-enhancing tumor",
    "3": "enhancing tumour",
}

labels_map_Colon = {"1": "colon cancer primaries"}

labels_map_HepaticVessel = {"1": "liver/hepatic vessel", "2": "liver/hepatic tumour"}

labels_map_Liver = {"1": "liver", "2": "liver cancer"}

labels_map_Lung = {"1": "lung cancer"}

labels_map_Pancreas = {"1": "pancreas", "2": "pancreas cancer"}
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


benchmark_plan = {
    "dataset_info": dataset_info,
    "tasks": [
        {
            "image_modality": "MRI",
            "image_description": "gadolinium-enhanced T1-weighted brain magnetic resonance imaging (MRI) scan",
            "image_folder": "MSD-BrainTumour/Images-T1gd",
            "mask_folder": "MSD-BrainTumour/Masks",
            "landmark_folder": "MSD-BrainTumour/Landmarks-Label1",
            "landmark_figure_folder": "MSD-BrainTumour/Landmarks-Label1-fig",
            "image_prefix": "",
            "image_suffix": ".nii.gz",
            "mask_prefix": "",
            "mask_suffix": ".nii.gz",
            "landmark_prefix": "",
            "landmark_suffix": ".json.gz",
            "labels_map": labels_map_BrainTumour,
            "landmarks_map": landmarks_map,
            "lines_map": lines_map,
            "angles_map": angles_map,
            "biometrics_map": biometrics_map,
            "target_label": 1,
            "cluster_size_threshold": CLUSTER_SIZE_THRESHOLD,
        },
        {
            "image_modality": "MRI",
            "image_description": "gadolinium-enhanced T1-weighted brain magnetic resonance imaging (MRI) scan",
            "image_folder": "MSD-BrainTumour/Images-T1gd",
            "mask_folder": "MSD-BrainTumour/Masks",
            "landmark_folder": "MSD-BrainTumour/Landmarks-Label2",
            "landmark_figure_folder": "MSD-BrainTumour/Landmarks-Label2-fig",
            "image_prefix": "",
            "image_suffix": ".nii.gz",
            "mask_prefix": "",
            "mask_suffix": ".nii.gz",
            "landmark_prefix": "",
            "landmark_suffix": ".json.gz",
            "labels_map": labels_map_BrainTumour,
            "landmarks_map": landmarks_map,
            "lines_map": lines_map,
            "angles_map": angles_map,
            "biometrics_map": biometrics_map,
            "target_label": 2,
            "cluster_size_threshold": CLUSTER_SIZE_THRESHOLD,
        },
        {
            "image_modality": "MRI",
            "image_description": "gadolinium-enhanced T1-weighted brain magnetic resonance imaging (MRI) scan",
            "image_folder": "MSD-BrainTumour/Images-T1gd",
            "mask_folder": "MSD-BrainTumour/Masks",
            "landmark_folder": "MSD-BrainTumour/Landmarks-Label3",
            "landmark_figure_folder": "MSD-BrainTumour/Landmarks-Label3-fig",
            "image_prefix": "",
            "image_suffix": ".nii.gz",
            "mask_prefix": "",
            "mask_suffix": ".nii.gz",
            "landmark_prefix": "",
            "landmark_suffix": ".json.gz",
            "labels_map": labels_map_BrainTumour,
            "landmarks_map": landmarks_map,
            "lines_map": lines_map,
            "angles_map": angles_map,
            "biometrics_map": biometrics_map,
            "target_label": 3,
            "cluster_size_threshold": CLUSTER_SIZE_THRESHOLD,
        },
        {
            "image_modality": "CT",
            "image_description": "abdominal computed tomography (CT) scan",
            "image_folder": "MSD-Colon/Images",
            "mask_folder": "MSD-Colon/Masks",
            "landmark_folder": "MSD-Colon/Landmarks-Label1",
            "landmark_figure_folder": "MSD-Colon/Landmarks-Label1-fig",
            "image_prefix": "",
            "image_suffix": ".nii.gz",
            "mask_prefix": "",
            "mask_suffix": ".nii.gz",
            "landmark_prefix": "",
            "landmark_suffix": ".json.gz",
            "labels_map": labels_map_Colon,
            "landmarks_map": landmarks_map,
            "lines_map": lines_map,
            "angles_map": angles_map,
            "biometrics_map": biometrics_map,
            "target_label": 1,
            "cluster_size_threshold": CLUSTER_SIZE_THRESHOLD,
        },
        {
            "image_modality": "CT",
            "image_description": "abdominal computed tomography (CT) scan",
            "image_folder": "MSD-HepaticVessel/Images",
            "mask_folder": "MSD-HepaticVessel/Masks",
            "landmark_folder": "MSD-HepaticVessel/Landmarks-Label2",
            "landmark_figure_folder": "MSD-HepaticVessel/Landmarks-Label2-fig",
            "image_prefix": "",
            "image_suffix": ".nii.gz",
            "mask_prefix": "",
            "mask_suffix": ".nii.gz",
            "landmark_prefix": "",
            "landmark_suffix": ".json.gz",
            "labels_map": labels_map_HepaticVessel,
            "landmarks_map": landmarks_map,
            "lines_map": lines_map,
            "angles_map": angles_map,
            "biometrics_map": biometrics_map,
            "target_label": 2,
            "cluster_size_threshold": CLUSTER_SIZE_THRESHOLD,
        },
        {
            "image_modality": "CT",
            "image_description": "abdominal computed tomography (CT) scan",
            "image_folder": "MSD-Liver/Images",
            "mask_folder": "MSD-Liver/Masks",
            "landmark_folder": "MSD-Liver/Landmarks-Label2",
            "landmark_figure_folder": "MSD-Liver/Landmarks-Label2-fig",
            "image_prefix": "",
            "image_suffix": ".nii.gz",
            "mask_prefix": "",
            "mask_suffix": ".nii.gz",
            "landmark_prefix": "",
            "landmark_suffix": ".json.gz",
            "labels_map": labels_map_Liver,
            "landmarks_map": landmarks_map,
            "lines_map": lines_map,
            "angles_map": angles_map,
            "biometrics_map": biometrics_map,
            "target_label": 2,
            "cluster_size_threshold": CLUSTER_SIZE_THRESHOLD,
        },
        {
            "image_modality": "CT",
            "image_description": "chest computed tomography (CT) scan",
            "image_folder": "MSD-Lung/Images",
            "mask_folder": "MSD-Lung/Masks",
            "landmark_folder": "MSD-Lung/Landmarks-Label1",
            "landmark_figure_folder": "MSD-Lung/Landmarks-Label1-fig",
            "image_prefix": "",
            "image_suffix": ".nii.gz",
            "mask_prefix": "",
            "mask_suffix": ".nii.gz",
            "landmark_prefix": "",
            "landmark_suffix": ".json.gz",
            "labels_map": labels_map_Lung,
            "landmarks_map": landmarks_map,
            "lines_map": lines_map,
            "angles_map": angles_map,
            "biometrics_map": biometrics_map,
            "target_label": 1,
            "cluster_size_threshold": CLUSTER_SIZE_THRESHOLD,
        },
        {
            "image_modality": "CT",
            "image_description": "abdominal computed tomography (CT) scan",
            "image_folder": "MSD-Pancreas/Images",
            "mask_folder": "MSD-Pancreas/Masks",
            "landmark_folder": "MSD-Pancreas/Landmarks-Label2",
            "landmark_figure_folder": "MSD-Pancreas/Landmarks-Label2-fig",
            "image_prefix": "",
            "image_suffix": ".nii.gz",
            "mask_prefix": "",
            "mask_suffix": ".nii.gz",
            "landmark_prefix": "",
            "landmark_suffix": ".json.gz",
            "labels_map": labels_map_Pancreas,
            "landmarks_map": landmarks_map,
            "lines_map": lines_map,
            "angles_map": angles_map,
            "biometrics_map": biometrics_map,
            "target_label": 2,
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
