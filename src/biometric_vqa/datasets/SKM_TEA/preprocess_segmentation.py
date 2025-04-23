import os
import argparse
from biometric_vqa.utils.benchmark_planner import BiometricVQA_BenchmarkPlannerSegmentation


# ====================================
# Dataset Info [!]
# Do not change keys in
#  - benchmark_plan
#  - labels_map
# ====================================
dataset_info = {
    "dataset": "SKM-TEA",
    "dataset_website": "https://aimi.stanford.edu/datasets/skm-tea-knee-mri",
    "dataset_data": [
        "https://aimi.stanford.edu/datasets/skm-tea-knee-mri",
    ],
    "license": [""],
    "paper": ["https://openreview.net/pdf?id=YDMFgD_qJuA"],
}

labels_map = {
    "1": "patellar cartilage",
    "2": "femoral cartilage",
    "3": "medial tibial cartilage",
    "4": "lateral tibial cartilage",
    "5": "medial meniscus",
    "6": "lateral meniscus",
}

benchmark_plan = {
    "dataset_info": dataset_info,
    "tasks": [
        {
            "image_modality": "MRI",
            "image_description": "knee magnetic resonance imaging (MRI) scan",
            "image_folder": "Images-E1",
            "mask_folder": "Masks",
            "image_prefix": "",
            "image_suffix": ".nii.gz",
            "mask_prefix": "",
            "mask_suffix": ".nii.gz",
            "labels_map": labels_map,
        },
        {
            "image_modality": "MRI",
            "image_description": "knee magnetic resonance imaging (MRI) scan",
            "image_folder": "Images-E2",
            "mask_folder": "Masks",
            "image_prefix": "",
            "image_suffix": ".nii.gz",
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
    planner_segmentation = BiometricVQA_BenchmarkPlannerSegmentation(
        dataset_dir,
        benchmark_plan,
        args.dataset_name,
        seed=args.random_seed,
        split_ratio=args.split_ratio,
        force_uint16_mask=True,
        reorient2RAS=True,
    )
    planner_segmentation.process()