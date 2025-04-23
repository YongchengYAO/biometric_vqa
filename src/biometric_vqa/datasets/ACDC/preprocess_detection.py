import os
import argparse
from biometric_vqa.utils.benchmark_planner import BiometricVQA_BenchmarkPlannerDetection

# ====================================
# Dataset Info [!]
# Do not change keys in
#  - benchmark_plan
# ====================================
dataset_info = {
    "dataset": "ACDC",
    "dataset_website": " https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html",
    "dataset_data": [
        "https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb",
    ],
    "license": ["CC BY-NC-SA 4.0"],
    "paper": ["https://doi.org/10.1109/TMI.2018.2837502"],
}

labels_map = {
    "1": "right ventricular cavity",
    "2": "myocardium",
    "3": "left ventricular cavity",
}

benchmark_plan = {
    "dataset_info": dataset_info,
    "tasks": [
        {
            "image_modality": "MRI",
            "image_description": "cardiac magnetic resonance imaging (MRI) scan",
            "image_folder": "Images",
            "mask_folder": "Masks",
            "image_prefix": "",
            "image_suffix": ".nii.gz",
            "mask_prefix": "",
            "mask_suffix": "_gt.nii.gz",
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
