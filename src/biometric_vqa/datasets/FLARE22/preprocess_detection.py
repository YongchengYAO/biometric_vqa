import os
import argparse
from biometric_vqa.utils.benchmark_planner import BiometricVQA_BenchmarkPlannerDetection


# ====================================
# Dataset Info [!]
# Do not change keys in
#  - benchmark_plan
#  - labels_map
# ====================================
dataset_info = {
    "dataset": "FLARE22",
    "dataset_website": "https://flare22.grand-challenge.org",
    "dataset_data": [
        "https://drive.google.com/drive/folders/1x0l-bxte46QFn5K_ZJzBxp8HsscF8v6t",
    ],
    "license": [""],
    "paper": ["https://doi.org/10.1016/S2589-7500(24)00154-7"],
}

labels_map = {
    "1": "liver",
    "2": "right kidney",
    "3": "spleen",
    "4": "pancreas",
    "5": "aorta",
    "6": "inferior vena cava (ivc)",
    "7": "right adrenal gland (rag)",
    "8": "left adrenal gland (lag)",
    "9": "gallbladder",
    "10": "esophagus",
    "11": "stomach",
    "12": "duodenum",
    "13": "left kidney",
}

benchmark_plan = {
    "dataset_info": dataset_info,
    "tasks": [
        {
            "image_modality": "CT",
            "image_description": "abdominal computed tomography (CT) scan",
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
