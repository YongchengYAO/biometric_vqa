import os
import argparse
from biometric_vqa.utils.benchmark_planner import BiometricVQA_BenchmarkPlannerDetection


# ====================================
# Dataset Info [!]
# Do not change keys in
#  - benchmark_plan
# ====================================
dataset_info = {
    "dataset": "AMOS22",
    "dataset_website": "https://amos22.grand-challenge.org",
    "dataset_data": [
        "https://zenodo.org/records/7262581",
    ],
    "license": ["CC BY 4.0"],
    "paper": ["https://doi.org/10.48550/arXiv.2206.08023"],
}

labels_map = {
    "1": "spleen",
    "2": "right kidney",
    "3": "left kidney",
    "4": "gall bladder",
    "5": "esophagus",
    "6": "liver",
    "7": "stomach",
    "8": "arota",
    "9": "postcava",
    "10": "pancreas",
    "11": "right adrenal gland",
    "12": "left adrenal gland",
    "13": "duodenum",
    "14": "bladder",
    "15": "prostate/uterus",
}

benchmark_plan = {
    "dataset_info": dataset_info,
    "tasks": [
        {
            "image_modality": "CT",
            "image_description": "abdominal computed tomography (CT) scan",
            "image_folder": "AMOS22-CT/Images",
            "mask_folder": "AMOS22-CT/Masks",
            "image_prefix": "",
            "image_suffix": ".nii.gz",
            "mask_prefix": "",
            "mask_suffix": ".nii.gz",
            "labels_map": labels_map,
        },
        {
            "image_modality": "MRI",
            "image_description": "abdominal magnetic resonance imaging (MRI) scan",
            "image_folder": "AMOS22-MRI/Images",
            "mask_folder": "AMOS22-MRI/Masks",
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
