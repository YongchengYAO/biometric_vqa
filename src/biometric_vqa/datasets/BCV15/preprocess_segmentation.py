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
    "dataset": "BCV15",
    "dataset_website": "https://www.synapse.org/Synapse:syn3193805/wiki/89480",
    "dataset_data": [
        "https://www.synapse.org/Synapse:syn3193805/wiki/217789",  # Abdomen data
        "https://www.synapse.org/Synapse:syn3193805/wiki/217790",  # Cervix data
    ],
    "license": [""],
    "paper": [""],
}

labels_map_Abdomen = {
    "1": "spleen",
    "2": "right kidney",
    "3": "left kidney",
    "4": "gallbladder",
    "5": "esophagus",
    "6": "liver",
    "7": "stomach",
    "8": "aorta",
    "9": "inferior vena cava",
    "10": "portal vein and splenic vein",
    "11": "pancreas",
    "12": "right adrenal gland",
    "13": "left adrenal gland",
}

labels_map_Cervix = {
    "1": "bladder",
    "2": "uterus",
    "3": "rectum",
    "4": "small bowel",
}

benchmark_plan = {
    "dataset_info": dataset_info,
    "tasks": [
        {
            "image_modality": "CT",
            "image_description": "abdominal computed tomography (CT) scan",
            "image_folder": "BCV15-Abdomen/Images",
            "mask_folder": "BCV15-Abdomen/Masks",
            "image_prefix": "img",
            "image_suffix": ".nii.gz",
            "mask_prefix": "label",
            "mask_suffix": ".nii.gz",
            "labels_map": labels_map_Abdomen,
        },
        {
            "image_modality": "CT",
            "image_description": "abdominal computed tomography (CT) scan",
            "image_folder": "BCV15-Cervix/Images",
            "mask_folder": "BCV15-Cervix/Masks",
            "image_prefix": "",
            "image_suffix": "-Image.nii.gz",
            "mask_prefix": "",
            "mask_suffix": "-Mask.nii.gz",
            "labels_map": labels_map_Cervix,
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
