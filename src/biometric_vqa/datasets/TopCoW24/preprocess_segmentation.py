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
    "dataset": "TopCoW24",
    "dataset_website": "https://topcow24.grand-challenge.org",
    "dataset_data": [
        "https://drive.switch.ch/index.php/s/rkqOO3adjmJVlMz",
    ],
    "license": [""],
    "paper": ["https://doi.org/10.48550/arXiv.2312.17670"],
}

labels_map = {
    "1": "basilar artery",
    "2": "right posterior cerebral artery",
    "3": "left posterior cerebral artery",
    "4": "right internal carotid arterr",
    "5": "right middle cerebral artery",
    "6": "left internal carotid artery",
    "7": "left middle cerebral artery",
    "8": "right posterior communicating artery",
    "9": "left posterior communicating artery",
    "10": "anterior communicating artery",
    "11": "right anterior cerebral artery",
    "12": "left anterior cerebral artery",
    "15": "third a2 segment",
}

benchmark_plan = {
    "dataset_info": dataset_info,
    "tasks": [
        {
            "image_modality": "MRI",
            "image_description": "Time of Flight Magnetic Resonance Angiography (TOF-MRA) scan",
            "image_folder": "TopCoW24-MR/Images",
            "mask_folder": "TopCoW24-MR/Masks",
            "image_prefix": "",
            "image_suffix": "_0000.nii.gz",
            "mask_prefix": "",
            "mask_suffix": ".nii.gz",
            "labels_map": labels_map,
        },
        {
            "image_modality": "CT",
            "image_description": "Computed Tomography Angiography (CTA) scan",
            "image_folder": "TopCoW24-CT/Images",
            "mask_folder": "TopCoW24-CT/Masks",
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
