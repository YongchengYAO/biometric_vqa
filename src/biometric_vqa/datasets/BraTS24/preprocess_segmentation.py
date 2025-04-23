import os
import argparse
from biometric_vqa.utils.benchmark_planner import BiometricVQA_BenchmarkPlannerSegmentation


# ====================================
# Dataset Info [!]
# Do not change keys in
#  - benchmark_plan
# ====================================
dataset_info = {
    "dataset": "",
    "dataset_website": "https://www.synapse.org/Synapse:syn53708249/wiki/",
    "dataset_data": [
        "https://www.synapse.org/Synapse:syn59059776",  # GLI
        "https://www.synapse.org/Synapse:syn59059764",  # MET
        "https://www.synapse.org/Synapse:syn59059779",  # MEN-RT
        "https://www.synapse.org/Synapse:syn58894466",  # PED
    ],
    "license": [""],
    "paper": [
        "https://doi.org/10.48550/arXiv.2405.18368",  # GLI
        "",  # MET
        "https://doi.org/10.48550/arXiv.2405.18383",  # MEN-RT
        "https://doi.org/10.48550/arXiv.2404.15009",  # PED
    ],
}

labels_map_GLI = {
    "1": "non-enhancing tumor core",
    "2": "surrounding non-enhancing flair hyperintensity",
    "3": "enhancing tumor tissue",
    "4": "resection cavity",
}

labels_map_MET = {
    "1": "non-enhancing tumor core",
    "2": "surrounding non-enhancing flair hyperintensity",
    "3": "enhancing tumor tissue",
}

labels_map_MEN_RT = {"1": "gross tumor volume"}

labels_map_PED = {
    "1": "enhancing tumor",
    "2": "non-enhancing tumor",
    "3": "cystic component",
    "4": "peritumoral edema",
}

benchmark_plan = {
    "dataset_info": dataset_info,
    "tasks": [
        {
            "image_modality": "MRI",
            "image_description": "gadolinium-enhanced T1-weighted brain magnetic resonance imaging (MRI) scan",
            "image_folder": "BraTS24-GLI/Images-t1c",
            "mask_folder": "BraTS24-GLI/Masks",
            "image_prefix": "",
            "image_suffix": "-t1c.nii.gz",
            "mask_prefix": "",
            "mask_suffix": "-seg.nii.gz",
            "labels_map": labels_map_GLI,
        },
        {
            "image_modality": "MRI",
            "image_description": "non-contrast T1-weighted brain magnetic resonance imaging (MRI) scan",
            "image_folder": "BraTS24-GLI/Images-t1n",
            "mask_folder": "BraTS24-GLI/Masks",
            "image_prefix": "",
            "image_suffix": "-t1n.nii.gz",
            "mask_prefix": "",
            "mask_suffix": "-seg.nii.gz",
            "labels_map": labels_map_GLI,
        },
        {
            "image_modality": "MRI",
            "image_description": "T2 Fluid Attenuated Inversion Recovery (FLAIR) brain magnetic resonance imaging (MRI) scan",
            "image_folder": "BraTS24-GLI/Images-t2f",
            "mask_folder": "BraTS24-GLI/Masks",
            "image_prefix": "",
            "image_suffix": "-t2f.nii.gz",
            "mask_prefix": "",
            "mask_suffix": "-seg.nii.gz",
            "labels_map": labels_map_GLI,
        },
        {
            "image_modality": "MRI",
            "image_description": "T2-weighted brain magnetic resonance imaging (MRI) scan",
            "image_folder": "BraTS24-GLI/Images-t2w",
            "mask_folder": "BraTS24-GLI/Masks",
            "image_prefix": "",
            "image_suffix": "-t2w.nii.gz",
            "mask_prefix": "",
            "mask_suffix": "-seg.nii.gz",
            "labels_map": labels_map_GLI,
        },
        {
            "image_modality": "MRI",
            "image_description": "contrast enhanced T1-weighted brain magnetic resonance imaging (MRI) scan",
            "image_folder": "BraTS24-MEN-RT/Images-t1c",
            "mask_folder": "BraTS24-MEN-RT/Masks",
            "image_prefix": "",
            "image_suffix": "_t1c.nii.gz",
            "mask_prefix": "",
            "mask_suffix": "_gtv.nii.gz",
            "labels_map": labels_map_MEN_RT,
        },
        {
            "image_modality": "MRI",
            "image_description": "contrast enhanced T1-weighted brain magnetic resonance imaging (MRI) scan",
            "image_folder": "BraTS24-MET/Images-t1c",
            "mask_folder": "BraTS24-MET/Masks",
            "image_prefix": "",
            "image_suffix": "-t1c.nii.gz",
            "mask_prefix": "",
            "mask_suffix": "-seg.nii.gz",
            "labels_map": labels_map_MET,
        },
        {
            "image_modality": "MRI",
            "image_description": "non-contrast T1-weighted brain magnetic resonance imaging (MRI) scan",
            "image_folder": "BraTS24-MET/Images-t1n",
            "mask_folder": "BraTS24-MET/Masks",
            "image_prefix": "",
            "image_suffix": "-t1n.nii.gz",
            "mask_prefix": "",
            "mask_suffix": "-seg.nii.gz",
            "labels_map": labels_map_MET,
        },
        {
            "image_modality": "MRI",
            "image_description": "T2 Fluid Attenuated Inversion Recovery (FLAIR) brain magnetic resonance imaging (MRI) scan",
            "image_folder": "BraTS24-MET/Images-t2f",
            "mask_folder": "BraTS24-MET/Masks",
            "image_prefix": "",
            "image_suffix": "-t2f.nii.gz",
            "mask_prefix": "",
            "mask_suffix": "-seg.nii.gz",
            "labels_map": labels_map_MET,
        },
        {
            "image_modality": "MRI",
            "image_description": "T2-weighted brain magnetic resonance imaging (MRI) scan",
            "image_folder": "BraTS24-MET/Images-t2w",
            "mask_folder": "BraTS24-MET/Masks",
            "image_prefix": "",
            "image_suffix": "-t2w.nii.gz",
            "mask_prefix": "",
            "mask_suffix": "-seg.nii.gz",
            "labels_map": labels_map_MET,
        },
        {
            "image_modality": "MRI",
            "image_description": "contrast enhanced T1-weighted brain magnetic resonance imaging (MRI) scan",
            "image_folder": "BraTS24-PED/Images-t1c",
            "mask_folder": "BraTS24-PED/Masks",
            "image_prefix": "",
            "image_suffix": "-t1c.nii.gz",
            "mask_prefix": "",
            "mask_suffix": "-seg.nii.gz",
            "labels_map": labels_map_PED,
        },
        {
            "image_modality": "MRI",
            "image_description": "non-contrast T1-weighted brain magnetic resonance imaging (MRI) scan",
            "image_folder": "BraTS24-PED/Images-t1n",
            "mask_folder": "BraTS24-PED/Masks",
            "image_prefix": "",
            "image_suffix": "-t1n.nii.gz",
            "mask_prefix": "",
            "mask_suffix": "-seg.nii.gz",
            "labels_map": labels_map_PED,
        },
        {
            "image_modality": "MRI",
            "image_description": "T2 Fluid Attenuated Inversion Recovery (FLAIR) brain magnetic resonance imaging (MRI) scan",
            "image_folder": "BraTS24-PED/Images-t2f",
            "mask_folder": "BraTS24-PED/Masks",
            "image_prefix": "",
            "image_suffix": "-t2f.nii.gz",
            "mask_prefix": "",
            "mask_suffix": "-seg.nii.gz",
            "labels_map": labels_map_PED,
        },
        {
            "image_modality": "MRI",
            "image_description": "T2-weighted brain magnetic resonance imaging (MRI) scan",
            "image_folder": "BraTS24-PED/Images-t2w",
            "mask_folder": "BraTS24-PED/Masks",
            "image_prefix": "",
            "image_suffix": "-t2w.nii.gz",
            "mask_prefix": "",
            "mask_suffix": "-seg.nii.gz",
            "labels_map": labels_map_PED,
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
