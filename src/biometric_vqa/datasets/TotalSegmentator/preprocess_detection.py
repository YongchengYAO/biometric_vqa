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
    "dataset": "TotalSegmentator",
    "dataset_website": "https://github.com/wasserth/TotalSegmentator",
    "dataset_data": [
        "https://zenodo.org/records/10047292",  # CT
        "https://zenodo.org/records/14710732",  # MR
    ],
    "license": ["CC BY 4.0", "CC BY-NC-SA 2.0"],
    "paper": ["https://doi.org/10.1148/ryai.230024"],
}

labels_map_CT = {
    "1": "spleen",
    "2": "right kidney",
    "3": "left kidney",
    "4": "gallbladder",
    "5": "liver",
    "6": "stomach",
    "7": "pancreas",
    "8": "right adrenal gland",
    "9": "left adrenal gland",
    "10": "left lung upper lobe",
    "11": "left lung lower lobe",
    "12": "right lung upper lobe",
    "13": "right lung middle lobe",
    "14": "right lung lower lobe",
    "15": "esophagus",
    "16": "trachea",
    "17": "thyroid gland",
    "18": "small bowel",
    "19": "duodenum",
    "20": "colon",
    "21": "urinary bladder",
    "22": "prostate",
    "23": "left kidney cyst",
    "24": "right kidney cyst",
    "25": "sacrum",
    "26": "vertebra S1",
    "27": "vertebra L5",
    "28": "vertebra L4",
    "29": "vertebra L3",
    "30": "vertebra L2",
    "31": "vertebra L1",
    "32": "vertebra T12",
    "33": "vertebra T11",
    "34": "vertebra T10",
    "35": "vertebra T9",
    "36": "vertebra T8",
    "37": "vertebra T7",
    "38": "vertebra T6",
    "39": "vertebra T5",
    "40": "vertebra T4",
    "41": "vertebra T3",
    "42": "vertebra T2",
    "43": "vertebra T1",
    "44": "vertebra C7",
    "45": "vertebra C6",
    "46": "vertebra C5",
    "47": "vertebra C4",
    "48": "vertebra C3",
    "49": "vertebra C2",
    "50": "vertebra C1",
    "51": "heart",
    "52": "aorta",
    "53": "pulmonary vein",
    "54": "brachiocephalic trunk",
    "55": "right subclavian artery",
    "56": "left subclavian artery",
    "57": "right common carotid artery",
    "58": "left common carotid artery",
    "59": "left brachiocephalic vein",
    "60": "right brachiocephalic vein",
    "61": "left atrial appendage",
    "62": "superior vena cava",
    "63": "inferior vena cava",
    "64": "portal vein and splenic vein",
    "65": "left iliac artery",
    "66": "right iliac artery",
    "67": "left iliac vein",
    "68": "right iliac vein",
    "69": "left humerus",
    "70": "right humerus",
    "71": "left scapula",
    "72": "right scapula",
    "73": "left clavicle",
    "74": "right clavicle",
    "75": "left femur",
    "76": "right femur",
    "77": "left hip",
    "78": "right hip",
    "79": "spinal cord",
    "80": "left gluteus maximus",
    "81": "right gluteus maximus",
    "82": "left gluteus medius",
    "83": "right gluteus medius",
    "84": "left gluteus minimus",
    "85": "right gluteus minimus",
    "86": "left autochthon",
    "87": "right autochthon",
    "88": "left iliopsoas",
    "89": "right iliopsoas",
    "90": "brain",
    "91": "skull",
    "92": "left 1st rib",
    "93": "left 2nd rib",
    "94": "left 3rd rib",
    "95": "left 4th rib",
    "96": "left 5th rib",
    "97": "left 6th rib",
    "98": "left 7th rib",
    "99": "left 8th rib",
    "100": "left 9th rib",
    "101": "left 10th rib",
    "102": "left 11th rib",
    "103": "left 12th rib",
    "104": "right 1st rib",
    "105": "right 2nd rib",
    "106": "right 3rd rib",
    "107": "right 4th rib",
    "108": "right 5th rib",
    "109": "right 6th rib",
    "110": "right 7th rib",
    "111": "right 8th rib",
    "112": "right 9th rib",
    "113": "right 10th rib",
    "114": "right 11th rib",
    "115": "right 12th rib",
    "116": "sternum",
    "117": "costal cartilages",
}

labels_map_MR = {
    "1": "spleen",
    "2": "right kidney",
    "3": "left kidney",
    "4": "gallbladder",
    "5": "liver",
    "6": "stomach",
    "7": "pancreas",
    "8": "right adrenal gland",
    "9": "left adrenal gland",
    "10": "left lung",
    "11": "right lung",
    "12": "esophagus",
    "13": "small bowel",
    "14": "duodenum",
    "15": "colon",
    "16": "urinary bladder",
    "17": "prostate",
    "18": "sacrum",
    "19": "vertebrae",
    "20": "intervertebral discs",
    "21": "spinal cord",
    "22": "heart",
    "23": "aorta",
    "24": "inferior vena cava",
    "25": "portal and splenic veins",
    "26": "left iliac artery",
    "27": "right iliac artery",
    "28": "left iliac vein",
    "29": "right iliac vein",
    "30": "left humerus",
    "31": "right humerus",
    "32": "left scapula",
    "33": "right scapula",
    "34": "left clavicle",
    "35": "right clavicle",
    "36": "left femur",
    "37": "right femur",
    "38": "left hip",
    "39": "right hip",
    "40": "left gluteus maximus",
    "41": "right gluteus maximus",
    "42": "left gluteus medius",
    "43": "right gluteus medius",
    "44": "left gluteus minimus",
    "45": "right gluteus minimus",
    "46": "left autochthon",
    "47": "right autochthon",
    "48": "left iliopsoas",
    "49": "right iliopsoas",
    "50": "brain",
}

benchmark_plan = {
    "dataset_info": dataset_info,
    "tasks": [
        {
            "image_modality": "CT",
            "image_description": "computed tomography (CT) scan",
            "image_folder": "TotalSegmentator-CT/Images",  # Directory containing the .nii.gz image files
            "mask_folder": "TotalSegmentator-CT/Masks",  # Directory containing the mask files
            "image_prefix": "",  # String before case ID in image filename (e.g., "" for "case123_0000.nii.gz")
            "image_suffix": ".nii.gz",  # String after case ID in image filename (e.g., "_0000.nii.gz" for "case123_0000.nii.gz")
            "mask_prefix": "",  # String before case ID in mask filename
            "mask_suffix": ".nii.gz",  # String after case ID in mask filename
            "labels_map": labels_map_CT,  # Dictionary mapping mask values to class labels
        },
        {
            "image_modality": "MRI",
            "image_description": "magnetic resonance imaging (MRI) scan",
            "image_folder": "TotalSegmentator-MR/Images",  # Directory containing the .nii.gz image files
            "mask_folder": "TotalSegmentator-MR/Masks",  # Directory containing the mask files
            "image_prefix": "",  # String before case ID in image filename (e.g., "" for "case123_0000.nii.gz")
            "image_suffix": ".nii.gz",  # String after case ID in image filename (e.g., "_0000.nii.gz" for "case123_0000.nii.gz")
            "mask_prefix": "",  # String before case ID in mask filename
            "mask_suffix": ".nii.gz",  # String after case ID in mask filename
            "labels_map": labels_map_MR,  # Dictionary mapping mask values to class labels
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
