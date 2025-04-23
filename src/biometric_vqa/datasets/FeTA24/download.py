import os
import shutil
import argparse
import synapseclient
import glob
import json
import zipfile
import gzip
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from pathlib import Path
from biometric_vqa.utils.preprocess_utils import process_dataset, move_folder


# ====================================
# Dataset Info [!]
# ====================================
# Dataset: FeTA24
# Challenge: https://fetachallenge.github.io
# Data: (only 80/120 cases are available after the FeTA24 challenge)
#   - Data from the University Children’s Hospital Zurich (80 cases):
#     https://www.synapse.org/Synapse:syn25649159/wiki/610007
# Format: nii.gz
# ====================================


# Map of landmark points to their corresponding slice dimension
LANDMARKS_SLICE_DIM = {
    "P1": 0,
    "P2": 0,
    "P3": 0,
    "P4": 0,
    "P5": 2,
    "P6": 2,
    "P7": 2,
    "P8": 2,
    "P9": 1,
    "P10": 1,
}

LABELS_NAME = {
    "1": "Corpus_Callosum_Length",
    "2": "Vermis_Height",
    "3": "Brain_Biparietal_Diameter",
    "4": "Skull_Biparietal_Diameter",
    "5": "Transverse_Cerebellar_Diameter",
}


def im_original_to_realigned(feta_dir, biometry_dir, out_dir, seg=False):
    """
    Transform images from original space to realigned space.

    Args:
        feta_dir (str): Path to the original FeTA images directory
        biometry_dir (str): Path to the biometric measurements directory
        out_dir (str): Path to the output directory
        seg (bool): If True, process segmentations instead of T2w images
    """
    # Convert paths to absolute paths
    feta_dir = os.path.abspath(feta_dir)
    biometry_dir = os.path.abspath(biometry_dir)
    out_dir = os.path.abspath(out_dir)

    # Validate input directories
    if not os.path.exists(feta_dir):
        raise FileNotFoundError(f"Folder {feta_dir} does not exist.")
    if not os.path.exists(biometry_dir):
        raise FileNotFoundError(f"Folder {biometry_dir} does not exist.")

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    # Get list of subjects (removing 'sub-' prefix)
    sub_list = sorted([f[4:] for f in os.listdir(feta_dir) if "sub" in f])

    # Initialize counter for saved files
    files_saved = 0

    # Process each subject
    for sub in sub_list:
        # Construct paths for current subject using os.path.join for cross-platform compatibility
        sub_path_feta = os.path.join(feta_dir, f"sub-{sub}", "anat")
        sub_path_bio = os.path.join(biometry_dir, f"sub-{sub}", "anat")

        # Skip if no biometry data exists
        if not os.path.exists(sub_path_bio):
            continue

        # Determine file suffix based on processing mode
        suffix = "dseg.nii.gz" if seg else "T2w.nii.gz"

        # Read input image and transformation
        imp = get_file(sub_path_feta, suffix=suffix)
        im = sitk.ReadImage(imp)
        trf = sitk.ReadTransform(get_file(sub_path_bio, suffix=".txt"))

        # Set up output path
        out_path = os.path.join(out_dir, f"sub-{sub}", "anat", os.path.basename(imp))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Choose interpolation method based on image type
        interMethod = sitk.sitkNearestNeighbor if seg else sitk.sitkLinear

        # Apply transformation and save
        im = sitk.Resample(im, im, trf, interMethod)
        sitk.WriteImage(im, out_path)
        files_saved += 1

    print(f"-- Total files saved: {files_saved}")


def get_file(folder, suffix):
    for file in os.listdir(folder):
        if file.endswith(suffix):
            return os.path.join(folder, file)
    return None


def process_landmark_points(
    landmark_mask_dir: str,
    landmark_json_dir: str,
    img_dir: str,
    fig_dir: str,
    slice_dim_map: dict = LANDMARKS_SLICE_DIM,
    labels_name: dict = LABELS_NAME,
):
    """Process landmark points from mask files and save coordinates and visualizations."""

    # Create output directories
    for dir_path in [landmark_json_dir, img_dir, fig_dir]:
        os.makedirs(dir_path, exist_ok=True)

    def plot_landmarks(img_data, points, point_names, slice_dim, coord, save_path):
        """Helper function to plot landmarks on image slices."""
        plt.figure()

        # Handle different slice orientations
        if slice_dim == 0:  # Sagittal
            plt.imshow(img_data[coord, :, :].T, cmap="gray", origin="lower")
            x_coords, y_coords = points[:, 1], points[:, 2]
            plt.xlabel("Anterior →", fontsize=14)
            plt.ylabel("Superior →", fontsize=14)
        elif slice_dim == 1:  # Coronal
            plt.imshow(img_data[:, coord, :].T, cmap="gray", origin="lower")
            x_coords, y_coords = points[:, 0], points[:, 2]
            plt.xlabel("Right →", fontsize=14)
            plt.ylabel("Superior →", fontsize=14)
        else:  # Axial
            plt.imshow(img_data[:, :, coord].T, cmap="gray", origin="lower")
            x_coords, y_coords = points[:, 0], points[:, 1]
            plt.xlabel("Right →", fontsize=14)
            plt.ylabel("Anterior →", fontsize=14)

        # Plot points and labels
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            plt.scatter(
                [x],
                [y],
                facecolors="#18A727",
                edgecolors="black",
                marker="o",
                s=80,
                linewidth=1.5,
                label=point_names[i],
            )
            plt.annotate(
                point_names[i],
                (x, y),
                xytext=(2, 2),
                textcoords="offset points",
                color="#FE9100",
                fontsize=14,
            )

        plt.margins(0)
        plt.savefig(save_path)
        plt.close()

    for nii_file in glob.glob(os.path.join(landmark_mask_dir, "*.nii.gz")):
        try:
            nii_path = Path(nii_file)
            data = nib.load(str(nii_path)).get_fdata()
            img_data = nib.load(
                os.path.join(
                    img_dir, nii_path.name.replace("meas.nii.gz", "T2w.nii.gz")
                )
            ).get_fdata()

            # This data structure is designed to be compatible with biometric data constructed from segmentation masks
            json_dict = {
                "slice_landmarks_x": [],
                "slice_landmarks_y": [],
                "slice_landmarks_z": [],
            }

            # Process each label
            for label in range(1, 6):
                # Get coordinates
                coords = np.where(data == label)
                points = np.array(list(zip(coords[0], coords[1], coords[2])))

                if len(points) != 2:
                    raise ValueError(
                        f"Label {label} has {len(points)} points, expected 2"
                    )

                # Determine point indices and names based on label
                subfolder = os.path.join(fig_dir, labels_name[str(label)])
                os.makedirs(subfolder, exist_ok=True)

                # Determine point order based on anatomical measurement type
                if label == 1:  # Corpus Callosum Length
                    idx_larger = np.argmax(points[:, 1])  # More anterior point
                    point_names = ["P1", "P2"]
                elif label == 2:  # Vermis Height
                    idx_larger = np.argmax(points[:, 2])  # Superior point
                    point_names = ["P3", "P4"]
                elif label == 3:  # Brain Biparietal Diameter
                    idx_larger = np.argmax(points[:, 0])  # Right point
                    point_names = ["P5", "P6"]
                elif label == 4:  # Skull Biparietal Diameter
                    idx_larger = np.argmax(points[:, 0])  # Right point
                    point_names = ["P7", "P8"]
                else:  # Transverse Cerebellar Diameter
                    idx_larger = np.argmax(points[:, 0])  # Right point
                    point_names = ["P9", "P10"]

                idx_smaller = 1 - idx_larger
                sorted_points = points[[idx_larger, idx_smaller]]

                # Save landmark coordinates
                slice_dim = slice_dim_map[point_names[0]]
                if slice_dim == 0:
                    json_dict_key = "slice_landmarks_x"
                elif slice_dim == 1:
                    json_dict_key = "slice_landmarks_y"
                else:
                    json_dict_key = "slice_landmarks_z"
                slice_idx = sorted_points[0].tolist()[slice_dim]
                landmarks_dict = {
                    "slice_idx": slice_idx,
                    "landmarks": {
                        point_names[0]: sorted_points[0].tolist(),
                        point_names[1]: sorted_points[1].tolist(),
                    },
                }
                json_dict[json_dict_key].append(landmarks_dict)

                # Plot visualization
                slice_dim = slice_dim_map[point_names[0]]
                for coord in sorted_points[:, slice_dim].astype(int):
                    save_path = os.path.join(
                        subfolder,
                        f"{nii_path.name.replace('_meas.nii.gz','')}_slice{coord}.png",
                    )
                    plot_landmarks(
                        img_data,
                        sorted_points,
                        point_names,
                        slice_dim,
                        coord,
                        save_path,
                    )

            # Save landmarks to JSON
            with gzip.open(
                os.path.join(
                    landmark_json_dir,
                    f"{nii_path.name.replace('_meas.nii.gz','')}.json.gz",
                ),
                "wt",
            ) as f:
                json.dump(json_dict, f, indent=4)
            print(f"Processed {nii_path.name}")

        except Exception as e:
            print(f"Error processing {nii_path.name}: {str(e)}")


def download_and_extract(dataset_dir, dataset_name):
    # Download files
    current_dir = os.getcwd()
    os.chdir(dataset_dir)
    tmp_dir = os.path.join(dataset_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    os.chdir(tmp_dir)
    print(f"Downloading {dataset_name} dataset to {dataset_dir}...")

    # ====================================
    # Add download logic here [!]
    # ====================================
    # Initialize Synapse client
    syn = synapseclient.Synapse()
    token = os.environ.get("SYNAPSE_TOKEN")
    if not token:
        raise ValueError("SYNAPSE_TOKEN environment variable not set")
    syn.login(authToken=token)

    # Download dataset using synapse client
    syn.get("syn25649833", downloadLocation=tmp_dir)

    # Extract the downloaded zip file
    with zipfile.ZipFile("feta_2.4.zip", "r") as zip_ref:
        zip_ref.extractall(tmp_dir)

    print("Creating directories...")
    dirs_to_create = [
        "Images",
        "Images-reoriented",
        "Masks",
        "Masks-reoriented",
        "Masks-landmarks-reoriented",
    ]
    for dir_name in dirs_to_create:
        os.makedirs(os.path.join(tmp_dir, dir_name), exist_ok=True)

    # Set up paths
    dir_feta = os.path.join(tmp_dir, "feta_2.4")
    dir_biometry = os.path.join(tmp_dir, "feta_2.4", "derivatives", "biometry")
    dir_reo = os.path.join(tmp_dir, "feta_2.4", "derivatives", "im_reo")

    print("Reorient images to biometry measurement space...")
    im_original_to_realigned(
        feta_dir=dir_feta, biometry_dir=dir_biometry, out_dir=dir_reo, seg=False
    )

    print("Reorient masks to biometry measurement space...")
    im_original_to_realigned(
        feta_dir=dir_feta, biometry_dir=dir_biometry, out_dir=dir_reo, seg=True
    )

    print("Moving reoriented images and masks...")
    process_dataset(
        [str(dir_reo)],
        "*_dseg.nii.gz",
        "_dseg.nii.gz",
        img_suffix="_T2w.nii.gz",
        out_dir=tmp_dir,
        images_fname="Images-reoriented",
        masks_fname="Masks-reoriented",
    )

    print("Moving landmark files...")
    landmark_count = 0
    for lm_file in glob.glob(
        os.path.join(dir_biometry, "**", "*meas.nii.gz"), recursive=True
    ):
        shutil.move(
            lm_file,
            os.path.join(
                tmp_dir, "Masks-landmarks-reoriented", os.path.basename(lm_file)
            ),
        )
        landmark_count += 1
    print(f"-- Moved {landmark_count} landmark files")

    print("Moving raw images and masks...")
    process_dataset(
        ["feta_2.4"],
        "*_dseg.nii.gz",
        "_dseg.nii.gz",
        img_suffix="_T2w.nii.gz",
        out_dir=tmp_dir,
    )

    # List of cases to remove (base filenames without extension)
    cases_to_remove = [
        "sub-005_rec-mial",
        "sub-025_rec-mial",
        "sub-066_rec-irtk",
        "sub-032_rec-mial",
        "sub-016_rec-mial",
    ]

    # Remove landmarks and images for each case
    for case in cases_to_remove:
        for path in [
            os.path.join(tmp_dir, "Masks-landmarks-reoriented", f"{case}_meas.nii.gz"),
            os.path.join(tmp_dir, "Images-reoriented", f"{case}_T2w.nii.gz"),
        ]:
            if os.path.exists(path):
                os.remove(path)
                print(f"Removed {os.path.basename(path)}")

    # Save landmark coordinates to json file
    process_landmark_points(
        landmark_mask_dir="Masks-landmarks-reoriented",
        landmark_json_dir="Landmarks",
        img_dir="Images-reoriented",
        fig_dir="Landmarks-fig",
        slice_dim_map=LANDMARKS_SLICE_DIM,
        labels_name=LABELS_NAME,
    )

    # Move folder to dataset_dir
    folders_to_move = [
        "Images",
        "Images-reoriented",
        "Landmarks",
        "Landmarks-fig",
        "Masks",
        "Masks-reoriented",
    ]
    for folder in folders_to_move:
        move_folder(
            os.path.join(tmp_dir, folder),
            os.path.join(dataset_dir, folder),
            create_dest=True,
        )
    # ====================================

    print(f"Download and extraction completed for {dataset_name}")
    os.chdir(dataset_dir)
    shutil.rmtree(tmp_dir)
    os.chdir(current_dir)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Download and extract dataset")
    parser.add_argument(
        "-d",
        "--dir_datasets_data",
        help="Directory path where datasets will be stored",
        required=True,
    )
    parser.add_argument(
        "-n",
        "--dataset_name",
        help="Name of the dataset",
        required=True,
    )
    args = parser.parse_args()

    # Create dataset directory
    dataset_dir = os.path.join(args.dir_datasets_data, args.dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    # Change to dataset directory
    os.chdir(dataset_dir)

    # Download and extract dataset
    download_and_extract(dataset_dir, args.dataset_name)
