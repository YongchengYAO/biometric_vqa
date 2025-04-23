import os
import shutil
import glob
import json
import rarfile
import zipfile
import gzip
import urllib.request
import argparse
import nibabel as nib
import matplotlib.pyplot as plt
from biometric_vqa.utils.data_conversion import convert_bmp_to_niigz
from biometric_vqa.utils.preprocess_utils import move_folder


# ====================================
# Dataset Info [!]
# ====================================
# Dataset: Ceph-Biometrics-400
# Data (original): https://figshare.com/s/37ec464af8e81ae6ebbf
# Data (HF): https://huggingface.co/datasets/YongchengYAO/Ceph-Biometrics-400
# Format (original): bm
# Format (HF): nii.gz
# ====================================


def process_landmarks_data(
    landmarks_txt_dir: str,
    landmarks_json_dir: str,
    n: int,
    img_sizes,
    flip_dim0=False,
    flip_dim1=False,
    swap_dim01=False,
) -> None:
    """
    Read landmark points from all txt files in a directory and save as JSON files.

    Args:
        in_dir (str): Directory containing the txt files
        out_dir (str): Directory where JSON files will be saved
        n (int): Number of lines to read from each file
        height_width_orig: Original height and width of the image
        swap_xy (bool): Whether to swap x and y coordinates
        slip_x (bool): Whether to flip coordinates along x-axis
        slip_y (bool): Whether to flip coordinates along y-axis
    """
    (
        os.makedirs(landmarks_json_dir, exist_ok=True)
        if not os.path.exists(landmarks_json_dir)
        else None
    )

    for txt_file in glob.glob(os.path.join(landmarks_txt_dir, "*.txt")):

        landmarks = {}
        filename = os.path.basename(txt_file)
        json_path = os.path.join(
            landmarks_json_dir, filename.replace(".txt", ".json.gz")
        )

        try:
            with open(txt_file, "r") as f:
                for i in range(n):
                    line = f.readline().strip()
                    if not line:
                        break
                    # Note: this is correct, DO NOT SWAP idx_dim0 and idx_dim1
                    # Assuming an image with height and width:
                    # - The data array read from bmp file is of size (height, width) -- dim0 is height, dim1 is width
                    # - The landmark coordinates are defined as the indices in width (dim1) and height (dim0) directions
                    idx_dim1, idx_dim0 = map(int, line.split(","))

                    # Apply transformations
                    # Note: this is correct
                    # DO NOT SWAP the order of transformations
                    if flip_dim0:
                        idx_dim0 = img_sizes[0] - idx_dim0
                    if flip_dim1:
                        idx_dim1 = img_sizes[1] - idx_dim1
                    if swap_dim01:  # this line should be AFTER slip_x and slip_y
                        idx_dim0, idx_dim1 = idx_dim1, idx_dim0

                    # Save landmark coordinates in 0-based indices
                    landmarks[f"P{i+1}"] = [
                        coord - 1 for coord in [1, idx_dim0, idx_dim1]
                    ]

            # This data structure is designed to be compatible with biometric data constructed from segmentation masks
            json_dict = {
                "slice_landmarks_x": [
                    {
                        "slice_idx": 1,
                        "landmarks": landmarks,
                    },
                ],
                "slice_landmarks_y": [],
                "slice_landmarks_z": [],
            }

            # Save to JSON or compressed JSON
            if json_path.endswith(".json.gz"):
                with gzip.open(json_path, "wt") as f:
                    json.dump(json_dict, f, indent=4)
            else:
                with open(json_path, "w") as f:
                    json.dump(json_dict, f, indent=4)

        except FileNotFoundError:
            print(f"Error: File {txt_file} not found")
        except ValueError:
            print(f"Error: Invalid format in file {txt_file}")
        except Exception as e:
            print(f"Error reading file {txt_file}: {str(e)}")


def plot_sagittal_slice_with_landmarks(
    nii_path: str, json_path: str, fig_path: str = None
):
    """Plot first slice from NIfTI file and overlay landmarks from JSON file.

    Args:
        nii_path (str): Path to .nii.gz file
        json_path (str): Path to landmarks JSON file
        fig_path (str, optional): Path to save the plot. If None, displays plot
    """
    # Load NIfTI image and extract first slice
    nii_img = nib.load(nii_path)
    slice_data = nii_img.get_fdata()[0, :, :]

    # Load landmark coordinates from JSON
    if json_path.endswith(".json.gz"):
        with gzip.open(json_path, "rt") as f:
            landmarks_json = json.load(f)
    else:
        with open(json_path, "r") as f:
            landmarks_json = json.load(f)

    # Setup visualization
    plt.figure(figsize=(12, 12))
    plt.imshow(
        slice_data.T, cmap="gray", origin="lower"
    )  # the transpose is necessary only for visualization

    # Extract and plot landmark coordinates
    coords_dim0 = []
    coords_dim1 = []
    landmarks = landmarks_json["slice_landmarks_x"][0]["landmarks"]
    for point_id, coords in landmarks.items():
        if len(coords) == 3:  # Check for valid [1, x, y] format
            # Note: this is definitely correct, DO NOT SWAP coords[1] and coords[2]
            coords_dim0.append(coords[1])
            coords_dim1.append(coords[2])

    # Add landmarks and labels
    plt.scatter(
        coords_dim0,
        coords_dim1,
        facecolors="#18A727",
        edgecolors="black",
        marker="o",
        s=80,
        linewidth=1.5,
    )
    for i, (x, y) in enumerate(zip(coords_dim0, coords_dim1), 1):
        plt.annotate(
            f"$\\mathbf{{{i}}}$",
            (x, y),
            xytext=(2, 2),
            textcoords="offset points",
            color="#FE9100",
            fontsize=14,
        )

    # Configure plot appearance
    plt.xlabel("Anterior →", fontsize=14)
    plt.ylabel("Superior →", fontsize=14)
    plt.margins(0)

    # Save or display the plot
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    print(f"Plot saved to: {fig_path}")
    plt.close()


def plot_sagittal_slice_with_landmarks_batch(
    image_dir: str, landmark_dir: str, fig_dir: str
):
    """Plot all cases from given directories.

    Args:
        image_dir (str): Directory containing .nii.gz files
        landmark_dir (str): Directory containing landmark JSON files
        fig_dir (str): Directory to save output figures

    """
    # Create output directory if it doesn't exist
    os.makedirs(fig_dir, exist_ok=True)

    # Process each .nii.gz file
    for nii_path in glob.glob(os.path.join(image_dir, "*.nii.gz")):
        base_name = os.path.splitext(os.path.splitext(os.path.basename(nii_path))[0])[0]
        json_path = os.path.join(landmark_dir, f"{base_name}.json.gz")
        fig_path = os.path.join(fig_dir, f"{base_name}.png")

        # Plot and save
        if os.path.exists(json_path):
            plot_sagittal_slice_with_landmarks(nii_path, json_path, fig_path)
        else:
            print(f"Warning: No landmark file found for {base_name}")


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
    # Download the file using urllib
    url = "https://figshare.com/ndownloader/articles/3471833?private_link=37ec464af8e81ae6ebbf"
    output_file = "Cephalogram400.zip"
    print(f"Downloading file from {url}...")
    urllib.request.urlretrieve(url, output_file)

    # Extract the ZIP file
    print("Extracting ZIP file...")
    with zipfile.ZipFile(output_file, "r") as zip_ref:
        zip_ref.extractall()

    # Find and extract all RAR files
    print("Extracting RAR files...")
    for file in os.listdir("."):
        if file.endswith(".rar"):
            with rarfile.RarFile(file) as rf:
                rf.extractall()

    # Create the Images-raw directory
    os.makedirs("Images-raw", exist_ok=True)
    # Move all BMP files from RawImage to Images-raw using glob
    for src_path in glob.glob(os.path.join("RawImage", "**", "*.bmp"), recursive=True):
        shutil.move(src_path, os.path.join("Images-raw", os.path.basename(src_path)))

    # Convert BMP files to 3D nii.gz
    Flag_flip_dim0 = True
    Flag_flip_dim1 = False
    Flag_swap_dim01 = True
    img_size_dim0, img_size_dim1 = convert_bmp_to_niigz(
        "Images-raw",
        "Images",
        slice_dim_type=0,
        pseudo_voxel_size=[0.1, 0.1, 0.1],
        flip_dim0=Flag_flip_dim0,
        flip_dim1=Flag_flip_dim1,
        swap_dim01=Flag_swap_dim01,
    )

    # Read landmark points from txt files and save as JSON
    process_landmarks_data(
        "400_senior",
        "Landmarks",
        19,
        img_sizes=[img_size_dim0, img_size_dim1],
        flip_dim0=Flag_flip_dim0,
        flip_dim1=Flag_flip_dim1,
        swap_dim01=Flag_swap_dim01,
    )

    # Plot slices with landmarks
    plot_sagittal_slice_with_landmarks_batch("Images", "Landmarks", "Landmarks-fig")

    # Move folder to dataset_dir
    folders_to_move = [
        "Images",
        "Landmarks",
        "Landmarks-fig",
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
