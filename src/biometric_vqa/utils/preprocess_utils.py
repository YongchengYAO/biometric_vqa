import os
import glob
import shutil
import sys
import filecmp
import hashlib
import nibabel as nib
import numpy as np
import json
from pathlib import Path


def move_folder(source_folder, destination_folder, create_dest=True):
    """
    Moves a folder from source to destination.

    Args:
        source_folder (str): Path to the source folder to move
        destination_folder (str): Path to the destination location
        create_dest (bool): Whether to create the destination parent directory if it doesn't exist

    Returns:
        bool: True if successful, False otherwise

    Raises:
        FileNotFoundError: If the source folder doesn't exist
    """
    # Check if source folder exists
    if not os.path.exists(source_folder):
        raise FileNotFoundError(f"Source folder does not exist: {source_folder}")
    # Create destination directory if it doesn't exist and create_dest is True
    if create_dest and not os.path.exists(os.path.dirname(destination_folder)):
        os.makedirs(os.path.dirname(destination_folder), exist_ok=True)
    try:
        # Check if destination folder exists
        if os.path.exists(destination_folder):
            # If destination exists, move contents
            for item in os.listdir(source_folder):
                s = os.path.join(source_folder, item)
                d = os.path.join(destination_folder, item)
                shutil.move(s, d)
        else:
            # If destination doesn't exist, move the entire folder
            shutil.move(source_folder, destination_folder)
        print(f"Successfully moved '{source_folder}' to '{destination_folder}'")
        return True
    except Exception as e:
        print(f"Failed to move folder: {e}")
        return False


def check_nii_header_for_img_mask(image_path, mask_path):
    # Inspect the mask files
    mask_nii = nib.load(mask_path)
    mask_data = mask_nii.get_fdata()
    mask_file_info = {
        "voxel_size": tuple(round(x, 3) for x in mask_nii.header.get_zooms()),
        "affine": np.round(mask_nii.affine, 3),
        "orientation": nib.orientations.aff2axcodes(mask_nii.affine),
        "array_size": mask_data.shape,
    }
    # Inspect the image files
    img_nii = nib.load(image_path)
    img_data = img_nii.get_fdata()
    image_file_info = {
        "voxel_size": tuple(round(x, 3) for x in img_nii.header.get_zooms()),
        "affine": np.round(img_nii.affine, 3),
        "orientation": nib.orientations.aff2axcodes(img_nii.affine),
        "array_size": img_data.shape,
    }
    # Check if mask and image properties match
    print(
        f"Checking properties for the image and mask images:\nImage: {image_path}\nMask: {mask_path}"
    )
    for key in mask_file_info:
        if isinstance(mask_file_info[key], np.ndarray):
            if not np.allclose(
                mask_file_info[key], image_file_info[key], atol=1e-5, rtol=1e-3
            ):
                raise ValueError(
                    f"\n\nMismatch in {key} between image and mask:\n"
                    f"Image {key}:\n{image_file_info[key]}\n"
                    f"Mask {key}:\n{mask_file_info[key]}\n"
                )
        elif mask_file_info[key] != image_file_info[key]:
            raise ValueError(
                f"\n\nMismatch in {key} between image and mask:\n"
                f"Image {key}:\n{image_file_info[key]}\n"
                f"Mask {key}:\n{mask_file_info[key]}\n"
            )
    print(f"Properties (NIfTI file header) match!\n")


def check_nii_header_for_img_mask_batch(image_dir, mask_dir):
    """
    Check NIfTI headers for all matching image and mask pairs in given directories

    Args:
        image_dir (str): Directory containing image files
        mask_dir (str): Directory containing mask files
    """
    # Get all nii.gz files in image directory
    image_files = glob.glob(os.path.join(image_dir, "*.nii.gz"))
    total_files = len(image_files)

    print(f"Found {total_files} image files. Starting header check...\n")

    for idx, image_path in enumerate(image_files, 1):
        # Get corresponding mask file name
        image_name = os.path.basename(image_path)
        mask_path = os.path.join(mask_dir, image_name)

        print(f"Processing file {idx}/{total_files}")

        # Check if mask exists
        if not os.path.exists(mask_path):
            print(f"WARNING: No matching mask found for {image_name}\n")
            continue

        try:
            check_nii_header_for_img_mask(image_path, mask_path)
        except ValueError as e:
            print(f"ERROR: {str(e)}")
            continue
        except Exception as e:
            print(f"ERROR: Unexpected error processing {image_name}: {str(e)}\n")
            continue

    print("\nHeader check completed for all files!")


def compare_nifti_folders(folder1, folder2, check_content=False, recursive=False):
    """
    Compare .nii.gz files in two folders, printing messages for files in folder1
    that don't exist in folder2.

    Args:
        folder1 (str): Path to the first folder
        folder2 (str): Path to the second folder
        check_content (bool): If True, compare file contents, not just names
        recursive (bool): If True, search subdirectories recursively

    Returns:
        list: List of missing files (relative paths)
    """

    def files_are_identical(file1, file2):
        """
        Check if two files have identical content using hash comparison.

        Args:
            file1 (Path): Path to first file
            file2 (Path): Path to second file

        Returns:
            bool: True if files have identical content, False otherwise
        """
        # For small files, use direct comparison
        if file1.stat().st_size < 100 * 1024 * 1024:  # Less than 100MB
            return filecmp.cmp(file1, file2, shallow=False)

        # For larger files, compare using hashing
        return get_file_hash(file1) == get_file_hash(file2)

    def get_file_hash(filepath, chunk_size=8192):
        """Calculate SHA-256 hash of a file in chunks to handle large files."""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                sha256.update(data)
        return sha256.hexdigest()

    folder1_path = Path(folder1)
    folder2_path = Path(folder2)

    # Make sure both folders exist
    if not folder1_path.exists():
        raise ValueError(f"Source folder does not exist: {folder1}")
    if not folder2_path.exists():
        raise ValueError(f"Target folder does not exist: {folder2}")

    # Get all .nii.gz files in folder1
    pattern = "**/*.nii.gz" if recursive else "*.nii.gz"
    files1 = list(folder1_path.glob(pattern))

    missing_files = []

    print(f"Comparing {len(files1)} .nii.gz files from {folder1} with {folder2}...")

    for file1 in files1:
        # Get relative path if recursive
        rel_path = file1.relative_to(folder1_path) if recursive else file1.name
        file2 = folder2_path / rel_path

        if not file2.exists():
            print(f"Missing file: {rel_path}")
            missing_files.append(str(rel_path))
        elif check_content and not files_are_identical(file1, file2):
            print(f"Different content: {rel_path}")
            missing_files.append(str(rel_path))

    if not missing_files:
        print("All files from folder1 exist in folder2")
    else:
        print(f"Found {len(missing_files)} missing or different files")

    return missing_files


def print_unique_values(nii_path, verbose=True):
    """
    Print the unique values in a NIfTI (.nii.gz) file

    Args:
        nii_path (str): Path to the NIfTI file
        verbose (bool): If True, print additional statistics
        max_display (int): Maximum number of values to display

    Returns:
        numpy.ndarray: Array of unique values
    """
    # Load the NIfTI file
    img = nib.load(nii_path)
    data = img.get_fdata()

    # Get unique values
    unique_vals = np.unique(data)

    # Print results
    filename = os.path.basename(nii_path)
    print(f"\nUnique values in {filename}:")
    print(f"Total unique values: {len(unique_vals)}")

    if verbose:
        print(f"Data type: {data.dtype}")
        print(f"Min value: {np.min(data)}")
        print(f"Max value: {np.max(data)}")
        print(f"Data shape: {data.shape}")

    # Display all unique values, regardless of the number
    print(f"Values: {unique_vals}")

    return unique_vals


# Example for processing a directory of files
def print_unique_values_batch(directory):
    """Process all .nii.gz files in a directory"""
    for nii_file in Path(directory).glob("*.nii.gz"):
        print_unique_values(str(nii_file))


def check_noninteger_labels(folder_path, log_out_dir):
    # List to store filenames with non-integer values
    non_integer_files = []

    # Get total number of files for progress tracking
    total_files = sum(
        1
        for _, _, files in os.walk(folder_path)
        for file in files
        if file.endswith(".nii.gz")
    )
    processed = 0

    # Walk through the directory
    print(f"Checking {total_files} files for non-integer labels...")
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".nii.gz"):
                processed += 1
                file_path = os.path.join(root, file)
                print(f" - Checking {processed}/{total_files}: {file}")

                try:
                    img = nib.load(file_path)
                    data = img.get_fdata()
                    unique_vals = np.unique(data)
                    is_all_integer = np.all(np.equal(np.mod(unique_vals, 1), 0))
                    if not is_all_integer:
                        non_integer_files.append(
                            {
                                "filename": file,
                                "unique_values": unique_vals.tolist(),  # Convert to list for JSON serialization
                            }
                        )
                except Exception as e:
                    print(f"\n\nError checking {file}: {str(e)}\n\n")

    # Print results and save to file if non-integer files found
    if non_integer_files:
        print(f"\nMasks with non-integer values in this folder: {folder_path}:\n")
        for item in non_integer_files:
            print(f"Filename: {item['filename']}")
            print("Unique values found:", item["unique_values"], "\n")

        # Save to file
        with open(f"{log_out_dir}/non_integer_mask_files.json", "w") as f:
            json.dump(non_integer_files, f, indent=2)

        sys.exit("\n\nError: Non-integer values found in segmentation masks\n\n")
    else:
        print("\nAll mask files contain integer values only!\n")


def split_4d_nifti(input_dir, out_dir):
    """
    Split 4D NIfTI files in the input directory into separate 3D files.
    Automatically detects the length of the 4th dimension.
    """
    # Get all .nii.gz files in the Images directory
    nifti_files = glob.glob(os.path.join(input_dir, "*.nii.gz"))

    for file_path in nifti_files:
        # Load the NIfTI file
        img = nib.load(file_path)
        data = img.get_fdata()

        # Check if it's a 4D image
        if len(data.shape) != 4:
            print(f"Skipping {file_path} - not a 4D image")
            continue

        # Get the length of the 4th dimension
        time_points = data.shape[3]

        # Create output directories if they don't exist
        for i in range(1, time_points + 1):
            os.makedirs(f"{out_dir}/Images-{i}", exist_ok=True)

        # Get the base filename without extension
        base_name = os.path.basename(file_path).replace(".nii.gz", "")

        # Split and save each volume
        for i in range(time_points):
            volume = data[:, :, :, i]
            new_img = nib.Nifti1Image(volume, img.affine)
            output_path = os.path.join(f"{out_dir}/Images-{i+1}", f"{base_name}.nii.gz")
            nib.save(new_img, output_path)
            if i == time_points - 1:
                print(f"Saved {output_path} (volume {i+1}/{time_points})\n")
            else:
                print(f"Saved {output_path} (volume {i+1}/{time_points})")


def process_dataset_mm(data_dirs, seg_pattern, modalities, base_suffix, replace=False):
    """Generic function to process multi-modality datasets with different patterns"""
    for data_dir in data_dirs:
        for seg_file in glob.glob(f"{data_dir}/**/{seg_pattern}", recursive=True):
            # Extract base ID
            base_id = os.path.basename(seg_file).replace(base_suffix, "")
            dir_name = os.path.dirname(seg_file)

            # Move segmentation file
            mv_cmd = (
                shutil.move
                if not replace
                else lambda src, dst: shutil.move(src, dst, copy_function=shutil.copy2)
            )
            os.makedirs("Masks", exist_ok=True)
            mv_cmd(seg_file, f"Masks/{os.path.basename(seg_file)}")

            # Move modality files
            for modality in modalities:
                if "_" in base_suffix:
                    img_file = f"{dir_name}/{base_id}_{modality}.nii.gz"
                else:
                    img_file = f"{dir_name}/{base_id}-{modality}.nii.gz"

                if os.path.exists(img_file):
                    os.makedirs(f"Images-{modality}", exist_ok=True)
                    mv_cmd(img_file, f"Images-{modality}/{os.path.basename(img_file)}")
                else:
                    print(f"Warning: Missing {modality} file for {base_id}")


def process_dataset(
    data_dirs,
    seg_pattern,
    base_suffix,
    img_suffix=".nii.gz",
    out_dir=None,
    replace=False,
    masks_fname="Masks",
    images_fname="Images",
):
    """
    Generic function to process datasets with optional arguments: output directory, file replacement
    Logic:
    1. Within the <data_dirs> folder, find segmentation files with a given pattern: <seg_pattern>
    2. Extract base ID from the segmentation file name by removing the <base_suffix>
    3. Move the segmentation file to the <out_dir>/Masks folder (if provided)
    4. Find the corresponding image file by appending the <img_suffix> to the base ID
    5. Move the image file to the <out_dir>/Images folder (if provided)
    """
    for data_dir in data_dirs:
        for seg_file in glob.glob(f"{data_dir}/**/{seg_pattern}", recursive=True):
            # Extract base ID
            base_id = os.path.basename(seg_file).replace(base_suffix, "")
            dir_name = os.path.dirname(seg_file)

            # Move segmentation file
            mv_cmd = (
                shutil.move
                if not replace
                else lambda src, dst: shutil.move(src, dst, copy_function=shutil.copy2)
            )
            masks_dir = f"{out_dir}/{masks_fname}" if out_dir else masks_fname
            os.makedirs(masks_dir, exist_ok=True)
            mv_cmd(seg_file, f"{masks_dir}/{os.path.basename(seg_file)}")

            # Move image files
            img_file = f"{dir_name}/{base_id}{img_suffix}"
            if os.path.exists(img_file):
                images_dir = f"{out_dir}/{images_fname}" if out_dir else images_fname
                os.makedirs(images_dir, exist_ok=True)
                mv_cmd(img_file, f"{images_dir}/{os.path.basename(img_file)}")
            else:
                print(f"Warning: Missing image file for {base_id}")


def match_and_clean_files(images_dir, masks_dir):
    print(
        f"Checking for matching image and mask files in {images_dir} and {masks_dir}...\n"
    )
    print("Removing image files without corresponding mask files...\n")

    # Get list of image files
    image_files = glob.glob(os.path.join(images_dir, "*_0000.nii.gz"))
    total_images = len(image_files)
    print(f"Found {total_images} image files")

    removed_count = 0

    # Process each image file
    for i, image_path in enumerate(image_files, 1):
        # Extract ID from image filename
        image_name = os.path.basename(image_path)
        image_id = image_name.replace("_0000.nii.gz", "")

        # Construct expected mask filename
        mask_path = os.path.join(masks_dir, f"{image_id}.nii.gz")

        print(f"Checking {i}/{total_images}: {image_name}")

        # Check if corresponding mask exists
        if not os.path.exists(mask_path):
            print(f" - Removing {image_name} - No corresponding mask found")
            os.remove(image_path)
            removed_count += 1
        else:
            print(f" - Found matching mask for {image_name}")

    print("\nSummary:")
    print(f"Total images processed: {total_images}")
    print(f"Images removed: {removed_count}")
    print(f"Images remaining: {total_images - removed_count}")


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
