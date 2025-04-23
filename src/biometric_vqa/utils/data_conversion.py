import SimpleITK as sitk
import os
import glob
import nrrd
import nibabel as nib
import numpy as np
import cv2
from pathlib import Path


def _reorient_niigz_RASplus(nifti_path, output_path):
    """
    Load a NIfTI file, reorient it to RAS+ (right-anterior-superior) using as_closest_canonical,
    and save the result while preserving the original data type.
    """
    # Load the image
    img = nib.load(nifti_path)
    # Get original data type
    original_dtype = img.get_fdata().dtype
    # Check current orientation
    current_orientation = nib.aff2axcodes(img.affine)
    if current_orientation == ("R", "A", "S"):
        print(f"{nifti_path} is already in RAS+ orientation.\n")
        if nifti_path != output_path:
            nib.save(img, output_path)
        return
    # Convert to RAS+ orientation
    canonical_img = nib.as_closest_canonical(img)
    # Create new image with original dtype
    reoriented_data = canonical_img.get_fdata().astype(original_dtype)
    new_img = nib.Nifti1Image(reoriented_data, canonical_img.affine, header=img.header)
    # Preserve original header information where possible
    new_img.header.set_data_dtype(original_dtype)
    # Save the reoriented image
    nib.save(new_img, output_path)
    print(f"Converted {nifti_path} to RAS+ orientation and saved as {output_path}.\n")


def reorient_niigz_RASplus_batch_inplace(dataset_dir):
    """
    Reorient all NIfTI files in a directory and its subdirectories to RAS+ orientation in place.
    This function modifies the original files rather than creating new ones.
    """
    # Find all .nii.gz files recursively in directory
    nii_files = list(glob.glob(f"{dataset_dir}/**/*.nii.gz", recursive=True))
    print(f"Reorienting {len(nii_files)} files to RAS+ orientation...\n")
    # Process each file
    for i, nii_file in enumerate(nii_files, 1):
        print(f" - [{i}/{len(nii_files)}] Processing: {os.path.basename(nii_file)}")
        # Reorient file and overwrite the original
        _reorient_niigz_RASplus(nii_file, nii_file)


def convert_nrrd_to_nifti(input_dir, output_dir, recursive=False):
    """
    Convert all .nrrd files in input_dir to .nii.gz files in output_dir

    Args:
        input_dir (str): Directory containing .nrrd files
        output_dir (str): Directory to save .nii.gz files
        recursive (bool): If True, search for .nrrd files in subdirectories
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get all .nrrd files in input directory
    pattern = "**/*.nrrd" if recursive else "*.nrrd"
    nrrd_files = list(Path(input_dir).glob(pattern))

    print(f"Found {len(nrrd_files)} .nrrd files")

    for nrrd_file in nrrd_files:
        try:
            print(f"Converting {nrrd_file.name}")

            # Read NRRD file
            data, header = nrrd.read(str(nrrd_file))

            # Get spacing (voxel size)
            space_directions = header.get("space directions")
            if space_directions is not None:
                voxel_size = np.array(
                    [np.linalg.norm(dir) for dir in space_directions if dir is not None]
                )
                print("Voxel dimensions calculated from spatial direction matrix")
            else:
                raise ValueError(
                    "No space directions found in NRRD header. Cannot determine voxel size."
                )

            # Get origin
            origin = header.get("space origin", [0.0, 0.0, 0.0])

            # Create affine matrix
            affine = np.eye(4)
            if space_directions is not None:
                affine[:3, :3] = np.array(
                    [dir if dir is not None else [0, 0, 0] for dir in space_directions]
                )
            else:
                affine[:3, :3] = np.diag(voxel_size)
            affine[:3, 3] = origin

            # Create NIfTI image
            nifti_img = nib.Nifti1Image(data, affine)

            # Set additional header information
            nifti_header = nifti_img.header
            nifti_header.set_zooms(voxel_size)

            # Create output filename
            output_file = Path(output_dir) / f"{nrrd_file.stem}.nii.gz"

            # Save NIfTI file
            nib.save(nifti_img, str(output_file))
            print(f"Saved to {output_file}")

        except Exception as e:
            print(f"Error converting {nrrd_file.name}: {e}")


def convert_mha_to_nifti(input_dir, output_dir, recursive=False):
    """
    Convert all .mha files in input_dir to .nii.gz files in output_dir

    Args:
        input_dir (str): Directory containing .mha files
        output_dir (str): Directory to save .nii.gz files
        recursive (bool): If True, search for .mha files in subdirectories
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get all .mha files in input directory
    pattern = "**/*.mha" if recursive else "*.mha"
    mha_files = list(Path(input_dir).glob(pattern))

    print(f"Found {len(mha_files)} .mha files")

    for mha_file in mha_files:
        try:
            # Read .mha file
            print(f"Converting {mha_file.name}")
            image = sitk.ReadImage(str(mha_file))

            # Create output filename
            output_file = Path(output_dir) / f"{mha_file.stem}.nii.gz"

            # Write as .nii.gz
            sitk.WriteImage(image, str(output_file))
            print(f"Saved to {output_file}")

        except Exception as e:
            print(f"Error converting {mha_file.name}: {e}")


def convert_nii_to_niigz(input_dir, output_dir, recursive=False):
    """
    Convert all .nii files in input_dir to .nii.gz files in output_dir

    Args:
        input_dir (str): Directory containing .nii files
        output_dir (str): Directory to save .nii.gz files
        recursive (bool): If True, search for .nii files in subdirectories
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get all .nii files in input directory
    pattern = "**/*.nii" if recursive else "*.nii"
    nii_files = list(Path(input_dir).glob(pattern))

    print(f"Found {len(nii_files)} .nii files")

    for nii_file in nii_files:
        try:
            # Read .nii file
            print(f"Converting {nii_file.name}")
            image = sitk.ReadImage(str(nii_file))

            # Create output filename
            output_file = Path(output_dir) / f"{nii_file.stem}.nii.gz"

            # Write as .nii.gz
            sitk.WriteImage(image, str(output_file))
            print(f"Saved to {output_file}")

        except Exception as e:
            print(f"Error converting {nii_file.name}: {e}")


def convert_mask_to_uint16_per_dir(mask_folder):
    """
    Convert all .nii.gz mask files in a folder to uint16 data type with proper header settings.
    This is useful for segmentation masks where we want integer labels without scaling.

    Args:
        mask_folder (str): Path to folder containing mask files
    """
    # List all .nii.gz files in the mask folder
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith(".nii.gz")]
    total_files = len(mask_files)
    print(f"Found {total_files} .nii.gz mask files to convert")

    for i, mask_file in enumerate(mask_files):
        # Create full path to mask file
        mask_path = os.path.join(mask_folder, mask_file)
        print(f" - [{i+1}/{total_files}] Processing mask file: {mask_file}")

        # Load the original NIfTI file
        orig_nii = nib.load(mask_path)
        # Convert data to uint16 type
        mask = orig_nii.get_fdata().astype(np.uint16)
        print("   Data: Converted data to uint16")

        # Copy the header to preserve metadata
        nii_header = orig_nii.header.copy()
        # Check and report current data type
        old_dtype = nii_header.get_data_dtype()
        if old_dtype != np.uint16:
            print(f"   File header: Current dtype={old_dtype}")
            nii_header.set_data_dtype(np.uint16)
            print("   File header: Updated dtype='np.uint16'")

        # Check and report current slope/intercept
        old_slope, old_inter = nii_header.get_slope_inter()
        if old_slope != 1 or old_inter != 0:
            print(f"   File header: Current slope={old_slope}, intercept={old_inter}")
            nii_header.set_slope_inter(1, 0)
            print("   File header: Updated slope=1, intercept=0")

        # Create new NIfTI image with updated mask and header
        new_img = nib.Nifti1Image(mask, orig_nii.affine, nii_header)

        # Save the modified image, overwriting the original
        nib.save(new_img, mask_path)


def convert_bmp_to_niigz(
    bmp_dir,
    niigz_dir,
    slice_dim_type,
    pseudo_voxel_size,
    flip_dim0=False,
    flip_dim1=False,
    swap_dim01=False,
):
    """
    Convert BMP image files to NIfTI (.nii.gz) format.
    This function converts 2D BMP images to 3D NIfTI volumes with specified slice orientation.
    The output NIfTI files will have RAS+ orientation with specified voxel size.
    Args:
        bmp_dir (str): Input directory containing BMP files to convert
        niigz_dir (str): Output directory where NIfTI files will be saved
        slice_dim_type (int): Slice dimension/orientation type:
            0: Sagittal (YZ plane)
            1: Coronal (XZ plane)
            2: Axial (XY plane)
        pseudo_voxel_size (list): List of 3 floats specifying voxel dimensions in mm [x,y,z]
        flip_dim0 (bool, optional): If True, flip image along dimension 0. Defaults to False.
        flip_dim1 (bool, optional): If True, flip image along dimension 1. Defaults to False.
        swap_dim01 (bool, optional): If True, swap dimensions 0 and 1. Defaults to False.
    Returns:
        tuple: Original image dimensions (height, width) of the first converted BMP
    """

    # Validate slice_dim_type
    if slice_dim_type not in [0, 1, 2]:
        raise ValueError("slice_dim_type must be 0, 1, or 2")

    # Convert pseudo_voxel_size to list if it's not already
    pseudo_voxel_size = list(pseudo_voxel_size)

    # Create output directory
    Path(niigz_dir).mkdir(parents=True, exist_ok=True)

    # Get all BMP files
    bmp_files = list(Path(bmp_dir).glob("*.bmp"))
    print(f"Found {len(bmp_files)} .bmp files")

    for bmp_file in bmp_files:
        try:
            print(f"Converting {bmp_file.name}")

            # Read BMP image
            img_2d = cv2.imread(str(bmp_file), cv2.IMREAD_GRAYSCALE)
            img_size_dim0, img_size_dim1 = img_2d.shape

            # Note: this is definitely correct, DO NOT SWAP the order of transformations
            if flip_dim0:
                img_2d = cv2.flip(img_2d, 0)  # 0 means flip vertically
            if flip_dim1:
                img_2d = cv2.flip(img_2d, 1)  # 1 means flip horizontally
            if swap_dim01:  # this line should be AFTER slip_x and slip_y
                img_2d = np.swapaxes(img_2d, 0, 1)

            # Create 3D array based on slice_dim_type
            if slice_dim_type == 0:  # Sagittal (YZ plane)
                img_3d = np.zeros(
                    (1, img_2d.shape[0], img_2d.shape[1]), dtype=img_2d.dtype
                )
                img_3d[0, :, :] = img_2d
            elif slice_dim_type == 1:  # Coronal (XZ plane)
                img_3d = np.zeros(
                    (img_2d.shape[0], 1, img_2d.shape[1]), dtype=img_2d.dtype
                )
                img_3d[:, 0, :] = img_2d
            else:  # Axial (XY plane)
                img_3d = np.zeros(
                    (img_2d.shape[0], img_2d.shape[1], 1), dtype=img_2d.dtype
                )
                img_3d[:, :, 0] = img_2d

            # Create affine matrix for RAS+ orientation
            # Set voxel size to 0.1mm in all dimensions
            affine = np.diag(pseudo_voxel_size + [1])

            # Create NIfTI image
            nii_img = nib.Nifti1Image(img_3d, affine)

            # Set header information
            nii_img.header.set_zooms(pseudo_voxel_size)

            # Save as NIfTI file
            output_file = Path(niigz_dir) / f"{bmp_file.stem}.nii.gz"
            nib.save(nii_img, str(output_file))
            print(f"Saved to {output_file}")

        except Exception as e:
            print(f"Error converting {bmp_file.name}: {e}")

    return img_size_dim0, img_size_dim1


def convert_jpg_to_niigz(
    jpg_dir,
    niigz_dir,
    slice_dim_type,
    pseudo_voxel_size,
    flip_dim0=False,
    flip_dim1=False,
    swap_dim01=False,
):
    """
    Convert JPG image files to NIfTI (.nii.gz) format.
    This function converts 2D JPG images to 3D NIfTI volumes with specified slice orientation.
    The output NIfTI files will have RAS+ orientation with specified voxel size.
    Args:
        jpg_dir (str): Input directory containing JPG files to convert
        niigz_dir (str): Output directory where NIfTI files will be saved
        slice_dim_type (int): Slice dimension/orientation type:
            0: Sagittal (YZ plane)
            1: Coronal (XZ plane)
            2: Axial (XY plane)
        pseudo_voxel_size (list): List of 3 floats specifying voxel dimensions in mm [x,y,z]
        flip_dim0 (bool, optional): If True, flip image along dimension 0. Defaults to False.
        flip_dim1 (bool, optional): If True, flip image along dimension 1. Defaults to False.
        swap_dim01 (bool, optional): If True, swap dimensions 0 and 1. Defaults to False.
    Returns:
        tuple: Original image dimensions (height, width) of the first converted JPG
    """

    # Validate slice_dim_type
    if slice_dim_type not in [0, 1, 2]:
        raise ValueError("slice_dim_type must be 0, 1, or 2")

    # Convert pseudo_voxel_size to list if it's not already
    pseudo_voxel_size = list(pseudo_voxel_size)

    # Create output directory
    Path(niigz_dir).mkdir(parents=True, exist_ok=True)

    # Get all JPG files
    jpg_files = list(Path(jpg_dir).glob("*.jpg"))
    print(f"Found {len(jpg_files)} .jpg files")

    for jpg_file in jpg_files:
        try:
            print(f"Converting {jpg_file.name}")

            # Read JPG image
            img_2d = cv2.imread(str(jpg_file), cv2.IMREAD_GRAYSCALE)
            img_size_dim0, img_size_dim1 = img_2d.shape

            # Note: this is definitely correct, DO NOT SWAP the order of transformations
            if flip_dim0:
                img_2d = cv2.flip(img_2d, 0)  # 0 means flip vertically
            if flip_dim1:
                img_2d = cv2.flip(img_2d, 1)  # 1 means flip horizontally
            if swap_dim01:  # this line should be AFTER flip_dim0 and flip_dim1
                img_2d = np.swapaxes(img_2d, 0, 1)

            # Create 3D array based on slice_dim_type
            if slice_dim_type == 0:  # Sagittal (YZ plane)
                img_3d = np.zeros(
                    (1, img_2d.shape[0], img_2d.shape[1]), dtype=img_2d.dtype
                )
                img_3d[0, :, :] = img_2d
            elif slice_dim_type == 1:  # Coronal (XZ plane)
                img_3d = np.zeros(
                    (img_2d.shape[0], 1, img_2d.shape[1]), dtype=img_2d.dtype
                )
                img_3d[:, 0, :] = img_2d
            else:  # Axial (XY plane)
                img_3d = np.zeros(
                    (img_2d.shape[0], img_2d.shape[1], 1), dtype=img_2d.dtype
                )
                img_3d[:, :, 0] = img_2d

            # Create affine matrix for RAS+ orientation
            # Set voxel size to 0.1mm in all dimensions
            affine = np.diag(pseudo_voxel_size + [1])

            # Create NIfTI image
            nii_img = nib.Nifti1Image(img_3d, affine)

            # Set header information
            nii_img.header.set_zooms(pseudo_voxel_size)

            # Save as NIfTI file
            output_file = Path(niigz_dir) / f"{jpg_file.stem}.nii.gz"
            nib.save(nii_img, str(output_file))
            print(f"Saved to {output_file}")

        except Exception as e:
            print(f"Error converting {jpg_file.name}: {e}")

    return img_size_dim0, img_size_dim1
