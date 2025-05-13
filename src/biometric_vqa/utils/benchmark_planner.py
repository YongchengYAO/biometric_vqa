import os
import json
import nibabel as nib
import numpy as np
import random
import glob
import sys
import cv2
import gzip
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import label, find_objects
from abc import ABC, abstractmethod
from biometric_vqa import __version__
from biometric_vqa.utils.preprocess_utils import convert_to_serializable
from biometric_vqa.utils.data_conversion import convert_mask_to_uint16_per_dir


class BiometricVQA_BenchmarkPlannerBase(ABC):
    def __init__(
        self,
        dataset_dir,
        bm_plan,
        dataset_name,
        seed=1024,
        split_ratio=0.7,
    ):
        self.version = __version__
        self.dataset_dir = dataset_dir
        self.bm_plan = bm_plan
        self.dataset_name = dataset_name
        self.seed = seed
        self.split_ratio = split_ratio

    @property
    @abstractmethod
    def task_type(self):
        """Abstract property that must return the type of task"""
        pass

    @property
    @abstractmethod
    def bm_plan_file(self):
        """Abstract property that must return the benchmark plan file path"""
        pass

    def _split_niigz_dataset(self, folder_path):
        # Set random seed for reproducibility
        random.seed(self.seed)
        # Split dataset into training and testing sets
        file_list = [
            os.path.basename(f)
            for f in glob.glob(os.path.join(folder_path, "*.nii.gz"))
        ]
        random.shuffle(file_list)
        split_idx = int(len(file_list) * self.split_ratio)
        train_ls = file_list[:split_idx]
        test_ls = file_list[split_idx:]
        return train_ls, test_ls

    def _reorient_niigz_RASplus(self, nifti_path, output_path):
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
        new_img = nib.Nifti1Image(
            reoriented_data, canonical_img.affine, header=img.header
        )
        # Preserve original header information where possible
        new_img.header.set_data_dtype(original_dtype)
        # Save the reoriented image
        nib.save(new_img, output_path)
        print(
            f"Converted {nifti_path} to RAS+ orientation and saved as {output_path}.\n"
        )

    def _reorient_niigz_RASplus_batch_inplace(self):
        """
        Reorient all NIfTI files in a directory and its subdirectories to RAS+ orientation in place.
        This function modifies the original files rather than creating new ones.
        """
        # Find all .nii.gz files recursively in directory
        nii_files = list(glob.glob(f"{self.dataset_dir}/**/*.nii.gz", recursive=True))
        print(f"Reorienting {len(nii_files)} files to RAS+ orientation...\n")
        # Process each file
        for i, nii_file in enumerate(nii_files, 1):
            print(f" - [{i}/{len(nii_files)}] Processing: {os.path.basename(nii_file)}")
            # Reorient file and overwrite the original
            self._reorient_niigz_RASplus(nii_file, nii_file)

    def update_tasks_number(self):
        self.bm_plan["tasks_number"] = len(self.bm_plan["tasks"])
        print(f"Updated tasks_number to {self.bm_plan['tasks_number']}")

    def reorient_niigz_RASPlus(self):
        print(
            f"Reorienting images and masks to RAS+ orientation for {self.dataset_name}...\n"
        )
        self._reorient_niigz_RASplus_batch_inplace()

    def save_benchmark_plan(self):
        print("Saving benchmark plan...\n")
        if self.bm_plan_file.endswith(".json.gz"):
            with gzip.open(self.bm_plan_file, "wt") as f:
                json.dump(
                    self.bm_plan, f, indent=4, default=convert_to_serializable
                )
        else:
            with open(self.bm_plan_file, "w") as f:
                json.dump(
                    self.bm_plan, f, indent=4, default=convert_to_serializable
                )
        print(f"Benchmark plan saved to {self.bm_plan_file}.\n")
        print(f"Dataset preprocessing for {self.dataset_name} completed.\n")

    @abstractmethod
    def process_each_task(self):
        """
        Placeholder method to be implemented by child classes

        example:

        # Process each task in the benchmark plan
        for task_idx, task in enumerate(self.bm_plan["tasks"], 1):
            print(
                f"{'='*50}\nProcessing {self.task_type} task {task_idx}/{len(self.bm_plan['tasks'])}\n{'='*50}"
            )
            # Update task ID
            task["task_ID"] = f"{task_idx:02d}"
            # Split the dataset into training and testing sets
            print("Splitting dataset into training and testing sets...")
            imgs_tr, imgs_ts = self._split_niigz_dataset(task["image_folder"])
            print(
                f"Split complete: {len(imgs_tr)} training, {len(imgs_ts)} testing cases\n"
            )
            # Update the profile of the training and testing sets
            print("Updating profiles for training set...\n")
            task = self._update_cases_profile(imgs_tr, task, "train")
            print("Updating profiles for testing set...\n")
            task = self._update_cases_profile(imgs_ts, task, "test")
            print(f"Finished processing task {task_idx}\n{'='*50}\n\n")
        """
        pass

    @abstractmethod
    def process(self):
        """
        Placeholder method to be implemented by child classes

        example:

        print(f"Preprocessing {self.dataset_name} dataset in {self.dataset_dir}...\n")
        self.update_tasks_number()
        if self.reorient2RAS:
            self.reorient_niigz_RASPlus()
        self.process_each_task()
        self.save_benchmark_plan()
        """
        pass

    @abstractmethod
    def _update_cases_profile(self):
        """Placeholder method to be implemented by child classes"""
        pass


class BiometricVQA_BenchmarkPlanner4SegDetect(BiometricVQA_BenchmarkPlannerBase):
    def __init__(
        self,
        dataset_dir,
        bm_plan,
        dataset_name,
        seed=1024,
        split_ratio=0.7,
        force_uint16_mask=True,
        reorient2RAS=True,
    ):
        # Call parent class's __init__
        super().__init__(
            dataset_dir,
            bm_plan,
            dataset_name,
            seed,
            split_ratio,
        )

        # Add additional attributes specific to this class
        self.force_uint16_mask = force_uint16_mask
        self.reorient2RAS = reorient2RAS
        self.mask_folders = self._get_mask_folders()

    def _get_mask_folders(self):
        """Get unique mask folders from tasks"""
        mask_folders = []
        for task in self.bm_plan["tasks"]:
            mask_folders.append(task["mask_folder"])
        return list(set(mask_folders))

    def _find_labels_map(self, mask_folder):
        for task in self.bm_plan["tasks"]:
            if task["mask_folder"] == mask_folder:
                return task["labels_map"]

    def _match_mask_to_image(self, image_file, task_info):
        # Match the mask file with the image file
        image_prefix = task_info["image_prefix"]
        image_suffix = task_info["image_suffix"]
        mask_prefix = task_info["mask_prefix"]
        mask_suffix = task_info["mask_suffix"]
        image_folder = task_info["image_folder"]
        mask_folder = task_info["mask_folder"]
        caseID = (
            os.path.basename(image_file)
            .replace(image_prefix, "")
            .replace(image_suffix, "")
        )
        mask_path = f"{mask_folder}/{mask_prefix}{caseID}{mask_suffix}"
        image_path = f"{image_folder}/{image_file}"
        if not os.path.exists(mask_path):
            error_msg = (
                f"\n\nError: Missing mask file for the image {image_path}"
                f"Expected mask file: {mask_path}\n"
                "Check the 'mask_folder', 'mask_prefix' and 'mask_suffix' in the 'benchmark_plan' dictionary.\n\n"
            )
            raise FileNotFoundError(error_msg)
        else:
            print(f"Found a mask file for {caseID}")
            print(f" - Image file: {image_path}")
            print(f" - Mask file: {mask_path}\n")
            return caseID, image_path, mask_path

    def _check_nii_header_for_img_mask(self, image_file, task_info):
        # Match the mask file with the image file
        caseID, image_path, mask_path = self._match_mask_to_image(image_file, task_info)
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
        print(f"Checking properties for case: {caseID} ...")
        for key in mask_file_info:
            if isinstance(mask_file_info[key], np.ndarray):
                if not np.allclose(
                    mask_file_info[key], image_file_info[key], atol=1e-5, rtol=1e-3
                ):
                    raise ValueError(
                        f"\n\nMismatch in {key} between image and mask for case {caseID}:\n"
                        f"Image {key}:\n{image_file_info[key]}\n"
                        f"Mask {key}:\n{mask_file_info[key]}\n"
                    )
            elif mask_file_info[key] != image_file_info[key]:
                raise ValueError(
                    f"\n\nMismatch in {key} between image and mask for case {caseID}:\n"
                    f"Image {key}:\n{image_file_info[key]}\n"
                    f"Mask {key}:\n{mask_file_info[key]}\n"
                )
        print(f"Properties (NIfTI file header) match!\n")
        return (
            caseID,
            mask_nii,
            mask_data,
            image_path,
            mask_path,
            image_file_info,
            mask_file_info,
        )

    def _validate_segmentation_labels_per_dir(self, mask_folder):
        labels_map = self._find_labels_map(self, mask_folder)
        if not labels_map:
            print("\n\nError: labels_map is empty!\n")
            sys.exit(1)
        # Get list of keys from labels_map
        valid_labels = np.array(list(labels_map.keys()))
        print(f"Valid labels from labels_map: {valid_labels}")
        # Check each .nii.gz file
        total_files = sum(
            1 for _ in glob.glob(f"{mask_folder}/**/*.nii.gz", recursive=True)
        )
        processed = 0
        for file_path in glob.glob(f"{mask_folder}/**/*.nii.gz", recursive=True):
            processed += 1
            print(
                f" - [{processed}/{total_files}] Checking: {os.path.basename(file_path)}"
            )
            try:
                img = nib.load(file_path)
                data = img.get_fdata()
                unique_vals = np.unique(
                    data[data != 0]
                )  # Get non-zero values directly as numbers
                invalid_mask = ~np.isin(
                    unique_vals, valid_labels.astype(unique_vals.dtype)
                )
                invalid_labels = unique_vals[invalid_mask]
                if len(invalid_labels) > 0:
                    print(f"\nError in file: {file_path}")
                    print(f"Found invalid labels: {invalid_labels}")
                    print(f"Valid labels are: {valid_labels}")
                    sys.exit(1)
            except Exception as e:
                print(f"\nError processing {file_path}: {str(e)}")
                sys.exit(1)

        print("\nAll files contain valid labels!")

    def validate_segmentation_labels(self):
        print(f"Validating segmentation mask labels for {self.dataset_name}...\n")
        for folder in self.mask_folders:
            self._validate_segmentation_labels_per_dir(self, folder)

    def convert_masks_to_uint16(self):
        print(f"Enforcing integers in masks for {self.dataset_name}...\n")
        for folder in self.mask_folders:
            mask_folder = os.path.join(self.dataset_dir, folder)
            convert_mask_to_uint16_per_dir(mask_folder)

    def process_each_task(self):
        # Process each task in the benchmark plan
        for task_idx, task in enumerate(self.bm_plan["tasks"], 1):
            print(
                f"{'='*50}\nProcessing {self.task_type} task {task_idx}/{len(self.bm_plan['tasks'])}\n{'='*50}"
            )
            # Update task ID
            task["task_ID"] = f"{task_idx:02d}"
            # Split the dataset into training and testing sets
            print("Splitting dataset into training and testing sets...")
            imgs_tr, imgs_ts = self._split_niigz_dataset(task["image_folder"])
            print(
                f"Split complete: {len(imgs_tr)} training, {len(imgs_ts)} testing cases\n"
            )
            # Update the profile of the training and testing sets
            print("Updating profiles for training set...\n")
            self._update_cases_profile(imgs_tr, task, "train")
            print("Updating profiles for testing set...\n")
            self._update_cases_profile(imgs_ts, task, "test")
            print(f"Finished processing task {task_idx}\n{'='*50}\n\n")

    def process(self):
        print(f"Preprocessing {self.dataset_name} dataset in {self.dataset_dir}...\n")
        self.update_tasks_number()
        if self.force_uint16_mask:
            self.convert_masks_to_uint16()
        if self.reorient2RAS:
            self.reorient_niigz_RASPlus()
        self.process_each_task()
        self.save_benchmark_plan()

    def _update_cases_profile(self):
        """Placeholder method to be implemented by child classes"""
        pass


class BiometricVQA_BenchmarkPlannerSegmentation(
    BiometricVQA_BenchmarkPlanner4SegDetect
):
    def __init__(
        self,
        dataset_dir,
        bm_plan,
        dataset_name,
        seed=1024,
        split_ratio=0.7,
        force_uint16_mask=True,
        reorient2RAS=True,
    ):
        # Call parent class's __init__
        super().__init__(
            dataset_dir,
            bm_plan,
            dataset_name,
            seed,
            split_ratio,
            force_uint16_mask,
            reorient2RAS,
        )

    @property
    def task_type(self):
        return "segmentation"

    @property
    def bm_plan_file(self):
        return os.path.join(
            self.dataset_dir, f"benchmark_plan_segmentation_v{self.version}.json.gz"
        )

    # Only used for getting the file name in Huggingface data loading script
    @classmethod
    def get_bm_plan_file(cls, dataset_dir, version):
        return os.path.join(
            dataset_dir, f"benchmark_plan_segmentation_v{version}.json.gz"
        )

    def __inspect_slices(self, profile, idx, slice_vals, unit_area):
        mask = slice_vals[0] > 0
        labels = slice_vals[0][mask]
        counts = slice_vals[1][mask]
        if len(labels) > 0:
            slice_profile = [
                {
                    "label": label,
                    "pixel_count": int(count),
                    "ROI_area": count * unit_area,
                }
                for label, count in zip(labels, counts)
            ]
            profile.append({"slice_idx": idx, "slice_profile": slice_profile})
        return profile

    def _update_cases_profile(self, images_list, task_info, split):
        if split not in ["train", "test"]:
            raise ValueError('\n\nError: split should be one of "train" or "test"\n\n')
        task_info["task_type"] = self.task_type
        task_info[f"{split}_cases_number"] = len(images_list)
        for i, img_file in enumerate(images_list, 1):
            print(
                f"{'-'*50}\n[{i}/{len(images_list)}] Processing: {os.path.basename(img_file)}\n{'-'*50}"
            )
            # Check if mask and image properties match
            (
                caseID,
                mask_nii,
                mask_data,
                image_path,
                mask_path,
                image_file_info,
                mask_file_info,
            ) = self._check_nii_header_for_img_mask(img_file, task_info)
            # Find non-zero slices in each dimension
            print(f"Updating profile for case: {caseID} ...")
            profile_per_slice_x = []
            profile_per_slice_y = []
            profile_per_slice_z = []
            voxel_sizes = mask_nii.header.get_zooms()
            # For x dimension
            print(" - Inspecting sagittal slices (slices along x-dimension) ...")
            unit_area_x = voxel_sizes[1] * voxel_sizes[2]
            for i in range(mask_data.shape[0]):
                slice_data_x = mask_data[i, :, :]
                if slice_data_x.ndim != 2 or 1 in slice_data_x.shape:
                    print(
                        f"\n\nWarning: Expected 2D sagittal slice but got shape {slice_data_x.shape} for the {i}-th slice along the x-dimension\n"
                    )
                else:
                    slice_vals = np.unique(slice_data_x, return_counts=True)
                    profile_per_slice_x = self.__inspect_slices(
                        profile_per_slice_x, i, slice_vals, unit_area_x
                    )
            # For y dimension
            print(" - Inspecting coronal slices (slices along y-dimension) ...")
            unit_area_y = voxel_sizes[0] * voxel_sizes[2]
            for i in range(mask_data.shape[1]):
                slice_data_y = mask_data[:, i, :]
                if slice_data_y.ndim != 2 or 1 in slice_data_y.shape:
                    print(
                        f"\n\nWarning: Expected 2D coronal slice but got shape {slice_data_y.shape} for the {i}-th slice along the y-dimension\n"
                    )
                else:
                    slice_vals = np.unique(slice_data_y, return_counts=True)
                    profile_per_slice_y = self.__inspect_slices(
                        profile_per_slice_y, i, slice_vals, unit_area_y
                    )
            # For z dimension
            print(" - Inspecting axial slices (slices along z-dimension) ...")
            unit_area_z = voxel_sizes[0] * voxel_sizes[1]
            for i in range(mask_data.shape[2]):
                slice_data_z = mask_data[:, :, i]
                if slice_data_z.ndim != 2 or 1 in slice_data_z.shape:
                    print(
                        f"\n\nWarning: Expected 2D axial slice but got shape {slice_data_z.shape} for the {i}-th slice along the z-dimension\n"
                    )
                else:
                    slice_vals = np.unique(slice_data_z, return_counts=True)
                    profile_per_slice_z = self.__inspect_slices(
                        profile_per_slice_z, i, slice_vals, unit_area_z
                    )
            # Update the cases profile
            if f"{split}_cases" not in task_info:
                task_info[f"{split}_cases"] = []
            task_info[f"{split}_cases"].append(
                {
                    "case_ID": caseID,
                    "image_file": image_path,
                    "mask_file": mask_path,
                    "image_file_info": image_file_info,
                    "mask_file_info": mask_file_info,
                    "slice_profiles_x": profile_per_slice_x,
                    "slice_profiles_y": profile_per_slice_y,
                    "slice_profiles_z": profile_per_slice_z,
                }
            )
            print(f"\nProfile updated for case {caseID}!\n{'-'*50}\n")

    @staticmethod
    def flatten_slice_profiles_2d(cases, slice_dim):
        if slice_dim == 0:
            slice_profiles_key = "slice_profiles_x"
        elif slice_dim == 1:
            slice_profiles_key = "slice_profiles_y"
        elif slice_dim == 2:
            slice_profiles_key = "slice_profiles_z"
        else:
            raise ValueError(f"\nError: slice_dim should be one of 0, 1, or 2\n")
        flatten_slice_profile = []
        for case in cases:
            mask_file = case.get("mask_file")
            image_file = case.get("image_file")
            slice_profiles_dirX = case.get(slice_profiles_key, [])
            for slice_profiles in slice_profiles_dirX:
                slice_idx = slice_profiles.get("slice_idx")
                slice_profile = slice_profiles.get("slice_profile", [])
                for profile in slice_profile:
                    label = profile.get("label")
                    pixel_count = profile.get("pixel_count")
                    roi_area = profile.get("ROI_area")
                    flatten_slice_profile.append(
                        {
                            "image_file": image_file,
                            "mask_file": mask_file,
                            "slice_dim": slice_dim,
                            "slice_idx": slice_idx,
                            "label": label,
                            "pixel_count": pixel_count,
                            "ROI_area": roi_area,
                        }
                    )
        return flatten_slice_profile


class BiometricVQA_BenchmarkPlannerDetection(BiometricVQA_BenchmarkPlanner4SegDetect):
    def __init__(
        self,
        dataset_dir,
        bm_plan,
        dataset_name,
        seed=1024,
        split_ratio=0.7,
        force_uint16_mask=True,
        reorient2RAS=True,
    ):
        # Call parent class's __init__
        super().__init__(
            dataset_dir,
            bm_plan,
            dataset_name,
            seed,
            split_ratio,
            force_uint16_mask,
            reorient2RAS,
        )

    @property
    def task_type(self):
        return "detection"

    @property
    def bm_plan_file(self):
        return os.path.join(
            self.dataset_dir, f"benchmark_plan_detection_v{self.version}.json.gz"
        )

    # Only used for getting the file name in Huggingface data loading script
    @classmethod
    def get_bm_plan_file(cls, dataset_dir, version):
        return os.path.join(dataset_dir, f"benchmark_plan_detection_v{version}.json.gz")

    def _find_bounding_boxes_2D(self, binary_mask, pixel_spacing):
        """
        Finds 2D bounding boxes for connected components in a binary mask.
        Args:
            binary_mask (np.ndarray): 2D binary mask array
            pixel_spacing (tuple): Physical spacing between pixels (dim1_spacing, dim2_spacing)
        Returns:
            list[dict]: List of bounding boxes, each containing:
                - min_coords: (dim1_min, dim2_min)
                - max_coords: (dim1_max, dim2_max)
                - center_coords: (dim1_center, dim2_center)
                - dimensions: (dim1_length, dim2_length) in pixels
                - sizes: (dim1_size, dim2_size) in physical units
        Raises:
            ValueError: If mask is empty or not 2D
        """
        # Input validation
        if binary_mask.ndim != 2:
            raise ValueError(f"Expected 2D array, got {binary_mask.ndim}D array")
        if binary_mask.sum() == 0:
            raise ValueError("Empty mask - no objects found")
        # Label connected components
        labeled_array, num_objects = label(binary_mask)
        bboxes = []
        # Process each object
        for object_id in range(1, num_objects + 1):
            # Create mask for this object
            object_mask = labeled_array == object_id
            # Get bounding box using find_objects
            slices = find_objects(object_mask)[0]
            # Extract coordinates
            dim1_min, dim1_max = slices[0].start, slices[0].stop - 1
            dim2_min, dim2_max = slices[1].start, slices[1].stop - 1
            # Calculate center coordinates
            dim1_center = int((dim1_min + dim1_max) / 2)
            dim2_center = int((dim2_min + dim2_max) / 2)
            # Calculate dimensions
            dim1_length = dim1_max - dim1_min + 1
            dim2_length = dim2_max - dim2_min + 1
            bbox_info = {
                "min_coords": (int(dim1_min), int(dim2_min)),
                "max_coords": (int(dim1_max), int(dim2_max)),
                "center_coords": (dim1_center, dim2_center),
                "dimensions": (dim1_length, dim2_length),
                "sizes": (
                    dim1_length * pixel_spacing[0],
                    dim2_length * pixel_spacing[1],
                ),
            }
            bboxes.append(bbox_info)
        return bboxes

    def _find_bounding_boxes_3D(self, binary_mask, voxel_spacing):
        """
        Finds 3D bounding boxes for connected components in a binary mask.
        Args:
            binary_mask (np.ndarray): 3D binary mask array
            voxel_spacing (tuple): Physical spacing between voxels (dim1_spacing, dim2_spacing, dim3_spacing)
        Returns:
            list[dict]: List of bounding boxes, each containing:
                - min_coords: (dim1_min, dim2_min, dim3_min)
                - max_coords: (dim1_max, dim2_max, dim3_max)
                - center_coords: (dim1_center, dim2_center, dim3_center)
                - dimensions: (dim1_length, dim2_length, dim3_length) in voxels
                - sizes: (dim1_size, dim2_size, dim3_size) in physical units
        Raises:
            ValueError: If mask is empty or not 3D
        """
        # Input validation
        if binary_mask.ndim != 3:
            raise ValueError(f"Expected 3D array, got {binary_mask.ndim}D array")
        if binary_mask.sum() == 0:
            raise ValueError("Empty mask - no non-zero elements found")
        # Label connected components
        labeled_array, num_clusters = label(binary_mask)
        bboxes = []
        # Process each cluster
        for cluster_id in range(1, num_clusters + 1):
            # Create mask for this cluster
            cluster_mask = labeled_array == cluster_id
            # Get bounding box using find_objects
            obj_info = find_objects(cluster_mask)[0]
            # Extract coordinates
            dim1_min, dim1_max = obj_info[0].start, obj_info[0].stop - 1
            dim2_min, dim2_max = obj_info[1].start, obj_info[1].stop - 1
            dim3_min, dim3_max = obj_info[2].start, obj_info[2].stop - 1
            # Calculate center coordinates
            dim1_center = int((dim1_min + dim1_max) / 2)
            dim2_center = int((dim2_min + dim2_max) / 2)
            dim3_center = int((dim3_min + dim3_max) / 2)
            # Calculate dimensions
            dim1_length = dim1_max - dim1_min + 1
            dim2_length = dim2_max - dim2_min + 1
            dim3_length = dim3_max - dim3_min + 1
            bbox_info = {
                "min_coords": (int(dim1_min), int(dim2_min), int(dim3_min)),
                "max_coords": (int(dim1_max), int(dim2_max), int(dim3_max)),
                "center_coords": (dim1_center, dim2_center, dim3_center),
                "dimensions": (dim1_length, dim2_length, dim3_length),
                "sizes": (
                    dim1_length * voxel_spacing[0],
                    dim2_length * voxel_spacing[1],
                    dim3_length * voxel_spacing[2],
                ),
            }
            bboxes.append(bbox_info)
        return bboxes

    def _inspect_2D_slice(self, profile, idx, mask_2d, pixel_spacing):
        slice_vals = np.unique(mask_2d, return_counts=True)
        mask = slice_vals[0] > 0
        labels = slice_vals[0][mask]
        if len(labels) > 0:
            slice_profile = [
                {
                    "label": label,
                    "bboxes": self._find_bounding_boxes_2D(
                        mask_2d == label, pixel_spacing
                    ),
                }
                for label in labels
            ]
            profile.append({"slice_idx": idx, "slice_profile": slice_profile})
        return profile

    def _inspect_3D_image(self, mask_3d, voxel_spacing):
        profile_3D = []
        labels = np.unique(mask_3d)
        labels = labels[labels != 0]
        if len(labels) > 0:
            for label in labels:
                binary_mask_3d = mask_3d == label
                bboxes = self._find_bounding_boxes_3D(binary_mask_3d, voxel_spacing)
                profile_3D.append({"label": label, "bboxes": bboxes})
        return profile_3D

    def _update_cases_profile(self, images_list, task_info, split):
        if split not in ["train", "test"]:
            raise ValueError('\n\nError: split should be one of "train" or "test"\n\n')
        task_info["task_type"] = self.task_type
        task_info[f"{split}_cases_number"] = len(images_list)
        for i, img_file in enumerate(images_list, 1):
            print(
                f"{'-'*50}\n[{i}/{len(images_list)}] Processing: {os.path.basename(img_file)}\n{'-'*50}"
            )
            # Check if mask and image properties match
            (
                caseID,
                mask_nii,
                mask_3d,
                image_path,
                mask_path,
                image_file_info,
                mask_file_info,
            ) = self._check_nii_header_for_img_mask(img_file, task_info)
            print(f"Updating profile for case: {caseID} ...")
            voxel_sizes = mask_nii.header.get_zooms()
            # Find bounding boxes for 2D slices
            print(" - Bounding box inspection for 2D slices")
            profile_per_slice_x = []
            profile_per_slice_y = []
            profile_per_slice_z = []
            # For x dimension
            print("   - Inspecting sagittal slices (slices along x-dimension) ...")
            for i in range(mask_3d.shape[0]):
                mask_2d_x = mask_3d[i, :, :]
                if mask_2d_x.ndim != 2 or 1 in mask_2d_x.shape:
                    print(
                        f"\n\nWarning: Expected 2D sagittal slice but got shape {mask_2d_x.shape} for the {i}-th slice along the x-dimension\n"
                    )
                else:
                    profile_per_slice_x = self._inspect_2D_slice(
                        profile_per_slice_x,
                        i,
                        mask_2d_x,
                        (voxel_sizes[1], voxel_sizes[2]),
                    )
            # For y dimension
            print("   - Inspecting coronal slices (slices along y-dimension) ...")
            for i in range(mask_3d.shape[1]):
                mask_2d_y = mask_3d[:, i, :]
                if mask_2d_y.ndim != 2 or 1 in mask_2d_y.shape:
                    print(
                        f"\n\nWarning: Expected 2D coronal slice but got shape {mask_2d_y.shape} for the {i}-th slice along the y-dimension\n"
                    )
                else:
                    profile_per_slice_y = self._inspect_2D_slice(
                        profile_per_slice_y,
                        i,
                        mask_2d_y,
                        (voxel_sizes[0], voxel_sizes[2]),
                    )
            # For z dimension
            print("   - Inspecting axial slices (slices along z-dimension) ...")
            for i in range(mask_3d.shape[2]):
                mask_2d_z = mask_3d[:, :, i]
                if mask_2d_z.ndim != 2 or 1 in mask_2d_z.shape:
                    print(
                        f"\n\nWarning: Expected 2D axial slice but got shape {mask_2d_z.shape} for the {i}-th slice along the z-dimension\n"
                    )
                else:
                    profile_per_slice_z = self._inspect_2D_slice(
                        profile_per_slice_z,
                        i,
                        mask_2d_z,
                        (voxel_sizes[0], voxel_sizes[1]),
                    )
            # Find bounding boxes for 3D objects
            print(" - Bounding box inspection for 3D images")
            profile_3D = self._inspect_3D_image(mask_3d, voxel_sizes)
            # Update the cases profile
            if f"{split}_cases" not in task_info:
                task_info[f"{split}_cases"] = []
            task_info[f"{split}_cases"].append(
                {
                    "case_ID": caseID,
                    "image_file": image_path,
                    "mask_file": mask_path,
                    "image_file_info": image_file_info,
                    "mask_file_info": mask_file_info,
                    "slice_profiles_x": profile_per_slice_x,
                    "slice_profiles_y": profile_per_slice_y,
                    "slice_profiles_z": profile_per_slice_z,
                    "profile_3D": profile_3D,
                }
            )
            print(f"\nProfile updated for case {caseID}!\n{'-'*50}\n")

    @staticmethod
    def flatten_slice_profiles_2d(cases, slice_dim):
        if slice_dim == 0:
            slice_profiles_key = "slice_profiles_x"
        elif slice_dim == 1:
            slice_profiles_key = "slice_profiles_y"
        elif slice_dim == 2:
            slice_profiles_key = "slice_profiles_z"
        else:
            raise ValueError(f"\nError: slice_dim should be one of 0, 1, or 2\n")
        flatten_slice_profile = []
        for case in cases:
            mask_file = case.get("mask_file")
            image_file = case.get("image_file")
            slice_profiles_dirX = case.get(slice_profiles_key, [])
            for slice_profiles in slice_profiles_dirX:
                slice_idx = slice_profiles.get("slice_idx")
                slice_profile = slice_profiles.get("slice_profile", [])
                for profile in slice_profile:
                    label = profile.get("label")
                    bboxes = profile.get("bboxes")
                    flatten_slice_profile.append(
                        {
                            "image_file": image_file,
                            "mask_file": mask_file,
                            "slice_dim": np.uint8(slice_dim),
                            "slice_idx": np.uint16(slice_idx),
                            "label": np.uint16(label),
                            "bounding_boxes": bboxes,
                        }
                    )
        return flatten_slice_profile


class BiometricVQA_BenchmarkPlannerBiometry(BiometricVQA_BenchmarkPlannerBase):
    def __init__(
        self,
        dataset_dir,
        bm_plan,
        dataset_name,
        seed=1024,
        split_ratio=0.7,
    ):
        # Call parent class's __init__
        super().__init__(
            dataset_dir,
            bm_plan,
            dataset_name,
            seed,
            split_ratio,
        )

    @property
    def task_type(self):
        return "biometry"

    @property
    def bm_plan_file(self):
        return os.path.join(
            self.dataset_dir, f"benchmark_plan_biometry_v{self.version}.json.gz"
        )

    # Only used for getting the file name in Huggingface data loading script
    @classmethod
    def get_bm_plan_file(cls, dataset_dir, version):
        return os.path.join(dataset_dir, f"benchmark_plan_biometry_v{version}.json.gz")

    def _match_landmark_to_image(self, image_file, task_info):
        # Match the mask file with the image file
        image_prefix = task_info["image_prefix"]
        image_suffix = task_info["image_suffix"]
        landmark_prefix = task_info["landmark_prefix"]
        landmark_suffix = task_info["landmark_suffix"]
        image_folder = task_info["image_folder"]
        landmark_folder = task_info["landmark_folder"]
        caseID = (
            os.path.basename(image_file)
            .replace(image_prefix, "")
            .replace(image_suffix, "")
        )
        landmark_path = f"{landmark_folder}/{landmark_prefix}{caseID}{landmark_suffix}"
        image_path = f"{image_folder}/{image_file}"
        if not os.path.exists(landmark_path):
            error_msg = (
                f"\n\nError: Missing landmarks file for the image {image_path}\n"
                f"Expected landmark file: {landmark_path}\n"
                "Check the 'landmark_folder', 'landmark_prefix' and 'landmark_suffix' in the 'benchmark_plan' dictionary.\n\n"
            )
            raise FileNotFoundError(error_msg)
        else:
            print(f"Found a landmark file for {caseID}")
            print(f" - Image file: {image_path}")
            print(f" - Landmark file: {landmark_path}\n")
            return caseID, image_path, landmark_path

    def _check_landmarks_number(self, image_file, task_info):
        # Match the landmark file with the image file
        caseID, image_path, landmark_path = self._match_landmark_to_image(
            image_file, task_info
        )
        # Load the landmarks
        if landmark_path.endswith(".json.gz"):
            with gzip.open(landmark_path, "rt") as f:
                landmark_json = json.load(f)
        else:  # Handle regular .json files
            with open(landmark_path, "r") as f:
                landmark_json = json.load(f)
        # Count landmarks
        point_ids = set()
        for orientation in landmark_json.keys():
            for slice_data in landmark_json[orientation]:
                point_ids.update(slice_data["landmarks"].keys())
        landmarks_num = len(point_ids)
        # Check the number of landmarks
        if len(task_info["landmarks_map"]) != landmarks_num:
            error_msg = (
                f"\n\nError: Number of landmarks in {landmark_path} does not match the expected number.\n"
                f"Expected: {len(task_info['landmarks_map'])}, Found: {landmarks_num}\n"
            )
            raise ValueError(error_msg)
        print(f"Number of landmarks in {landmark_path} matches the expected number.\n")
        return caseID, image_path, landmark_path, landmark_json

    def _cal_point2vec(self, point1, point2):
        return np.array(point2) - np.array(point1)

    def _cal_vec2acuteAngle(self, vec1, vec2):
        return np.arccos(
            np.abs(np.dot(vec1, vec2)) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        )

    def _cal_angle(
        self, metric_key, metric_map_name, task_info, landmarks, voxel_sizes, slice_dim
    ):
        # Get 2 lines that form the angle
        line1_key = task_info[metric_map_name][metric_key]["element_keys"][0]
        line2_key = task_info[metric_map_name][metric_key]["element_keys"][1]
        line_map_name = task_info[metric_map_name][metric_key]["element_map_name"]
        # Get the points that define the line1 and line2
        point1_line1_key = task_info[line_map_name][line1_key]["element_keys"][0]
        point2_line1_key = task_info[line_map_name][line1_key]["element_keys"][1]
        point1_line2_key = task_info[line_map_name][line2_key]["element_keys"][0]
        point2_line2_key = task_info[line_map_name][line2_key]["element_keys"][1]
        point1_line1 = landmarks[point1_line1_key] * np.array(voxel_sizes)
        point2_line1 = landmarks[point2_line1_key] * np.array(voxel_sizes)
        point1_line2 = landmarks[point1_line2_key] * np.array(voxel_sizes)
        point2_line2 = landmarks[point2_line2_key] * np.array(voxel_sizes)
        # Calculate the vectors for the line1 and line2
        vec1 = self._cal_point2vec(point1_line1, point2_line1)
        vec2 = self._cal_point2vec(point1_line2, point2_line2)
        # Calculate the acute angle between the lines
        angle = self._cal_vec2acuteAngle(vec1, vec2)
        # Get the slice index (where the measurement is made)
        slice_idx = landmarks[point1_line1_key][slice_dim]
        return np.degrees(angle), slice_idx

    def _cal_distance(
        self, metric_key, metric_map_name, task_info, landmarks, voxel_sizes, slice_dim
    ):
        # Get the points that define the distance
        point1_key = task_info[metric_map_name][metric_key]["element_keys"][0]
        point2_key = task_info[metric_map_name][metric_key]["element_keys"][1]
        point1 = landmarks[point1_key] * np.array(voxel_sizes)
        point2 = landmarks[point2_key] * np.array(voxel_sizes)
        # Calculate the distance between the points
        distance = np.linalg.norm(self._cal_point2vec(point1, point2))
        # Get the slice index (where the measurement is made)
        slice_idx = landmarks[point1_key][slice_dim]
        return distance, slice_idx

    def _update_cases_profile(self, images_list, task_info, split):
        if split not in ["train", "test"]:
            raise ValueError('\n\nError: split should be one of "train" or "test"\n\n')
        # Updata task type and case number
        task_info["task_type"] = self.task_type
        task_info[f"{split}_cases_number"] = len(images_list)
        for i, img_file in enumerate(images_list, 1):
            print(
                f"{'-'*50}\n[{i}/{len(images_list)}] Processing: {os.path.basename(img_file)}\n{'-'*50}"
            )
            # Find and check the landmarks
            caseID, image_path, landmark_path, landmarks_json = (
                self._check_landmarks_number(img_file, task_info)
            )
            # Get voxel size
            img_nii = nib.load(image_path)
            voxel_sizes = img_nii.header.get_zooms()
            # Update biometrics for this case
            print(f"Updating profile for case: {caseID} ...")
            slice_profiles_x = []
            slice_profiles_y = []
            slice_profiles_z = []
            for _, metric in enumerate(task_info["biometrics_map"], 1):
                metric_type = metric["metric_type"]
                metric_map_name = metric["metric_map_name"]
                metric_key = metric["metric_key"]
                slice_dim = metric["slice_dim"]
                if slice_dim == 0:
                    slice_landmarks = landmarks_json["slice_landmarks_x"]
                elif slice_dim == 1:
                    slice_landmarks = landmarks_json["slice_landmarks_y"]
                elif slice_dim == 2:
                    slice_landmarks = landmarks_json["slice_landmarks_z"]
                # Calculate the metric value
                if metric_type == "angle":
                    # Get the appropriate landmarks for this biometric measurement
                    line1_key = task_info[metric_map_name][metric_key]["element_keys"][
                        0
                    ]
                    line2_key = task_info[metric_map_name][metric_key]["element_keys"][
                        1
                    ]
                    line_map_name = task_info[metric_map_name][metric_key][
                        "element_map_name"
                    ]
                    point1_line1_key = task_info[line_map_name][line1_key][
                        "element_keys"
                    ][0]
                    point2_line1_key = task_info[line_map_name][line1_key][
                        "element_keys"
                    ][1]
                    point1_line2_key = task_info[line_map_name][line2_key][
                        "element_keys"
                    ][0]
                    point2_line2_key = task_info[line_map_name][line2_key][
                        "element_keys"
                    ][1]
                    for i, slice_data in enumerate(slice_landmarks):
                        if (
                            point1_line1_key in slice_data["landmarks"]
                            and point2_line1_key in slice_data["landmarks"]
                            and point1_line2_key in slice_data["landmarks"]
                            and point2_line2_key in slice_data["landmarks"]
                        ):
                            landmarks = slice_data["landmarks"]
                            break
                    metric_value, slice_idx = self._cal_angle(
                        metric_key,
                        metric_map_name,
                        task_info,
                        landmarks,
                        voxel_sizes,
                        slice_dim,
                    )
                    metric_unit = "degree"
                elif metric_type == "distance":
                    # Get the appropriate landmarks for this biometric measurement
                    point1_key = task_info[metric_map_name][metric_key]["element_keys"][
                        0
                    ]
                    point2_key = task_info[metric_map_name][metric_key]["element_keys"][
                        1
                    ]
                    for i, slice_data in enumerate(slice_landmarks):
                        if (
                            point1_key in slice_data["landmarks"]
                            and point2_key in slice_data["landmarks"]
                        ):
                            landmarks = slice_data["landmarks"]
                            break
                    metric_value, slice_idx = self._cal_distance(
                        metric_key,
                        metric_map_name,
                        task_info,
                        landmarks,
                        voxel_sizes,
                        slice_dim,
                    )
                    metric_unit = "mm"
                else:
                    raise ValueError(f"Invalid metric_type: {metric_type}")
                slice_profile = {
                    "metric_type": metric_type,
                    "metric_map_name": metric_map_name,
                    "metric_key": metric_key,
                    "metric_value": metric_value,
                    "metric_unit": metric_unit,
                    "slice_dim": slice_dim,
                }
                if slice_dim == 0:
                    slice_profiles_x.append(
                        {
                            "slice_idx": slice_idx,
                            "slice_profile": [slice_profile],
                        }
                    )
                elif slice_dim == 1:
                    slice_profiles_y.append(
                        {
                            "slice_idx": slice_idx,
                            "slice_profile": [slice_profile],
                        }
                    )
                elif slice_dim == 2:
                    slice_profiles_z.append(
                        {
                            "slice_idx": slice_idx,
                            "slice_profile": [slice_profile],
                        }
                    )
                else:
                    raise ValueError("Invalid slice dimension")
            # Update the cases profile
            if f"{split}_cases" not in task_info:
                task_info[f"{split}_cases"] = []
            task_info[f"{split}_cases"].append(
                {
                    "case_ID": caseID,
                    "image_file": image_path,
                    "landmark_file": landmark_path,
                    "slice_profiles_x": slice_profiles_x,
                    "slice_profiles_y": slice_profiles_y,
                    "slice_profiles_z": slice_profiles_z,
                }
            )
            print(f"\nProfile updated for case {caseID}!\n{'-'*50}\n")

    @staticmethod
    def flatten_slice_profiles_2d(cases, slice_dim):
        if slice_dim == 0:
            slice_profiles_key = "slice_profiles_x"
        elif slice_dim == 1:
            slice_profiles_key = "slice_profiles_y"
        elif slice_dim == 2:
            slice_profiles_key = "slice_profiles_z"
        else:
            raise ValueError(f"\nError: slice_dim should be one of 0, 1, or 2\n")
        flatten_slice_profile = []
        for case in cases:
            landmark_file = case.get("landmark_file")
            image_file = case.get("image_file")
            slice_profiles_dirX = case.get(slice_profiles_key, [])
            for slice_profiles in slice_profiles_dirX:
                slice_idx = slice_profiles.get("slice_idx")
                slice_profile = slice_profiles.get("slice_profile", [])
                for profile in slice_profile:
                    flatten_slice_profile.append(
                        {
                            "image_file": image_file,
                            "landmark_file": landmark_file,
                            "slice_dim": slice_dim,
                            "slice_idx": slice_idx,
                            "biometric_profile": profile,
                        }
                    )
        return flatten_slice_profile

    def process_each_task(self):
        # Process each task in the benchmark plan
        for task_idx, task in enumerate(self.bm_plan["tasks"], 1):
            print(
                f"{'='*50}\nProcessing {self.task_type} task {task_idx}/{len(self.bm_plan['tasks'])}\n{'='*50}"
            )
            # Update task ID
            task["task_ID"] = f"{task_idx:02d}"
            # Split the dataset into training and testing sets
            print("Splitting dataset into training and testing sets...")
            imgs_tr, imgs_ts = self._split_niigz_dataset(task["image_folder"])
            print(
                f"Split complete: {len(imgs_tr)} training, {len(imgs_ts)} testing cases\n"
            )
            # Update the profile of the training and testing sets
            print("Updating profiles for training set...\n")
            self._update_cases_profile(imgs_tr, task, "train")
            print("Updating profiles for testing set...\n")
            self._update_cases_profile(imgs_ts, task, "test")
            print(f"Finished processing task {task_idx}\n{'='*50}\n\n")

    def process(self):
        print(f"Preprocessing {self.dataset_name} dataset in {self.dataset_dir}...\n")
        self.update_tasks_number()
        self.process_each_task()
        self.save_benchmark_plan()


class BiometricVQA_BenchmarkPlannerBiometry_fromSeg(
    BiometricVQA_BenchmarkPlanner4SegDetect, BiometricVQA_BenchmarkPlannerBiometry
):
    def __init__(
        self,
        dataset_dir,
        bm_plan,
        dataset_name,
        seed=1024,
        split_ratio=0.7,
        shrunk_bbox_scale=0.9,
        enlarged_bbox_scale=1.1,
        force_uint16_mask=True,
        reorient2RAS=True,
        visualization=False,
    ):
        # Call parent class's __init__
        super().__init__(
            dataset_dir,
            bm_plan,
            dataset_name,
            seed,
            split_ratio,
            force_uint16_mask,
            reorient2RAS,
        )
        self.visualization = visualization
        self.shrunk_bbox_scale = shrunk_bbox_scale
        self.enlarged_bbox_scale = enlarged_bbox_scale

    @property
    def task_type(self):
        return "biometry"

    @property
    def bm_plan_file(self):
        return os.path.join(
            self.dataset_dir, f"benchmark_plan_biometry_v{self.version}.json.gz"
        )

    # Only used for getting the file name in Huggingface data loading script
    @classmethod
    def get_bm_plan_file(cls, dataset_dir, version):
        return os.path.join(dataset_dir, f"benchmark_plan_biometry_v{version}.json.gz")

    def _match_landmark_to_image_fromSeg(self, image_file, task_info):
        # Match the mask file with the image file
        caseID = (
            os.path.basename(image_file)
            .replace(task_info["image_prefix"], "")
            .replace(task_info["image_suffix"], "")
        )
        landmark_path = os.path.join(
            task_info["landmark_folder"],
            f"{task_info['landmark_prefix']}{caseID}{task_info['landmark_suffix']}",
        )
        image_path = f"{task_info['image_folder']}/{image_file}"
        if not os.path.exists(landmark_path):
            error_msg = (
                f"\n\nError: Missing landmarks file for the image {image_path}"
                f"Expected landmark file: {landmark_path}\n"
                "Check the 'landmark_folder', 'landmark_prefix' and 'landmark_suffix' in the 'benchmark_plan' dictionary.\n\n"
            )
            raise FileNotFoundError(error_msg)
        else:
            print(f"Found a landmark file for {caseID}")
            print(f" - Image file: {image_path}")
            print(f" - Landmark file: {landmark_path}\n")
            return caseID, image_path, landmark_path

    def _get_appropriate_scale(self, pixel_size, img_size, init_scale=10):
        """
        Calculate appropriate scale bar size in mm and pixels.
        Args:
            pixel_sizes (float): Size of one pixel in mm
            img_width (int): Image width in pixels
            img_height (int): Image height in pixels
            init_scale (int): Initial scale in mm (default 10mm)
        Returns:
            tuple: (scale_mm, scale_pixels) - Selected scale in mm and pixels
        """
        scales = [
            1,
            2,
            5,
            10,
            15,
            20,
            25,
            30,
            40,
            50,
            60,
            70,
            80,
            90,
            100,
        ]  # Standard scales in mm
        # Convert initial scale to pixels
        scale_pixels_num = int(init_scale / pixel_size)
        # Scale should be between 5% and 25% of smallest image dimension
        min_pixels = img_size * 0.05
        max_pixels = img_size * 0.25
        if scale_pixels_num < min_pixels:
            # Find next larger scale
            for scale in scales:
                if scale > init_scale:
                    return self._get_appropriate_scale(pixel_size, img_size, scale)
        elif scale_pixels_num > max_pixels:
            # Find next smaller scale
            for scale in reversed(scales):
                if scale < init_scale:
                    return self._get_appropriate_scale(pixel_size, img_size, scale)
        return init_scale, scale_pixels_num

    def _find_scaled_bounding_boxes_2D(self, binary_mask, scale):
        # Input validation
        if binary_mask.ndim != 2:
            raise ValueError(f"Expected 2D array, got {binary_mask.ndim}D array")
        if binary_mask.sum() == 0:
            raise ValueError("Empty mask - no objects found")
        if scale <= 0:
            raise ValueError(f"Invalid scale value: {scale}. It must be positive.")
        # Label connected components
        labeled_array, num_objects = label(binary_mask)
        bboxes = []
        # Process each object
        for object_id in range(1, num_objects + 1):
            # Create mask for this object
            object_mask = labeled_array == object_id
            # Get bounding box using find_objects
            slices = find_objects(object_mask)[0]
            # Get original bounding box coordinates
            dim0_min, dim0_max = slices[0].start, slices[0].stop - 1
            dim1_min, dim1_max = slices[1].start, slices[1].stop - 1
            # Calculate center coordinates
            dim0_center = (dim0_min + dim0_max) / 2
            dim1_center = (dim1_min + dim1_max) / 2
            # Calculate original dimensions
            dim0_length = dim0_max - dim0_min + 1
            dim1_length = dim1_max - dim1_min + 1
            # Calculate enlarged dimensions
            dim0_length_scaled = int(dim0_length * scale)
            dim1_length_scaled = int(dim1_length * scale)
            # Calculate new min/max coordinates while keeping center fixed
            dim0_min_scaled = int(dim0_center - dim0_length_scaled / 2)
            dim0_max_scaled = int(dim0_center + dim0_length_scaled / 2)
            dim1_min_scaled = int(dim1_center - dim1_length_scaled / 2)
            dim1_max_scaled = int(dim1_center + dim1_length_scaled / 2)
            # Clip to image boundaries
            dim0_min_scaled = max(0, dim0_min_scaled)
            dim0_max_scaled = min(binary_mask.shape[0] - 1, dim0_max_scaled)
            dim1_min_scaled = max(0, dim1_min_scaled)
            dim1_max_scaled = min(binary_mask.shape[1] - 1, dim1_max_scaled)
            bbox_info = {
                "min_coords": (int(dim0_min_scaled), int(dim1_min_scaled)),
                "max_coords": (int(dim0_max_scaled), int(dim1_max_scaled)),
            }
            bboxes.append(bbox_info)
        return bboxes

    def __fit_ellipses(
        self, mask_2d, cluster_size_threshold, pixel_sizes, slice_dim, slice_idx
    ):
        # Find connected components and store them with sizes
        labeled_array, _ = label(mask_2d)
        sizes = np.bincount(labeled_array.ravel())[1:]
        # Store visualization info
        valid_ellipses = []
        valid_centers = []
        valid_axes = []
        valid_angles = []
        valid_landmarks_coords = []
        valid_ROIs = []
        # Sort clusters by size (largest to smallest)
        sorted_cluster_indices = np.argsort(-sizes)  # Negative for descending order
        # Loop through all clusters
        landmarks = []
        for cluster_idx in sorted_cluster_indices:
            cluster_label = cluster_idx + 1
            cluster_size = sizes[cluster_label - 1]
            if cluster_size < cluster_size_threshold:
                continue
            # Get mask for current cluster
            mask_1ROI = (labeled_array == cluster_label).astype(np.uint8)
            # Fit ellipse to current cluster
            contours, _ = cv2.findContours(
                mask_1ROI, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # Convert contour points to real-world coordinates
            contour_real = contours[0].squeeze() * pixel_sizes
            # Fit ellipse in real-world coordinates
            ellipse_real = cv2.fitEllipse(contour_real.astype(np.float32))
            center_real, axes_real, angle = ellipse_real
            # Convert center back to pixel coordinates
            center = (
                center_real[0] / pixel_sizes[0],
                center_real[1] / pixel_sizes[1],
            )
            # Convert axes back to pixel coordinates while preserving aspect ratio
            axes = (
                axes_real[0] / pixel_sizes[0],
                axes_real[1] / pixel_sizes[1],
            )
            # Calculate ellipse points in pixel coordinates
            angle_rad = np.deg2rad(angle)
            a, b = axes[0] / 2, axes[1] / 2
            major_x = a * np.cos(angle_rad)
            major_y = a * np.sin(angle_rad)
            minor_x = -b * np.sin(angle_rad)
            minor_y = b * np.cos(angle_rad)
            # Calculate landmark coordinates in pixel space
            idx1_dim1 = center[0] + major_x
            idx1_dim0 = center[1] + major_y
            idx2_dim1 = center[0] - major_x
            idx2_dim0 = center[1] - major_y
            idx3_dim1 = center[0] + minor_x
            idx3_dim0 = center[1] + minor_y
            idx4_dim1 = center[0] - minor_x
            idx4_dim0 = center[1] - minor_y
            # Calculate axis lengths
            p1p2_length = np.sqrt(
                (idx1_dim0 - idx2_dim0) ** 2 + (idx1_dim1 - idx2_dim1) ** 2
            )
            p3p4_length = np.sqrt(
                (idx3_dim0 - idx4_dim0) ** 2 + (idx3_dim1 - idx4_dim1) ** 2
            )
            # Skip if either axis has zero length
            if p1p2_length < 1e-6 or p3p4_length < 1e-6:
                continue
            # Swap points if needed for consistency
            if p1p2_length < p3p4_length:
                idx1_dim0, idx3_dim0 = idx3_dim0, idx1_dim0
                idx1_dim1, idx3_dim1 = idx3_dim1, idx1_dim1
                idx2_dim0, idx4_dim0 = idx4_dim0, idx2_dim0
                idx2_dim1, idx4_dim1 = idx4_dim1, idx2_dim1
            # Reorder points based on index values
            if (idx1_dim0 < idx2_dim0) or (
                idx1_dim0 == idx2_dim0 and idx1_dim1 < idx2_dim1
            ):
                idx1_dim0, idx2_dim0 = idx2_dim0, idx1_dim0
                idx1_dim1, idx2_dim1 = idx2_dim1, idx1_dim1

            if (idx3_dim0 < idx4_dim0) or (
                idx3_dim0 == idx4_dim0 and idx3_dim1 < idx4_dim1
            ):
                idx3_dim0, idx4_dim0 = idx4_dim0, idx3_dim0
                idx3_dim1, idx4_dim1 = idx4_dim1, idx3_dim1

            # Get bounding boxes and check if landmarks are within
            enlarged_bboxes = self._find_scaled_bounding_boxes_2D(
                mask_1ROI, self.enlarged_bbox_scale
            )
            shrunk_bboxes = self._find_scaled_bounding_boxes_2D(
                mask_1ROI, self.shrunk_bbox_scale
            )
            enlarged_min, enlarged_max = (
                enlarged_bboxes[0]["min_coords"],
                enlarged_bboxes[0]["max_coords"],
            )
            shrunk_min, shrunk_max = (
                shrunk_bboxes[0]["min_coords"],
                shrunk_bboxes[0]["max_coords"],
            )
            # Check if all landmarks are within the buffer zone between shrunk and enlarged boxes
            points = [
                (idx1_dim0, idx1_dim1),
                (idx2_dim0, idx2_dim1),
                (idx3_dim0, idx3_dim1),
                (idx4_dim0, idx4_dim1),
            ]
            all_within = all(
                enlarged_min[0] <= p[0] <= enlarged_max[0]
                and enlarged_min[1] <= p[1] <= enlarged_max[1]
                and (
                    p[0] <= shrunk_min[0]
                    or p[0] >= shrunk_max[0]
                    or p[1] <= shrunk_min[1]
                    or p[1] >= shrunk_max[1]
                )
                for p in points
            )
            if all_within:
                # Create landmark dictionary
                landmark_dict = {}
                if slice_dim == 0:
                    landmark_dict = {
                        "P1": [
                            int(slice_idx),
                            int(round(idx1_dim0)),
                            int(round(idx1_dim1)),
                        ],
                        "P2": [
                            int(slice_idx),
                            int(round(idx2_dim0)),
                            int(round(idx2_dim1)),
                        ],
                        "P3": [
                            int(slice_idx),
                            int(round(idx3_dim0)),
                            int(round(idx3_dim1)),
                        ],
                        "P4": [
                            int(slice_idx),
                            int(round(idx4_dim0)),
                            int(round(idx4_dim1)),
                        ],
                        "ROI_pixels_count": int(cluster_size),
                    }
                elif slice_dim == 1:
                    landmark_dict = {
                        "P1": [
                            int(round(idx1_dim0)),
                            int(slice_idx),
                            int(round(idx1_dim1)),
                        ],
                        "P2": [
                            int(round(idx2_dim0)),
                            int(slice_idx),
                            int(round(idx2_dim1)),
                        ],
                        "P3": [
                            int(round(idx3_dim0)),
                            int(slice_idx),
                            int(round(idx3_dim1)),
                        ],
                        "P4": [
                            int(round(idx4_dim0)),
                            int(slice_idx),
                            int(round(idx4_dim1)),
                        ],
                        "ROI_pixels_count": int(cluster_size),
                    }
                else:
                    landmark_dict = {
                        "P1": [
                            int(round(idx1_dim0)),
                            int(round(idx1_dim1)),
                            int(slice_idx),
                        ],
                        "P2": [
                            int(round(idx2_dim0)),
                            int(round(idx2_dim1)),
                            int(slice_idx),
                        ],
                        "P3": [
                            int(round(idx3_dim0)),
                            int(round(idx3_dim1)),
                            int(slice_idx),
                        ],
                        "P4": [
                            int(round(idx4_dim0)),
                            int(round(idx4_dim1)),
                            int(slice_idx),
                        ],
                        "ROI_pixels_count": int(cluster_size),
                    }
                landmarks.append(landmark_dict)
                # Store visualization info
                valid_ellipses.append(ellipse_real)
                valid_centers.append(center)
                valid_axes.append(axes)
                valid_angles.append(angle)
                valid_landmarks_coords.append(points)
                valid_ROIs.append(mask_1ROI)
        valid_ellipses_info = {
            "ellipses": valid_ellipses,
            "centers": valid_centers,
            "axes": valid_axes,
            "angles": valid_angles,
            "landmarks_coords": valid_landmarks_coords,
            "ROIs": valid_ROIs,
        }
        return landmarks, valid_ellipses_info

    def __plot_img_ellipse_landmarks(
        self,
        image_2d,
        pixel_sizes,
        valid_ellipses_info,
        slice_dim,
        slice_idx,
        case_id,
        landmarks_fig_dir,
    ):
        # Extract ellipse information
        valid_ellipses = valid_ellipses_info["ellipses"]
        valid_centers = valid_ellipses_info["centers"]
        valid_axes = valid_ellipses_info["axes"]
        valid_angles = valid_ellipses_info["angles"]
        valid_ROIs = valid_ellipses_info["ROIs"]
        valid_landmarks_coords = valid_ellipses_info["landmarks_coords"]
        colors = [
            "#4285F4",
            "#EA4335",
            "#FDB813",
            "#34A853",
        ]

        # Create visualization
        img_height, img_width = image_2d.shape
        aspect_ratio = img_width / img_height
        base_size = 10
        figsize = (
            (base_size * aspect_ratio, base_size)
            if aspect_ratio > 1
            else (base_size, base_size / aspect_ratio)
        )
        # Calculate aspect ratio based on pixel sizes
        aspect_ratio = pixel_sizes[1] / pixel_sizes[0]
        # Plot image and landmarks with correct aspect ratio
        plt.figure(figsize=figsize)
        plt.imshow(
            image_2d.T,
            cmap="gray",
            origin="lower",
            aspect=aspect_ratio,
            zorder=-1,
        )

        # Plot all valid ellipses and landmarks
        for i in range(len(valid_ellipses)):
            # Add ellipse
            ellipse_patch = plt.matplotlib.patches.Ellipse(
                xy=(valid_centers[i][1], valid_centers[i][0]),
                width=valid_axes[i][1],
                height=valid_axes[i][0],
                angle=-valid_angles[i],
                fill=False,
                color="red",
                linewidth=2,
                zorder=1,
            )
            plt.gca().add_patch(ellipse_patch)
            # Plot mask contour
            plt.contour(
                valid_ROIs[i].T,
                levels=[0.5],
                colors="#97D540",
                linewidths=2,
                origin="lower",
                zorder=0,
            )
            # Plot landmarks
            for j, (x, y) in enumerate(valid_landmarks_coords[i]):
                plt.scatter(
                    x,
                    y,
                    color=colors[j],
                    edgecolors="black",
                    marker="o",
                    s=60,
                    linewidth=1,
                    label=f"P{j+1}",
                    zorder=2,
                )
            # Plot axes
            plt.plot(
                [
                    valid_landmarks_coords[i][0][0],
                    valid_landmarks_coords[i][1][0],
                ],
                [
                    valid_landmarks_coords[i][0][1],
                    valid_landmarks_coords[i][1][1],
                ],
                color="#F37020",
                linestyle="-",
                linewidth=2,
                label="major axis",
                zorder=3,
            )
            plt.plot(
                [
                    valid_landmarks_coords[i][2][0],
                    valid_landmarks_coords[i][3][0],
                ],
                [
                    valid_landmarks_coords[i][2][1],
                    valid_landmarks_coords[i][3][1],
                ],
                color="#FBBC05",
                linestyle="-",
                linewidth=2,
                label="minor axis",
                zorder=3,
            )
        # Add scale bar
        min_idx = np.argmin(image_2d.shape[:2])
        scale_mm, num_pixels_dim_min = self._get_appropriate_scale(
            pixel_sizes[min_idx],
            image_2d.shape[min_idx],
            init_scale=10,
        )
        num_pixels_dim_max = int(scale_mm / pixel_sizes[1 - min_idx])
        if min_idx == 0:
            scale_pixels_dim0, scale_pixels_dim1 = (
                num_pixels_dim_min,
                num_pixels_dim_max,
            )
        else:
            scale_pixels_dim0, scale_pixels_dim1 = (
                num_pixels_dim_max,
                num_pixels_dim_min,
            )
        start_x, start_y = int(img_height * 0.05), int(img_width * 0.05)
        end_x, end_y = (
            start_x + scale_pixels_dim0,
            start_y + scale_pixels_dim1,
        )
        plt.plot([start_x, end_x], [start_y, start_y], "w-", linewidth=2)
        plt.plot([start_x, start_x], [start_y, end_y], "w-", linewidth=2)
        plt.text(
            end_x + img_height * 0.01,
            start_y,
            f"{scale_mm} mm",
            color="white",
            horizontalalignment="left",
        )
        # Set title and labels
        if slice_dim == 0:
            slice_filename = f"Sagittal_{slice_idx}.png"
            plt.xlabel("Anterior →", fontsize=14)
            plt.ylabel("Superior →", fontsize=14)
        elif slice_dim == 1:
            slice_filename = f"Coronal_{slice_idx}.png"
            plt.xlabel("Right →", fontsize=14)
            plt.ylabel("Superior →", fontsize=14)
        else:
            slice_filename = f"Axial_{slice_idx}.png"
            plt.xlabel("Right →", fontsize=14)
            plt.ylabel("Anterior →", fontsize=14)
        plt.tight_layout(pad=1.5, rect=[0.05, 0.05, 0.95, 0.95])
        # Save visualization
        case_fig_dir = os.path.join(landmarks_fig_dir, case_id)
        os.makedirs(case_fig_dir, exist_ok=True)
        plt.savefig(
            os.path.join(case_fig_dir, slice_filename),
            bbox_inches="tight",
        )
        plt.close()

    def _extract_ellipse_landmarks(self, task_info):
        """Extract ellipse landmarks from binary masks and save them as JSON files with visualizations.
        Logic:
        1. Load mask and image data, extract case ID
        2. Process each dimension (sagittal, coronal, axial)
        3. For each slice: find connected components, fit ellipse, calculate landmarks
        4. Generate visualization with landmarks and scale bars
        5. Save landmarks to JSON and visualizations to PNG files
        """
        mask_prefix = task_info["mask_prefix"]
        mask_suffix = task_info["mask_suffix"]
        image_prefix = task_info["image_prefix"]
        image_suffix = task_info["image_suffix"]
        landmark_prefix = task_info["landmark_prefix"]
        landmark_suffix = task_info["landmark_suffix"]
        img_dir = task_info["image_folder"]
        mask_dir = task_info["mask_folder"]
        target_label = task_info["target_label"]
        cluster_size_threshold = task_info["cluster_size_threshold"]
        landmarks_json_dir = task_info["landmark_folder"]
        landmarks_fig_dir = task_info["landmark_figure_folder"]

        mask_files = glob.glob(os.path.join(mask_dir, "*.nii.gz"))
        print(f"Found {len(mask_files)} mask files")
        # Process each mask file
        for i, mask_file in enumerate(mask_files, 1):
            case_id = (
                os.path.basename(mask_file)
                .replace(mask_prefix, "")
                .replace(mask_suffix, "")
            )
            print(
                f"\n[{i}/{len(mask_files)}] Processing: {case_id}...\nMask file: {mask_file}"
            )
            # Load mask and image data
            mask_nii = nib.load(mask_file)
            mask_data = mask_nii.get_fdata()
            mask_binary = (mask_data == target_label).astype(np.uint8)
            image_file = os.path.join(
                img_dir,
                f"{image_prefix}{case_id}{image_suffix}",
            )
            image_nii = nib.load(image_file)
            image_data = image_nii.get_fdata()
            voxel_sizes = image_nii.header.get_zooms()
            # Initialize landmark storage
            slice_landmarks_x, slice_landmarks_y, slice_landmarks_z = [], [], []
            # Process each dimension
            for slice_dim in range(3):
                print(f" - Processing dimension {slice_dim}...")
                n_slices = mask_binary.shape[slice_dim]
                voxel_array = np.array(voxel_sizes)
                if slice_dim == 0:
                    pixel_sizes = voxel_array[[1, 2]]
                elif slice_dim == 1:
                    pixel_sizes = voxel_array[[0, 2]]
                else:
                    pixel_sizes = voxel_array[[0, 1]]
                # Process each slice
                for slice_idx in tqdm(
                    range(n_slices), desc=f" -- Processing slices (dim{slice_dim})"
                ):
                    # Extract 2D slice based on dimension
                    if slice_dim == 0:
                        mask_2d, image_2d = (
                            mask_binary[slice_idx, :, :],
                            image_data[slice_idx, :, :],
                        )
                    elif slice_dim == 1:
                        mask_2d, image_2d = (
                            mask_binary[:, slice_idx, :],
                            image_data[:, slice_idx, :],
                        )
                    else:
                        mask_2d, image_2d = (
                            mask_binary[:, :, slice_idx],
                            image_data[:, :, slice_idx],
                        )
                    if not np.any(mask_2d):
                        continue
                    # Fit ellipses
                    landmarks, valid_ellipses_info = self.__fit_ellipses(
                        mask_2d,
                        cluster_size_threshold,
                        pixel_sizes,
                        slice_dim,
                        slice_idx,
                    )
                    # Visualize landmarks and save to file
                    if self.visualization and len(valid_ellipses_info["ellipses"]) > 0:
                        self.__plot_img_ellipse_landmarks(
                            image_2d,
                            pixel_sizes,
                            valid_ellipses_info,
                            slice_dim,
                            slice_idx,
                            case_id,
                            landmarks_fig_dir,
                        )
                    # Store landmarks for current slice if any valid ellipses were found
                    if len(landmarks) > 0:
                        slice_dict = {
                            "slice_idx": int(slice_idx),
                            "landmarks": landmarks,
                        }
                        if slice_dim == 0:
                            slice_landmarks_x.append(slice_dict)
                        elif slice_dim == 1:
                            slice_landmarks_y.append(slice_dict)
                        else:
                            slice_landmarks_z.append(slice_dict)
            # Save all landmarks to JSON
            final_dict = {
                "slice_landmarks_x": slice_landmarks_x,
                "slice_landmarks_y": slice_landmarks_y,
                "slice_landmarks_z": slice_landmarks_z,
            }
            (
                os.makedirs(landmarks_json_dir)
                if not os.path.exists(landmarks_json_dir)
                else None
            )
            output_file = os.path.join(
                landmarks_json_dir, f"{landmark_prefix}{case_id}{landmark_suffix}"
            )
            # Check if output file ends with .json.gz or .json
            if output_file.endswith(".json.gz"):
                with gzip.open(output_file, "wt") as f:
                    json.dump(final_dict, f, indent=4)
            else:
                with open(output_file, "w") as f:
                    json.dump(final_dict, f, indent=4)
            print(f"Saved landmarks to {output_file}")

    def _get_biometrics(self, task_info, landmarks, voxel_sizes, slice_dim):
        biometrics = []
        for _, metric in enumerate(task_info["biometrics_map"], 1):
            metric_type = metric["metric_type"]
            metric_map_name = metric["metric_map_name"]
            metric_key = metric["metric_key"]
            if metric_type == "angle":
                metric_value, _ = self._cal_angle(
                    metric_key,
                    metric_map_name,
                    task_info,
                    landmarks,
                    voxel_sizes,
                    slice_dim,
                )
                metric_unit = "degree"
            elif metric_type == "distance":
                metric_value, _ = self._cal_distance(
                    metric_key,
                    metric_map_name,
                    task_info,
                    landmarks,
                    voxel_sizes,
                    slice_dim,
                )
                metric_unit = "mm"
            else:
                raise ValueError(f"Invalid metric_type: {metric_type}")
            biometrics.append(
                {
                    "metric_type": metric_type,
                    "metric_map_name": metric_map_name,
                    "metric_key": metric_key,
                    "metric_value": metric_value,
                    "metric_unit": metric_unit,
                    "slice_dim": slice_dim,
                }
            )
        return biometrics

    def _get_biometrics_batch(self, task_info, landmarks_json, voxel_sizes, slice_dim):
        slice_profiles = []
        if slice_dim == 0:
            slice_landmarks = landmarks_json["slice_landmarks_x"]
        elif slice_dim == 1:
            slice_landmarks = landmarks_json["slice_landmarks_y"]
        elif slice_dim == 2:
            slice_landmarks = landmarks_json["slice_landmarks_z"]
        else:
            raise ValueError("Invalid slice dimension")
        for _, slice_landmark in enumerate(slice_landmarks, 1):
            landmarks_list = slice_landmark["landmarks"]
            slice_profile = []
            for landmarks in landmarks_list:
                slice_idx = slice_landmark["slice_idx"]
                slice_profile.append(
                    self._get_biometrics(task_info, landmarks, voxel_sizes, slice_dim)
                )
            slice_profiles.append(
                {
                    "slice_idx": slice_idx,
                    "slice_profile": slice_profile,
                }
            )
        return slice_profiles

    def _update_cases_profile(self, images_list, task_info, split):
        if split not in ["train", "test"]:
            raise ValueError('\n\nError: split should be one of "train" or "test"\n\n')
        task_info["task_type"] = self.task_type
        task_info[f"{split}_cases_number"] = len(images_list)
        # Fit an ellipse to the chosen ROI and get 4 landmarks on the ellipse
        self._extract_ellipse_landmarks(task_info)
        for i, img_file in enumerate(images_list, 1):
            print(
                f"{'-'*50}\n[{i}/{len(images_list)}] Processing: {os.path.basename(img_file)}\n{'-'*50}"
            )
            # Check if mask and image properties match
            (
                caseID,
                mask_nii,
                _,
                image_path,
                mask_path,
                image_file_info,
                mask_file_info,
            ) = self._check_nii_header_for_img_mask(img_file, task_info)
            # Find and check the landmarks
            _, _, landmark_path = self._match_landmark_to_image_fromSeg(
                img_file, task_info
            )
            # Load landmarks from either .json.gz or .json file
            if landmark_path.endswith(".json.gz"):
                with gzip.open(landmark_path, "rt") as f:
                    landmarks_json = json.load(f)
            else:
                with open(landmark_path, "r") as f:
                    landmarks_json = json.load(f)
            # Get voxel size
            voxel_sizes = mask_nii.header.get_zooms()
            # Update biometrics for this case
            print(f"Updating profile for case: {caseID} ...")
            # Generate profile for sagittal, coronal and axial slices
            if len(landmarks_json["slice_landmarks_x"]) > 0:
                print(" - Generating profile for sagittal slices...")
                slice_profiles_x = self._get_biometrics_batch(
                    task_info, landmarks_json, voxel_sizes, slice_dim=0
                )
            else:
                slice_profiles_x = []
            if len(landmarks_json["slice_landmarks_y"]) > 0:
                print(" - Generating profile for coronal slices...")
                slice_profiles_y = self._get_biometrics_batch(
                    task_info, landmarks_json, voxel_sizes, slice_dim=1
                )
            else:
                slice_profiles_y = []
            if len(landmarks_json["slice_landmarks_z"]) > 0:
                print(" - Generating profile for axial slices...")
                slice_profiles_z = self._get_biometrics_batch(
                    task_info, landmarks_json, voxel_sizes, slice_dim=2
                )
            else:
                slice_profiles_z = []
            # Update the cases profile
            if f"{split}_cases" not in task_info:
                task_info[f"{split}_cases"] = []
            task_info[f"{split}_cases"].append(
                {
                    "case_ID": caseID,
                    "image_file": image_path,
                    "landmark_file": landmark_path,
                    "mask_file": mask_path,
                    "image_file_info": image_file_info,
                    "mask_file_info": mask_file_info,
                    "slice_profiles_x": slice_profiles_x,
                    "slice_profiles_y": slice_profiles_y,
                    "slice_profiles_z": slice_profiles_z,
                }
            )
            print(f"\nProfile updated for case {caseID}!\n{'-'*50}\n")

    @staticmethod
    def flatten_slice_profiles_2d(cases, slice_dim):
        if slice_dim == 0:
            slice_profiles_key = "slice_profiles_x"
        elif slice_dim == 1:
            slice_profiles_key = "slice_profiles_y"
        elif slice_dim == 2:
            slice_profiles_key = "slice_profiles_z"
        else:
            raise ValueError(f"\nError: slice_dim should be one of 0, 1, or 2\n")
        flatten_slice_profile = []
        for case in cases:
            landmark_file = case.get("landmark_file")
            image_file = case.get("image_file")
            mask_file = case.get("mask_file")
            # For the "Tumor-Lesion-Size" task, slice_profiles_dirX could be empty
            slice_profiles_dirX_ls = case.get(slice_profiles_key, [])
            if len(slice_profiles_dirX_ls) == 0:
                continue
            else:
                for slice_profiles_dict in slice_profiles_dirX_ls:
                    slice_idx = slice_profiles_dict.get("slice_idx")
                    slice_profiles_ls = slice_profiles_dict.get("slice_profile")

                    # For the "Tumor-Lesion-Size" task, slice_profiles is a list of profiles for each ellipse
                    # - Each ellipse profile is a list of dictionaries
                    # - Each dictionary contains the slice_idx and the biometric profile for the major and minor axes
                    biometric_ellipses_ls = []
                    metric_value_major_axis = None
                    metric_value_minor_axis = None
                    for ellipse_profile_ls in slice_profiles_ls:
                        for axis_profile_dict in ellipse_profile_ls:
                            if axis_profile_dict.get("metric_key") == "L-1-2":
                                metric_value_major_axis = axis_profile_dict.get(
                                    "metric_value"
                                )
                            elif axis_profile_dict.get("metric_key") == "L-3-4":
                                metric_value_minor_axis = axis_profile_dict.get(
                                    "metric_value"
                                )
                            else:
                                raise ValueError(
                                    f"\nError: metric_key should be one of L-1-2 or L-3-4\n"
                                )
                            metric_type = axis_profile_dict.get("metric_type")
                            metric_map_name = axis_profile_dict.get("metric_map_name")
                            metric_unit = axis_profile_dict.get("metric_unit")

                        # Check if both major and minor axes are present. If not, skip this profile
                        if (
                            metric_value_major_axis is None
                            or metric_value_minor_axis is None
                        ):
                            continue
                        else:
                            biometric_ellipses_ls.append(
                                {
                                    "metric_type": metric_type,
                                    "metric_map_name": metric_map_name,
                                    "metric_key_major_axis": "L-1-2",
                                    "metric_value_major_axis": metric_value_major_axis,
                                    "metric_key_minor_axis": "L-3-4",
                                    "metric_value_minor_axis": metric_value_minor_axis,
                                    "metric_unit": metric_unit,
                                }
                            )
                    flatten_slice_profile.append(
                        {
                            "image_file": image_file,
                            "landmark_file": landmark_file,
                            "mask_file": mask_file,
                            "slice_dim": slice_dim,
                            "slice_idx": slice_idx,
                            "biometric_profile": biometric_ellipses_ls,
                        }
                    )
        return flatten_slice_profile

    def process_each_task(self):
        # Process each task in the benchmark plan
        for task_idx, task in enumerate(self.bm_plan["tasks"], 1):
            print(
                f"{'='*50}\nProcessing {self.task_type} task {task_idx}/{len(self.bm_plan['tasks'])}\n{'='*50}"
            )
            # Update task ID
            task["task_ID"] = f"{task_idx:02d}"
            # Split the dataset into training and testing sets
            print("Splitting dataset into training and testing sets...")
            imgs_tr, imgs_ts = self._split_niigz_dataset(task["image_folder"])
            print(
                f"Split complete: {len(imgs_tr)} training, {len(imgs_ts)} testing cases\n"
            )
            # Update the profile of the training and testing sets
            print("Updating profiles for training set...\n")
            self._update_cases_profile(imgs_tr, task, "train")
            print("Updating profiles for testing set...\n")
            self._update_cases_profile(imgs_ts, task, "test")
            print(f"Finished processing task {task_idx}\n{'='*50}\n\n")

    def process(self):
        print(f"Preprocessing {self.dataset_name} dataset in {self.dataset_dir}...\n")
        self.update_tasks_number()
        if self.force_uint16_mask:
            self.convert_masks_to_uint16()
        if self.reorient2RAS:
            self.reorient_niigz_RASPlus()
        self.process_each_task()
        self.save_benchmark_plan()
