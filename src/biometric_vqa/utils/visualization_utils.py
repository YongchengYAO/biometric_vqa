import json
import os
import glob
import matplotlib.pyplot as plt
import nibabel as nib


def plot_slice_with_landmarks(nii_path: str, json_path: str, fig_path: str = None):
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
    with open(json_path, "r") as f:
        landmarks = json.load(f)

    # Setup visualization
    plt.figure(figsize=(12, 12))
    plt.imshow(
        slice_data.T, cmap="gray", origin="lower"
    )  # the transpose is necessary only for visualization

    # Extract and plot landmark coordinates
    x_coords = []
    y_coords = []
    for point_id, coords in landmarks.items():
        if len(coords) == 3:  # Check for valid [1, x, y] format
            # Note: this is definitely correct, DO NOT SWAP coords[1] and coords[2]
            x_coords.append(coords[1])
            y_coords.append(coords[2])

    # Add landmarks and labels
    plt.scatter(
        x_coords,
        y_coords,
        facecolors="#18A727",
        edgecolors="black",
        marker="o",
        s=80,
        linewidth=1.5,
    )
    for i, (x, y) in enumerate(zip(x_coords, y_coords), 1):
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
    if fig_path:
        plt.savefig(fig_path, bbox_inches="tight", dpi=300)
        print(f"Plot saved to: {fig_path}")
    else:
        plt.show()

    plt.close()


def plot_slice_with_landmarks_batch(image_dir: str, landmark_dir: str, fig_dir: str):
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
        json_path = os.path.join(landmark_dir, f"{base_name}.json")
        fig_path = os.path.join(fig_dir, f"{base_name}.png")

        # Plot and save
        if os.path.exists(json_path):
            plot_slice_with_landmarks(nii_path, json_path, fig_path)
        else:
            print(f"Warning: No landmark file found for {base_name}")


def plot_2Darray_wRASinfo(img_data, slice_dim, pixel_sizes, save_path):
    """Helper function to plot 2D image slices with RAS orientation info."""
    # Create visualization
    img_height, img_width = img_data.shape
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

    # Handle different slice orientations
    if slice_dim == 0:  # Sagittal
        plt.imshow(
            img_data.T,
            cmap="gray",
            origin="lower",
            aspect=aspect_ratio,
        )
        plt.xlabel("Anterior →", fontsize=14)
        plt.ylabel("Superior →", fontsize=14)
    elif slice_dim == 1:  # Coronal
        plt.imshow(
            img_data.T,
            cmap="gray",
            origin="lower",
            aspect=aspect_ratio,
        )
        plt.xlabel("Right →", fontsize=14)
        plt.ylabel("Superior →", fontsize=14)
    else:  # Axial
        plt.imshow(img_data.T, cmap="gray", origin="lower", aspect=aspect_ratio)
        plt.xlabel("Right →", fontsize=14)
        plt.ylabel("Anterior →", fontsize=14)

    plt.margins(0)
    plt.savefig(save_path)
    plt.close()
