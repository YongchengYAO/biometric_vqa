from PIL import ImageDraw, ImageFont, Image
import numpy as np
import cv2


def add_landmarks_and_line_overlay(pil_img, p1_coords, p2_coords):
    """
    Add landmarks (points) and a line connecting them to an image.

    Args:
        pil_img: PIL Image in RGB or RGBA format
        p1_coords: List of [dim0, dim1] coordinates for the first point
        p2_coords: List of [dim0, dim1] coordinates for the second point

    Returns:
        PIL Image with the landmarks and connecting line overlay
    """
    # Create a drawing object
    draw = ImageDraw.Draw(pil_img)

    # Convert coordinates format (similar to bbox function)
    # First coordinate is height (y) and second is width (x)
    x1, y1 = p1_coords[1], p1_coords[0]
    x2, y2 = p2_coords[1], p2_coords[0]

    # Draw the line connecting the points (green)
    draw.line([(x1, y1), (x2, y2)], fill="#00FF00", width=2)

    # Draw the points (red)
    point_radius = 3
    draw.ellipse(
        [
            (x1 - point_radius, y1 - point_radius),
            (x1 + point_radius, y1 + point_radius),
        ],
        fill="#FF0000",
    )
    draw.ellipse(
        [
            (x2 - point_radius, y2 - point_radius),
            (x2 + point_radius, y2 + point_radius),
        ],
        fill="#FF0000",
    )

    return pil_img


def add_bbox_overlay(pil_img, bbox_min_coords, bbox_max_coords):
    """
    Add a bounding box overlay to an image.

    Args:
        pil_img: PIL Image in RGB or RGBA format
        bbox_min_coords: List of [dim0_min, dim1_min] coordinates for the top-left corner of the bounding box
        bbox_max_coords: List of [dim0_max, dim1_max] coordinates for the bottom-right corner of the bounding box

    NOTE: For the coordinate definition in the BiometricVQA dataset, please refer to the
          `biometric_vqa.utils.benchmark_planner.BiometricVQA_BenchmarkPlannerDetection._find_bounding_boxes_2D`

    Returns:
        PIL Image with the bounding box overlay
    """
    # NOTE: For bbox_min_coords and bbox_max_coords:
    #           the first coordinate is the height (y-axis) direction and the second is the width (x-axis) direction;
    #           the origin is at the upper-left corner of the image.
    #       For PIL Image, the origin is at the upper-left corner of the image.
    #           So, x-coordinate = dim1_coordinate, y-coordinate = dim0_coordinate

    # Convert input bounding box corrdinates to xy coordinates for PIL Image
    x_min = bbox_min_coords[1]
    y_min = bbox_min_coords[0]
    x_max = bbox_max_coords[1]
    y_max = bbox_max_coords[0]

    # ref: https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html
    draw = ImageDraw.Draw(pil_img)
    draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="#00FF00", width=2)
    return pil_img


def add_mask_overlay_contour(pil_img, mask_2d_binary):
    """
    Add a green contour outline to an image based on a binary mask.

    Args:
        pil_img: PIL Image in RGB or RGBA format
        mask_2d_binary: Binary numpy array representing the mask

    Returns:
        PIL Image with the mask contour overlaid in green
    """

    # Convert PIL image to numpy array for OpenCV
    img_np = np.array(pil_img)

    # Make sure mask is the right type and size
    mask = mask_2d_binary.astype(np.uint8)
    if mask.shape[:2] != img_np.shape[:2]:
        mask = cv2.resize(
            mask, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a copy of the image to draw on
    img_with_contour = img_np.copy()

    # Draw contours on the image
    cv2.drawContours(img_with_contour, contours, -1, (0, 255, 0), 2)

    # Convert back to PIL
    return Image.fromarray(img_with_contour)


def add_mask_overlay_solid(pil_img, mask_2d_binary):
    """
    Add a semi-transparent green overlay to an image based on a binary mask.

    Args:
        pil_img: PIL Image in RGB or RGBA format
        mask_2d_binary: Binary numpy array representing the mask

    Returns:
        PIL Image with the mask overlaid in green
    """
    # Create a green overlay image
    overlay = Image.new("RGBA", pil_img.size, (0, 255, 0, 0))

    # Convert mask to PIL image format and resize if needed
    mask_pil = Image.fromarray((mask_2d_binary * 64).astype(np.uint8), mode="L")
    if mask_pil.size != pil_img.size:
        mask_pil = mask_pil.resize(pil_img.size)

    # Set the mask as the alpha channel for the overlay
    overlay.putalpha(mask_pil)

    # Convert original image to RGBA
    pil_img = pil_img.convert("RGBA")

    # Composite the images
    pil_img = Image.alpha_composite(pil_img, overlay)

    # Convert back to RGB for display
    pil_img = pil_img.convert("RGB")

    return pil_img


def add_scale_label(pil_img, pixel_sizes, slice_dim):
    """Add scale label to image."""
    draw = ImageDraw.Draw(pil_img)

    # Get image dimensions - in PIL, size returns (width, height)
    img_width, img_height = pil_img.size

    # Define a class with the _get_appropriate_scale method
    class ScaleCalculator:
        def _get_appropriate_scale(self, pixel_size, img_size, init_scale=10):
            scales = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
            scale_pixels_num = int(init_scale / pixel_size)
            min_pixels = img_size * 0.05
            max_pixels = img_size * 0.25

            if scale_pixels_num < min_pixels:
                for scale in scales:
                    if scale > init_scale:
                        return self._get_appropriate_scale(pixel_size, img_size, scale)
            elif scale_pixels_num > max_pixels:
                for scale in reversed(scales):
                    if scale < init_scale:
                        return self._get_appropriate_scale(pixel_size, img_size, scale)

            return init_scale, scale_pixels_num

    scale_calculator = ScaleCalculator()

    # Find which dimension is smaller
    #   In the 2D array: height = first dimension, width = second dimension
    #   In pixel_sizes: [height_scale, width_scale]
    #   In PIL image: img_width = second dimension, img_height = first dimension
    if img_height < img_width:  # Height is the smaller dimension
        pixel_size_min = pixel_sizes[0]  # Height pixel size
        image_dim_min = img_height
    else:  # Width is the smaller dimension
        pixel_size_min = pixel_sizes[1]  # Width pixel size
        image_dim_min = img_width

    # Calculate appropriate scale
    scale_mm, scale_pixels_min = scale_calculator._get_appropriate_scale(
        pixel_size_min, image_dim_min, init_scale=10
    )

    # Calculate scale for the other dimension
    if img_height < img_width:
        scale_pixels_height = scale_pixels_min
        scale_pixels_width = int(scale_mm / pixel_sizes[1])
    else:
        scale_pixels_width = scale_pixels_min
        scale_pixels_height = int(scale_mm / pixel_sizes[0])

    # Position for scale bar (5% from the edge)
    start_x, start_y = int(img_width * 0.05), int(img_height * 0.05)
    end_x, end_y = start_x + scale_pixels_width, start_y + scale_pixels_height

    # Draw scale bar (white lines)
    line_width = 2
    # Draw horizontal line
    draw.line(
        [(start_x, start_y), (end_x, start_y)],
        fill=(255, 255, 255),
        width=line_width,
    )
    # Draw vertical line
    draw.line(
        [(start_x, start_y), (start_x, end_y)],
        fill=(255, 255, 255),
        width=line_width,
    )

    # Set text font
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()

    # Add scale text
    draw.text(
        (start_x + 5, start_y + 5), f"{scale_mm} mm", fill=(255, 255, 255), font=font
    )

    return pil_img


def add_scale_and_orientation_label(pil_img, pixel_sizes, slice_dim):
    """Add scale bar and orientation labels to image."""
    draw = ImageDraw.Draw(pil_img)

    # Get image dimensions - in PIL, size returns (width, height)
    img_width, img_height = pil_img.size

    # Define a class with the _get_appropriate_scale method
    class ScaleCalculator:
        def _get_appropriate_scale(self, pixel_size, img_size, init_scale=10):
            scales = [1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
            scale_pixels_num = int(init_scale / pixel_size)
            min_pixels = img_size * 0.05
            max_pixels = img_size * 0.25

            if scale_pixels_num < min_pixels:
                for scale in scales:
                    if scale > init_scale:
                        return self._get_appropriate_scale(pixel_size, img_size, scale)
            elif scale_pixels_num > max_pixels:
                for scale in reversed(scales):
                    if scale < init_scale:
                        return self._get_appropriate_scale(pixel_size, img_size, scale)

            return init_scale, scale_pixels_num

    scale_calculator = ScaleCalculator()

    # Find which dimension is smaller
    #   In the 2D array: height = first dimension, width = second dimension
    #   In pixel_sizes: [height_scale, width_scale]
    #   In PIL image: img_width = second dimension, img_height = first dimension
    if img_height < img_width:  # Height is the smaller dimension
        pixel_size_min = pixel_sizes[0]  # Height pixel size
        image_dim_min = img_height
    else:  # Width is the smaller dimension
        pixel_size_min = pixel_sizes[1]  # Width pixel size
        image_dim_min = img_width

    # Calculate appropriate scale
    scale_mm, scale_pixels_min = scale_calculator._get_appropriate_scale(
        pixel_size_min, image_dim_min, init_scale=10
    )

    # Calculate scale for the other dimension
    if img_height < img_width:
        scale_pixels_height = scale_pixels_min
        scale_pixels_width = int(scale_mm / pixel_sizes[1])
    else:
        scale_pixels_width = scale_pixels_min
        scale_pixels_height = int(scale_mm / pixel_sizes[0])

    # Position for scale bar (5% from the edge)
    start_x, start_y = int(img_width * 0.05), int(img_height * 0.05)
    end_x, end_y = start_x + scale_pixels_width, start_y + scale_pixels_height

    # Draw scale bar (white lines)
    line_width = 2
    # Draw horizontal line
    draw.line(
        [(start_x, start_y), (end_x, start_y)],
        fill=(255, 255, 255),
        width=line_width,
    )
    # Draw vertical line
    draw.line(
        [(start_x, start_y), (start_x, end_y)],
        fill=(255, 255, 255),
        width=line_width,
    )

    # Set text font
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()

    # Add scale text
    draw.text(
        (start_x + 5, start_y + 5), f"{scale_mm} mm", fill=(255, 255, 255), font=font
    )

    # Add orientation labels based on slice_dim
    label_padding = 10
    if slice_dim == 0:
        draw.text((start_x, end_y + 5), "Anterior", fill=(255, 255, 255), font=font)
        draw.text((end_x + 5, start_y), "Superior", fill=(255, 255, 255), font=font)
    elif slice_dim == 1:
        draw.text((start_x, end_y + 5), "Right", fill=(255, 255, 255), font=font)
        draw.text((end_x + 5, start_y), "Superior", fill=(255, 255, 255), font=font)
    else:
        draw.text((start_x, end_y + 5), "Right", fill=(255, 255, 255), font=font)
        draw.text((end_x + 5, start_y), "Anterior", fill=(255, 255, 255), font=font)

    return pil_img
