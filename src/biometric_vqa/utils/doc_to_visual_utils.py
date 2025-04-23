from PIL import ImageDraw, ImageFont


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
    # In the 2D array: height = first dimension, width = second dimension
    # In pixel_sizes: [height_scale, width_scale]
    # In PIL image: img_width = second dimension, img_height = first dimension
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
    # In the 2D array: height = first dimension, width = second dimension
    # In pixel_sizes: [height_scale, width_scale]
    # In PIL image: img_width = second dimension, img_height = first dimension
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
