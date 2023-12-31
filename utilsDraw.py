import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import colorsys


def draw_gradient_contours(
    img, contour, name="Colored Contours", wait=False, draw=True
):
    length = len(contour)

    # Convert grayscale to BGR if necessary
    if len(img.shape) == 2:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img.copy()

    for i, point in enumerate(contour):
        ratio = i / (length + 1)  # adjusted to account for the endpoint
        # Interpolate between blue and red with green in the middle
        if ratio < 0.5:
            # Transition from blue to green in the first half
            blue = 255 - int(510 * ratio)  # Decrease blue
            green = int(510 * ratio)  # Increase green
            red = 0
        else:
            # Transition from green to red in the second half
            blue = 0
            green = 255 - int(510 * (ratio - 0.5))  # Decrease green
            red = int(510 * (ratio - 0.5))  # Increase red

        color = (blue, green, red)

        cv2.circle(img_color, tuple(point[0]), 2, color, 2)

    return scale_piece(img_color, name, 2, wait, draw)


def plot_histogram(angle_differences):
    num_pts = len(angle_differences)

    max_angle = max(angle_differences)
    min_angle = min(angle_differences)

    # Create the plot
    plt.figure(figsize=(18, 6))
    # Generate a color based on the angle difference magnitude
    colors = plt.cm.jet(np.abs(angle_differences) / max(angle_differences))

    # Create a scatter plot with a color gradient
    sc = plt.scatter(range(num_pts), angle_differences, c=colors, cmap="jet")
    # Plotting the line without markers
    plt.plot(angle_differences, color="grey", alpha=0.5)

    # Add color bar based on the color map
    plt.colorbar(sc, label="Angle Magnitude")

    plt.title("Angle Differences Histogram")
    plt.xlabel("Index of Angle Difference")
    plt.ylabel("Angle Difference (degrees)")
    plt.grid(True)

    # Reduce the number of x-axis labels
    plt.xticks(range(0, num_pts, 10))

    # Set the y-axis limit to 180 degrees
    plt.ylim(min_angle, max_angle)
    plt.yticks(range(min_angle, max_angle, 10))

    # plt.show()


def generate_spaced_colors(n):
    """
    Generate 'n' RGB colors with maximum spacing.

    Parameters:
    n (int): The number of colors to generate.

    Returns:
    list of tuples: A list of RGB colors.
    """
    colors = []
    for i in range(n):
        # Evenly distribute the hue across 360 degrees
        hue = i / n
        # Convert HSL to RGB (using fixed saturation and lightness)
        rgb = colorsys.hls_to_rgb(hue, 0.5, 0.5)
        # Convert to 0-255 scale for RGB
        rgb_scaled = tuple(int(val * 255) for val in rgb)
        colors.append(rgb_scaled)

    return colors


def draw_segmented_contours(
    img, contours, name="Segmented Contours", wait=False, draw=True
):
    # Check if contours is a single contour or a list of contours
    if isinstance(contours[0], np.ndarray) and len(contours[0].shape) == 2:
        # It's a single contour
        contours = [contours]  # Make it a list of one contour
        colors = [(0, 255, 0)]  # Single color for a single contour
    else:
        # It's a list of contours
        colors = generate_spaced_colors(len(contours))

    # Convert grayscale to BGR if necessary
    if len(img.shape) == 2:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img.copy()

    for i, contour in enumerate(contours):
        for point in contour:
            cv2.circle(img_color, tuple(point[0]), 2, colors[i], 2)

    return scale_piece(img_color, name, 2, wait, draw)


def scale_piece(img, name="", scale_factor=2.0, wait=False, draw=True):
    # Resize for display, preserving aspect ratio
    new_width = img.shape[1] * scale_factor
    new_height = img.shape[0] * scale_factor
    resized_img = cv2.resize(img, (int(new_width), int(new_height)))

    if draw:
        cv2.imshow(name, resized_img)
        if wait:
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return resized_img


def show_all(image_list, name="", row_count=5, scale_factor=2, wait=True, draw=True):
    rows = []

    # Find the maximum width and height in the entire list
    max_width = max(img.shape[1] for img in image_list)
    max_height = max(img.shape[0] for img in image_list)

    # Resize or pad images to the max height and width
    resized_imgs = []
    for img in image_list:
        # Convert to RGB if necessary
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # Calculate padding dimensions
        pad_top = (max_height - img.shape[0]) // 2
        pad_bottom = max_height - img.shape[0] - pad_top
        pad_left = (max_width - img.shape[1]) // 2
        pad_right = max_width - img.shape[1] - pad_left

        # Pad the image
        padded_img = cv2.copyMakeBorder(
            img,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )
        resized_imgs.append(padded_img)

    for i in range(0, len(resized_imgs), row_count):
        batch = resized_imgs[i : i + row_count]

        # Fill in the last row with black images if necessary
        while len(batch) < row_count:
            black_img = np.zeros((max_height, max_width, 3), dtype=np.uint8)
            batch.append(black_img)

        # Concatenate horizontally
        row = np.hstack(batch)
        rows.append(row)

    # Concatenate vertically
    concat_img = np.vstack(rows)

    return scale_piece(concat_img, name, scale_factor, wait, draw)
