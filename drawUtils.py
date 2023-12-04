import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import colorsys


def draw_gradient_contours(img, contour, name="Colored Contours"):
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

    # Resize for display, preserving aspect ratio
    scaling_factor = 2
    new_width = img_color.shape[1] * scaling_factor
    new_height = img_color.shape[0] * scaling_factor
    resized_img = cv2.resize(img_color, (new_width, new_height))

    cv2.imshow(name, resized_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


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


def draw_segmented_contours(img, contours, name="Segmented Contours"):
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

    # Resize for display, preserving aspect ratio
    scaling_factor = 2
    new_width = img_color.shape[1] * scaling_factor
    new_height = img_color.shape[0] * scaling_factor
    resized_img = cv2.resize(img_color, (new_width, new_height))

    cv2.imshow(name, resized_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
