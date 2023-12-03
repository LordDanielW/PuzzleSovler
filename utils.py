import cv2
import csv
import numpy as np
import os
import glob
import math
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from itertools import combinations
import colorsys


def find_bounding_box(image):
    # For grayscale image, we don't compare against a 3-channel color.
    # Instead, we simply look for pixels that are not black (i.e., not 0).
    pts = np.argwhere(image > 0)  # Assuming black pixels have value 0.

    # Find the minimum and maximum x and y coordinates.
    min_y, min_x = pts.min(axis=0)
    max_y, max_x = pts.max(axis=0)

    # Create a box around those points.
    # Adding a border of 5 pixels around the non-black area.
    min_y = max(min_y - 5, 0)
    min_x = max(min_x - 5, 0)
    max_y = min(max_y + 5, image.shape[0])
    max_x = min(max_x + 5, image.shape[1])

    # Return the coordinates of the box.
    return [(min_x, min_y), (max_x, max_y)]


def rotate_image_easy(image, angle):
    # Define the color of the border and the size of the border.
    background_color = (0, 0, 0)  # black
    border_size = max(image.shape)  # You may adjust this size as needed.

    # Create an expanded image by adding a border around the original image.
    expanded_image = cv2.copyMakeBorder(
        image,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=background_color,
    )

    # Using OpenCV's built-in rotation function to rotate the image.
    # 'borderValue=background_color' fills the border with the specified color.
    # 'interpolation=cv2.INTER_LINEAR' is used here, but you can adjust as needed.
    rotated_image = cv2.rotate(expanded_image, angle)

    # Find the bounding box of the non-black areas.
    box = find_bounding_box(rotated_image)

    # Crop the image using the bounding box.
    cropped_image = rotated_image[box[0][1] : box[1][1], box[0][0] : box[1][0]]

    return cropped_image


def rotate_image(image, angle):
    # Define the color of the border and the size of the border.
    background_color = (0, 0, 0)  # black
    border_size = max(image.shape)  # You may adjust this size as needed.

    # Create an expanded image by adding a border around the original image.
    expanded_image = cv2.copyMakeBorder(
        image,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=background_color,
    )

    # Using scipy's ndimage to rotate the image within the expanded border.
    # 'reshape=False' ensures the output shape is the same as the input image.
    # 'cval=0' sets the background to black where there are no image data.
    # 'order=1' (bilinear interpolation) is used here, but you can adjust as needed.
    rotated_image = ndimage.rotate(
        expanded_image, angle, reshape=False, cval=0, order=1
    )

    # Find the bounding box of the non-black areas.
    box = find_bounding_box(rotated_image)

    # Crop the image using the bounding box.
    cropped_image = rotated_image[box[0][1] : box[1][1], box[0][0] : box[1][0]]

    return cropped_image


def read_puzzle_pieces_info(csv_file):
    puzzlePiecesInfo = []
    with open(csv_file, "r") as file:
        csvreader = csv.DictReader(file)
        for row in csvreader:
            # Convert all values to integers if needed
            info = {
                key: int(value) if value.isdigit() else value
                for key, value in row.items()
            }
            puzzlePiecesInfo.append(info)
    return puzzlePiecesInfo


def load_puzzle_pieces(puzzle_folder):
    puzzlePieces = []
    pieceInfo = []
    i = 0
    while True:
        filepath = os.path.join(puzzle_folder, f"piece_{i}.png")
        if os.path.exists(filepath):
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                puzzlePieces.append(img)
                pieceInfo.append({"piece_name": f"piece_{i}"})
            i += 1
        else:
            break
    return puzzlePieces, pieceInfo


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
    colors = generate_spaced_colors(len(contours))

    # Convert grayscale to BGR if necessary
    if len(img.shape) == 2:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img.copy()

    for i, contour in enumerate(contours):
        for ii, point in enumerate(contour):
            cv2.circle(img_color, tuple(point[0]), 2, colors[i], 2)

    # Resize for display, preserving aspect ratio
    scaling_factor = 2
    new_width = img_color.shape[1] * scaling_factor
    new_height = img_color.shape[0] * scaling_factor
    resized_img = cv2.resize(img_color, (new_width, new_height))

    cv2.imshow(name, resized_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def draw_segmented_contours2(img, contours, name="Segmented Contours"):
    colors = generate_spaced_colors(len(contours))

    # Convert grayscale to BGR if necessary
    if len(img.shape) == 2:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img.copy()

    for i, contour in enumerate(contours):
        for point in contour:
            # Ensure point is in the format [x, y]
            if point.shape == (1, 2):
                x, y = point.ravel()
                cv2.circle(img_color, (x, y), 2, colors[i], 2)
            else:
                print("Invalid point format:", point)

    # Resize for display, preserving aspect ratio
    scaling_factor = 2
    new_width = img_color.shape[1] * scaling_factor
    new_height = img_color.shape[0] * scaling_factor
    resized_img = cv2.resize(img_color, (new_width, new_height))

    cv2.imshow(name, resized_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
