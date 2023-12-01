import cv2
import csv
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from utils import (
    find_bounding_box,
    rotate_image_easy,
    rotate_image,
    read_puzzle_pieces_info,
    load_puzzle_pieces,
)

debugVisuals = False
originalPath = "Puzzles/Original/"
shuffledPath = "Puzzles/Shuffled/"
solvedPath = "Puzzles/Solved/"


def draw_gradient_contours(img, contour):
    length = len(contour)

    # Convert grayscale to BGR if necessary
    if len(img.shape) == 2:
        img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img.copy()

    for i, point in enumerate(contour):
        ratio = i / (length - 1)  # adjusted to account for the endpoint
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

        cv2.circle(img_color, tuple(point[0]), 1, color, -1)

    # Resize for display, preserving aspect ratio
    scaling_factor = 4
    new_width = img_color.shape[1] * scaling_factor
    new_height = img_color.shape[0] * scaling_factor
    resized_img = cv2.resize(img_color, (new_width, new_height))

    cv2.imshow("Colored Contours", resized_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def find_contour(img, debugVisuals=False):
    inverted = 255 - img
    if debugVisuals:
        cv2.imshow("Inverted", inverted)
        cv2.waitKey(0)

    _, thresh = cv2.threshold(inverted, 25, 255, cv2.THRESH_BINARY)
    if debugVisuals:
        cv2.imshow("Thresholded", thresh)
        cv2.waitKey(0)

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if debugVisuals and len(contours) > 0:
        draw_gradient_contours(img, contours[0])

    return contours[0]


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
    plt.xticks(range(0, num_pts, max(1, num_pts // 20)))

    # Set the y-axis limit to 180 degrees
    plt.ylim(min_angle, max_angle)
    plt.yticks(range(min_angle, max_angle, max(1, max_angle // 20)))

    plt.show()


def calculate_angle(vec1, vec2):
    # Calculate the dot product
    dot = np.dot(vec1, vec2)
    # Calculate the determinant (cross product in 2D)
    det = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    # Calculate the angle and convert to degrees
    angle = np.degrees(np.arctan2(det, dot))
    return int(angle)


def create_angle_histogram(sample_points, piece):
    angle_differences = []
    number_of_points = len(sample_points)

    if number_of_points < 3:
        print("Not enough points to calculate angles.")
        return []

    # Calculate the vectors and angles between them
    for i in range(number_of_points):
        pt0 = sample_points[i - 1][
            0
        ]  # Previous point, wrapping around to the last for the first
        pt1 = sample_points[i][0]  # Current point
        pt2 = sample_points[(i + 1) % number_of_points][
            0
        ]  # Next point, wrapping around

        vec1 = pt1 - pt0
        vec2 = pt2 - pt1

        angle_diff = calculate_angle(vec1, vec2)
        angle_differences.append(angle_diff)

    # print("Sum of Angle Difference: ", sum(angle_differences))
    # print("Sample Points: ", len(sample_points))
    # print("Angle Differences: ", len(angle_differences))

    # Apply Gausian Blur (filter)
    angle_differences = gaussian_filter(angle_differences, 0.50)

    # Plot the Integral
    # Compute the cumulative sum which acts as an integral of the angle differences
    integral_of_angles = np.cumsum(angle_differences)

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(integral_of_angles, marker="o", linestyle="-")
    plt.title("Integral of Angle Differences")
    plt.xlabel("Index")
    plt.ylabel("Cumulative Angle Difference (degrees)")
    plt.grid(True)
    # plt.show()

    draw_gradient_contours(piece, sample_points)
    # Create the plot
    plot_histogram(angle_differences)

    return angle_differences


def main():
    puzzle_name = "jigsaw1"
    pieces = load_puzzle_pieces(os.path.join(shuffledPath, puzzle_name))

    for i, piece in enumerate(pieces):
        # Find the contours of the puzzle piece
        contour = find_contour(piece)
        # Resample the contour by taking every '4' points
        sample_points = contour[::4]
        angle_histogram = create_angle_histogram(sample_points, piece)


# Runs only if called as main file
if __name__ == "__main__":
    main()
