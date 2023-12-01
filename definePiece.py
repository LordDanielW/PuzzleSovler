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
    # cv2.destroyAllWindows()Heyu


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
    plt.xticks(range(0, num_pts, 10))

    # Set the y-axis limit to 180 degrees
    plt.ylim(min_angle, max_angle)
    plt.yticks(range(min_angle, max_angle, 10))

    # plt.show()


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

    # Apply Gausian Blur (filter)
    angle_differences = gaussian_filter(angle_differences, 1.0)

    # Plot the Integral
    # Compute the cumulative sum which acts as an integral of the angle differences
    integral_of_angles = np.cumsum(angle_differences)

    # # print("Sum of Angle Difference: ", sum(angle_differences))
    # # print("Sample Points: ", len(sample_points))
    # print("Angle Differences: ", angle_differences)
    # print("integral_of_angles: ", integral_of_angles)

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(integral_of_angles, marker="o", linestyle="-")
    plt.title("Integral of Angle Differences")
    plt.xlabel("Index")
    plt.ylabel("Cumulative Angle Difference (degrees)")
    plt.grid(True)
    # plt.show()

    # draw_gradient_contours(piece, sample_points)
    # Create the plot
    # plot_histogram(angle_differences)

    plt.show()

    return angle_differences


def find_key_points(angle_differences, peak_threshold):
    # Identify all local maxima and minima
    key_points = []
    for i in range(1, len(angle_differences) - 1):
        if (
            angle_differences[i - 1] < angle_differences[i] > angle_differences[i + 1]
            and angle_differences[i] > peak_threshold
        ):
            key_points.append((i, "max"))
        elif (
            angle_differences[i - 1] > angle_differences[i] < angle_differences[i + 1]
            and angle_differences[i] < -peak_threshold
        ):
            key_points.append((i, "min"))
    return key_points


def find_flat_side(angle_differences, peak_threshold=10, min_flat_length=5):
    flat_sides = []

    key_points = find_key_points(angle_differences, peak_threshold)

    # Find consecutive maxima without a minima in between
    i = 0
    while i < (len(key_points) - 1):
        if key_points[i][1] == "max":
            # Look ahead to find the next max without a min in between
            for j in range(i + 1, len(key_points)):
                if key_points[j][1] == "min":
                    break
                if key_points[j][1] == "max":
                    # Found two consecutive maxima, now check for flatness between them
                    start_max = key_points[i][0]
                    end_max = key_points[j][0]
                    flat_length = 0
                    start = None
                    for k in range(start_max, end_max + 1):
                        if abs(angle_differences[k]) <= peak_threshold:
                            if start is None:
                                start = k
                            flat_length += 1
                        else:
                            # If we encounter a non-flat angle, check if we just passed a flat edge
                            if flat_length >= min_flat_length:
                                # end = k - 1
                                flat_sides.append((start_max, end_max))
                                break  # Stop checking since this is no longer a flat edge
                            start = None
                            flat_length = 0
                    i = j  # Move the index to the next maxima
                    break
        i += 1

    print("Flat Sides:", len(flat_sides))
    return flat_sides


def draw_flat_sides_on_piece(piece, sample_points, flat_sides):
    # Copy the image to draw on
    piece_with_flats = piece.copy()

    # Convert grayscale to BGR if necessary
    if len(piece_with_flats.shape) == 2 or piece_with_flats.shape[2] == 1:
        piece_with_flats = cv2.cvtColor(piece_with_flats, cv2.COLOR_GRAY2BGR)

    # Iterate over the flat sides and draw them on the image
    for start_idx, end_idx in flat_sides:
        for i in range(start_idx, end_idx + 1):
            # Draw the segment of the flat side
            start_point = tuple(sample_points[i][0])
            end_point = tuple(sample_points[(i + 1) % len(sample_points)][0])
            cv2.line(piece_with_flats, start_point, end_point, (0, 255, 0), 2)

    # Resize for display, preserving aspect ratio
    scaling_factor = 4
    new_width = piece_with_flats.shape[1] * scaling_factor
    new_height = piece_with_flats.shape[0] * scaling_factor
    resized_img = cv2.resize(piece_with_flats, (new_width, new_height))

    # Display the image with the drawn flat sides
    cv2.imshow("Piece with Flat Sides", resized_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def main():
    puzzle_name = "jigsaw1"
    pieces = load_puzzle_pieces(os.path.join(shuffledPath, puzzle_name))

    for i, piece in enumerate(pieces):
        print("Piece: ", i)
        # Find the contours of the puzzle piece
        contour = find_contour(piece)
        # Resample the contour by taking every '4' points
        sample_points = contour[::4]
        angle_histogram = create_angle_histogram(sample_points, piece)

        # flat_sides = find_flat_side(angle_histogram)
        # # Draw the flat sides over the puzzle piece
        # draw_flat_sides_on_piece(piece, sample_points, flat_sides)


# Runs only if called as main file
if __name__ == "__main__":
    main()
