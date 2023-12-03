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

from utils import (
    find_bounding_box,
    rotate_image_easy,
    rotate_image,
    read_puzzle_pieces_info,
    load_puzzle_pieces,
)

from puzzleClass import PieceInfo, PuzzleInfo, SideInfo

debugVisuals = False
originalPath = "Puzzles/Original/"
shuffledPath = "Puzzles/Shuffled/"
solvedPath = "Puzzles/Solved/"


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


def draw_gradient_contours(img, contour, name="Colored Contours"):
    length = len(contour)

    # if length==1:
    #     print("hello")

    # return

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
    # cv2.destroyAllWindows()Heyu


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

    # for contour in contours:
    #     print(len(contour))

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


# segments a piece into 4 sides.
# returns sides as normalized histograms in CW order
def segmentSides(
    piece, debugVis=False, downSampleFactor=4, cornerTrim=3, flat_edge_tolerance=9
):
    contour = find_contour(piece, False)
    sample_points = contour[::downSampleFactor]

    angle_differences = []
    number_of_points = len(sample_points)
    if debugVis:
        draw_gradient_contours(piece, sample_points, "Downsampled Edge")

    if number_of_points < 3:
        print("Not enough points to calculate angles.")
        raise

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

    # these variable control window size for filtration and corner detection (feel free to change them as needed, but mind that they are related)
    # if the windowSize is increased, the peakHeight may need to be decreased and vice versa
    windowSize = 3
    minPeakHeight_deg = 10

    angle_differences = circular_centered_window_filter(angle_differences, windowSize)

    # find peaks
    localPeakindicies = find_peaks_circular(
        angle_differences, windowSize, minPeakHeight_deg, False
    )

    # peak candidates
    peakCandidates = [
        sample_points[i] for i in localPeakindicies if i < len(sample_points)
    ]

    if debugVis:
        draw_gradient_contours(piece, peakCandidates, "Peak Candidates")

    # ordered sets
    cornerSetCandidates = increasing_combinations(
        len(peakCandidates) - 1, 4
    )  # sets of 4 ordered indicies from peakCandidates

    bestScore = -1
    bestIndex = -1
    for i, ptIndicies in enumerate(cornerSetCandidates):
        orderedPoints = [
            peakCandidates[ii] for ii in ptIndicies if ii < len(peakCandidates)
        ]
        score = score_of_shape(orderedPoints, 15)
        if score != -1 and score > bestScore:
            bestScore = score
            bestIndex = i

    finalIndicies = cornerSetCandidates[bestIndex]
    finalCorners = [peakCandidates[i] for i in finalIndicies if i < len(peakCandidates)]
    finalCornerIndicies = [
        localPeakindicies[i] for i in finalIndicies if i < len(localPeakindicies)
    ]

    if debugVis:
        draw_gradient_contours(piece, finalCorners, "Final Corners")

    edgeHistograms = []
    edgePoints = []
    for i in range(4):
        indexLow = finalCornerIndicies[i]
        indexHigh = finalCornerIndicies[(i + 1) % 4]

        currentIndex = indexLow + cornerTrim
        isolatedEdge = []
        isolatedEdgePoints = []  # this is just for debugging currently
        while (currentIndex + cornerTrim - 1) % len(angle_differences) != indexHigh:
            isolatedEdge.append(angle_differences[currentIndex])
            isolatedEdgePoints.append(sample_points[currentIndex])
            currentIndex = (currentIndex + 1) % len(angle_differences)
        edgeHistograms.append(isolatedEdge)
        edgePoints.append(isolatedEdgePoints)

    if debugVis:
        draw_segmented_contours(piece, edgePoints)
        for i, hist in enumerate(edgeHistograms):
            # draw_gradient_contours(piece, edgePoints[i],"Edge Contour "+str(i))
            simpleHistogram(hist, "edge " + str(i))

    thisPiece = PieceInfo()

    # Define the piece's sides
    for i in range(4):
        thisPiece.sides[i].side_Index = i
        thisPiece.sides[i].Histogram = edgeHistograms[i]
        thisPiece.sides[i].start_corner_index = finalCornerIndicies[i]
        thisPiece.sides[i].end_corner_index = finalCornerIndicies[(i + 1) % 4]
        if (
            np.all(edgeHistograms[i] == 0)
            or np.sum(np.abs(edgeHistograms[i])) < flat_edge_tolerance
        ):
            thisPiece.sides[i].isEdge = True
        else:
            thisPiece.sides[i].isEdge = False

    # Define the piece
    thisPiece.puzzle_piece = piece
    thisPiece.puzzle_contours_all = contour
    thisPiece.puzzle_sampled_contours = sample_points

    count_Flat = 0
    for i in range(4):
        if thisPiece.sides[i].isEdge:
            count_Flat += 1
            thisPiece.isEdge = True
    if count_Flat == 2:
        thisPiece.isCorner = True
    else:
        thisPiece.isCorner = False

    (
        thisPiece.top_y,
        thisPiece.left_x,
        thisPiece.bottom_y,
        thisPiece.right_x,
    ) = find_bounding_box(contour)

    return thisPiece


def create_angle_histogram(sample_points, piece):
    angle_differences = []
    number_of_points = len(sample_points)

    draw_gradient_contours(piece, sample_points, "downsampled contour")

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

    windowSize = 3
    minPeakHeight_deg = 10

    # Apply Gausian Blur (filter)
    # angle_differences = gaussian_filter(angle_differences, 1.0)
    angle_differences = circular_centered_window_filter(angle_differences, windowSize)

    # find peaks
    localPeakindicies = find_peaks_circular(
        angle_differences, windowSize, minPeakHeight_deg, False
    )

    # peak candidates
    peakCandidates = [
        sample_points[i] for i in localPeakindicies if i < len(sample_points)
    ]

    draw_gradient_contours(piece, peakCandidates, "peak candidates")

    # ordered sets
    cornerSetCandidates = increasing_combinations(
        len(peakCandidates) - 1, 4
    )  # sets of 4 ordered indicies from peakCandidates

    bestScore = -1
    bestIndex = -1
    for i, ptIndicies in enumerate(cornerSetCandidates):
        orderedPoints = [
            peakCandidates[ii] for ii in ptIndicies if ii < len(peakCandidates)
        ]
        score = score_of_shape(orderedPoints, 15)
        if score != -1 and score > bestScore:
            bestScore = score
            bestIndex = i

    finalIndicies = cornerSetCandidates[bestIndex]
    finalCorners = [peakCandidates[i] for i in finalIndicies if i < len(peakCandidates)]

    draw_gradient_contours(piece, finalCorners, "Final corners")

    # Plot the Integral
    # Compute the cumulative sum which acts as an integral of the angle differences
    integral_of_angles = np.cumsum(angle_differences)

    # # print("Sum of Angle Difference: ", sum(angle_differences))
    # # print("Sample Points: ", len(sample_points))
    # print("Angle Differences: ", angle_differences)
    # print("integral_of_angles: ", integral_of_angles)

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.plot(angle_differences, marker="o", linestyle="-")
    plt.title("Integral of Angle Differences")
    plt.xlabel("Index")
    plt.ylabel("Cumulative Angle Difference (degrees)")
    plt.grid(True)
    # plt.show()

    # Create the plot
    # plot_histogram(angle_differences)

    plt.show()

    return angle_differences


def simpleHistogram(data, name="Simple Histogram"):
    plt.figure(figsize=(10, 5))
    plt.plot(data, marker="o", linestyle="-")
    plt.title(name)
    plt.xlabel("Index")
    plt.ylabel("Angle Difference (degrees/dist)")
    plt.grid(True)

    plt.show()


def circular_centered_window_filter(arr, window_width):
    """
    Applies a moving average filter to a circular array of integers, centering
    the window around each element.

    Parameters:
    arr (list of int): The input array, considered as circular.
    window_width (int): The width of the moving window.

    Returns:
    list of float: The filtered array, with each element being the average
                   of the window. The length of the output list will be
                   equal to len(arr).
    """
    if window_width <= 0:
        raise ValueError("Window width must be greater than 0")
    if window_width > len(arr):
        raise ValueError("Window width cannot be greater than the array length")
    if window_width % 2 == 0:
        raise ValueError("Window width must be odd to center it around each element")

    n = len(arr)
    half_window = window_width // 2
    filtered = []

    for i in range(n):
        window_sum = 0
        # Iterate over the window elements considering the circular nature
        for j in range(-half_window, half_window + 1):
            # Index wraps around if it goes past the boundaries of the array
            index = (i + j) % n
            window_sum += arr[index]

        window_average = window_sum / window_width
        filtered.append(window_average)

    return filtered


def find_peaks_circular(
    arr, window_width, min_peak_height, include_negative_peaks=False
):
    """
    Finds peaks in a circular array.

    Parameters:
    arr (list of int): The input array, considered as circular.
    window_width (int): The width of the window for considering a peak.
    min_peak_height (int): The minimum height to consider an element a peak.
    include_negative_peaks (bool): If True, considers both positive and negative peaks;
                                   otherwise, only positive peaks.

    Returns:
    list of int: Indices of the peaks in the array.
    """
    if window_width <= 0:
        raise ValueError("Window width must be greater than 0")
    if window_width > len(arr):
        raise ValueError("Window width cannot be greater than the array length")
    if window_width % 2 == 0:
        raise ValueError("Window width must be odd for symmetry in the neighborhood")

    n = len(arr)
    half_window = window_width // 2
    peak_indices = []

    for i in range(n):
        peak_candidate = arr[i]
        if include_negative_peaks and abs(peak_candidate) < min_peak_height:
            continue
        if not include_negative_peaks and (peak_candidate < min_peak_height):
            continue

        is_peak = True
        for j in range(-half_window, half_window + 1):
            if j == 0:
                continue  # Skip the center element itself
            neighbor_index = (i + j) % n
            if include_negative_peaks:
                if abs(arr[neighbor_index]) > abs(peak_candidate):
                    is_peak = False
                    break
            else:
                if arr[neighbor_index] > peak_candidate:
                    is_peak = False
                    break

        if is_peak:
            peak_indices.append(i)

    return peak_indices


def increasing_combinations(n, combo_size):
    """
    Generate unique combinations of numbers between 0 and n (inclusive),
    ensuring numbers in a combination are always increasing.

    Parameters:
    n (int): The upper bound of the number range.
    combo_size (int): The size of each combination.

    Returns:
    list of tuples: A list of unique combinations.
    """
    if combo_size <= 0 or combo_size > n + 1:
        raise ValueError("Invalid combination size")

    # Generate standard increasing combinations
    return list(combinations(range(n + 1), combo_size))


# def calculate_angle_2(v1, v2):
#     """Calculate the angle between two vectors."""
#     dot_product = v1[0]*v2[0] + v1[1]*v2[1]
#     magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
#     magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
#     cos_angle = dot_product / (magnitude_v1 * magnitude_v2)
#     angle_radians = math.acos(cos_angle)
#     return math.degrees(angle_radians)


def score_of_shape(vertices, angleTolerance_deg):
    if len(vertices) != 4:
        print("rip")
        print(vertices)
        return -1
        # raise ValueError("There must be exactly four vertices.")

    badCornerFlag = False
    for i in range(4):
        # Calculate vectors
        vertex = vertices[i]
        val = type(vertices)
        v1 = (
            vertices[i][0][0] - vertices[i - 1][0][0],
            vertices[i][0][1] - vertices[i - 1][0][1],
        )
        v2 = (
            vertices[(i + 1) % 4][0][0] - vertices[i][0][0],
            vertices[(i + 1) % 4][0][1] - vertices[i][0][1],
        )

        # Calculate angle at vertex
        angle = calculate_angle(v1, v2)

        if abs(angle - 90) > angleTolerance_deg:
            badCornerFlag = True
    if badCornerFlag:
        return -1

    # distn = pow(vertices[i][0][0] - vertices[(i+1)%4][0][0],2) + pow(vertices[i][0][1] - vertices[(i+1)%4][0][1],2)
    dist1 = pow(vertices[0][0][0] - vertices[1][0][0], 2) + pow(
        vertices[0][0][1] - vertices[1][0][1], 2
    )
    dist2 = pow(vertices[1][0][0] - vertices[2][0][0], 2) + pow(
        vertices[1][0][1] - vertices[2][0][1], 2
    )
    dist3 = pow(vertices[2][0][0] - vertices[3][0][0], 2) + pow(
        vertices[2][0][1] - vertices[3][0][1], 2
    )
    dist4 = pow(vertices[3][0][0] - vertices[0][0][0], 2) + pow(
        vertices[3][0][1] - vertices[0][0][1], 2
    )

    score = shoelace_area(vertices)

    return score


def shoelace_area(vertices):
    """
    Calculates the area of a quadrilateral given by four vertices.

    Parameters:
    vertices (list of tuples): A list of four (x, y) coordinates representing the vertices of the shape.

    Returns:
    float: The area of the shape.
    """
    if len(vertices) != 4:
        raise ValueError("There must be exactly four vertices.")

    # Add the first vertex to the end to close the loop
    vertices.append(vertices[0])

    area = 0
    for i in range(4):
        x1, y1 = vertices[i][0]
        x2, y2 = vertices[i + 1][0]
        area += (x1 * y2) - (x2 * y1)

    return abs(area) / 2


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
    puzzle_name = "jigsaw6"
    pieces = load_puzzle_pieces(os.path.join(shuffledPath, puzzle_name))

    for i, piece in enumerate(pieces):
        segmentSides(piece, True, 4, 3)

        # print("Piece: ", i)
        # # Find the contours of the puzzle piece
        # contour = find_contour(piece,False)
        # # Resample the contour by taking every '4' points
        # sample_points = contour[::4]
        # angle_histogram = create_angle_histogram(sample_points, piece)

        # flat_sides = find_flat_side(angle_histogram)
        # # Draw the flat sides over the puzzle piece
        # draw_flat_sides_on_piece(piece, sample_points, flat_sides)


# Runs only if called as main file
if __name__ == "__main__":
    main()
