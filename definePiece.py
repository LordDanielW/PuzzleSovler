import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from itertools import combinations

from utilsLoad import (
    load_puzzle_pieces,
)

from utilsDraw import (
    draw_gradient_contours,
    plot_histogram,
    draw_segmented_contours,
    scale_piece,
    show_all,
)

from classPuzzle import PieceInfo, PuzzleInfo, SideInfo

debugVisuals = False
originalPath = "Puzzles/Original/"
shuffledPath = "Puzzles/Shuffled/"
solvedPath = "Puzzles/Solved/"


def find_contour(img, debugVisuals=False):
    inverted = 255 - img

    _, thresh = cv2.threshold(inverted, 25, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    if debugVisuals and len(contours) > 0:
        debugImgs = []
        debugImgs.append(scale_piece(img, "Original", 2, False, False))
        debugImgs.append(scale_piece(inverted, "Inverted", 2, False, False))
        debugImgs.append(scale_piece(thresh, "Thresholded", 2, False, False))
        debugImgs.append(
            draw_gradient_contours(img, contours[0], "Contoured", False, False)
        )
        show_all(
            debugImgs, "Original / Inverted / Thresholded / Contoured", 5, 1, True, True
        )

    return contours[0]


def calculate_angle(vec1, vec2):
    # Calculate the dot product
    dot = np.dot(vec1, vec2)
    # Calculate the determinant (cross product in 2D)
    det = vec1[0] * vec2[1] - vec1[1] * vec2[0]
    # Calculate the angle and convert to degrees
    angle = np.degrees(np.arctan2(det, dot))
    return int(angle)


def calculate_angle_differences(sample_points):
    number_of_points = len(sample_points)

    if number_of_points < 3:
        print("Not enough points to calculate angles.")
        raise

    angle_differences = []
    for i in range(len(sample_points)):
        pt0 = sample_points[i - 1][0]
        pt1 = sample_points[i][0]
        pt2 = sample_points[(i + 1) % len(sample_points)][0]

        vec1 = pt1 - pt0
        vec2 = pt2 - pt1

        angle_diff = calculate_angle(vec1, vec2)
        angle_differences.append(angle_diff)
    return angle_differences


def determine_final_corners(sample_points, peak_indices):
    peakCandidates = [sample_points[i] for i in peak_indices if i < len(sample_points)]

    cornerSetCandidates = increasing_combinations(
        len(peakCandidates) - 1, 4
    )  # sets of 4 ordered indices from peakCandidates
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
        peak_indices[i] for i in finalIndicies if i < len(peak_indices)
    ]

    return finalCorners, finalCornerIndicies


def segment_edges_and_calculate_histograms(
    sample_points,
    angle_differences,
    finalCornerIndicies,
    cornerTrim,
):
    edgeHistograms = []
    edgePoints = []
    for i in range(4):
        indexLow = finalCornerIndicies[i]
        indexHigh = finalCornerIndicies[(i + 1) % 4]

        currentIndex = (indexLow + cornerTrim) % len(angle_differences)
        isolatedEdge = []
        isolatedEdgePoints = []  # this is just for debugging currently
        while (currentIndex + cornerTrim - 1) % len(angle_differences) != indexHigh:
            isolatedEdge.append(angle_differences[currentIndex])
            isolatedEdgePoints.append(sample_points[currentIndex])
            currentIndex = (currentIndex + 1) % len(angle_differences)

        edgeHistograms.append(isolatedEdge)
        edgePoints.append(isolatedEdgePoints)

    return edgeHistograms, edgePoints


def filter_middle_peaks(piece, sample_points, localPeakIndicies):
    height, width = piece.shape[:2]
    width_quarter = width * 3 // 8
    width_three_quarters = 5 * width // 8
    height_quarter = height * 3 // 8
    height_three_quarters = 5 * height // 8

    filteredPeakIndicies = []
    for index in localPeakIndicies:
        x, y = sample_points[index][0]
        if not (width_quarter <= x <= width_three_quarters) and not (
            height_quarter <= y <= height_three_quarters
        ):
            filteredPeakIndicies.append(index)

    return filteredPeakIndicies


def simpleHistogram(data, name="Simple Histogram", show=True):
    plt.figure(figsize=(10, 5))
    plt.plot(data, marker="o", linestyle="-")
    plt.title(name)
    plt.xlabel("Index")
    plt.ylabel("Angle Difference (degrees/dist)")
    plt.grid(True)

    plt.show()


def circular_centered_window_filter(arr, window_width):
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


# Finds peaks in a circular array.
def find_peaks_circular(
    arr, window_width, min_peak_height, include_negative_peaks=False
):
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


# Calculates the area of a quadrilateral given by four vertices.
def shoelace_area(vertices):
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


# segments a piece into 4 sides.
# returns sides as normalized histograms in CW order
def segmentSides(
    piece, debugVis=False, downSampleFactor=4, cornerTrim=3, flat_edge_tolerance=10
):
    debugImgs = []

    # Contour
    contour = find_contour(piece, debugVis)

    # Downsample
    sample_points = contour[::downSampleFactor]
    if debugVis:
        debugImgs.append(
            draw_gradient_contours(
                piece, sample_points, "Downsampled Edge", False, False
            )
        )

    # Angle differences
    angle_differences = calculate_angle_differences(sample_points)

    # these variable control window size for filtration and corner detection
    # if the windowSize is increased, the peakHeight may need to be decreased and vice versa
    windowSize = 3
    minPeakHeight_deg = 10

    # Circular window filter
    angle_differences = circular_centered_window_filter(angle_differences, windowSize)

    # Plot the Integral
    integral_of_angles = np.cumsum(angle_differences)
    if debugVis:
        simpleHistogram(integral_of_angles, "Integral of Angle Differences", False)

    # Find Peaks
    localPeakindicies = find_peaks_circular(
        angle_differences, windowSize, minPeakHeight_deg, False
    )
    if debugVis:
        peakCandidates = [
            sample_points[i] for i in localPeakindicies if i < len(sample_points)
        ]
        debugImgs.append(
            draw_gradient_contours(piece, peakCandidates, "Found Peaks", False, False)
        )

    # Filter Middle Peaks
    # filteredPeakindicies = localPeakindicies
    filteredPeakindicies = filter_middle_peaks(piece, sample_points, localPeakindicies)
    if debugVis:
        filtered_candidates = [
            sample_points[i] for i in filteredPeakindicies if i < len(sample_points)
        ]
        debugImgs.append(
            draw_gradient_contours(
                piece, filtered_candidates, "Filtered Peaks", False, False
            )
        )

    # Find Corners
    finalCorners, finalCornerIndicies = determine_final_corners(
        sample_points, filteredPeakindicies
    )
    if debugVis:
        debugImgs.append(
            draw_gradient_contours(piece, finalCorners, "Final Corners", False, False)
        )

    # Segment Edges
    edgeHistograms, edgePoints = segment_edges_and_calculate_histograms(
        sample_points,
        angle_differences,
        finalCornerIndicies,
        cornerTrim,
    )
    if debugVis:
        debugImgs.append(
            draw_segmented_contours(piece, edgePoints, "Segmented Edges", False, False)
        )
        show_all(debugImgs, "Define Pieces", 5, 1, True, True)
        for i, hist in enumerate(edgeHistograms):
            # draw_gradient_contours(piece, edgePoints[i],"Edge Contour "+str(i))
            simpleHistogram(hist, "edge " + str(i))

    # Create PieceInfo object
    thisPiece = PieceInfo()

    # Define Sides
    for i in range(4):
        thisPiece.sides[i].Histogram = edgeHistograms[i]
        thisPiece.sides[i].Points = edgePoints[i]
        # thisPiece.sides[i].start_corner_index = finalCornerIndicies[i]
        # thisPiece.sides[i].end_corner_index = finalCornerIndicies[(i + 1) % 4]
        if (
            np.all(edgeHistograms[i] == 0)
            or np.sum(np.abs(edgeHistograms[i])) < flat_edge_tolerance
        ):
            thisPiece.sides[i].isEdge = True
        else:
            thisPiece.sides[i].isEdge = False

    # Define Piece
    thisPiece.puzzle_piece = piece
    thisPiece.puzzle_contours_all = contour
    thisPiece.puzzle_sampled_contours = sample_points
    thisPiece.corners = finalCorners[-1:] + finalCorners[:-1]  # rotate corners CW by 1

    count_Flat = 0
    for i in range(4):
        if thisPiece.sides[i].isEdge:
            count_Flat += 1
            thisPiece.isEdge = True
    if count_Flat == 2:
        thisPiece.isCorner = True
    else:
        thisPiece.isCorner = False

    # Rotate Piece so side 0 is top
    thisPiece.rotate_to_top()

    # Rotate Piece 4 times, append all images with edge points as contours, show all
    if debugVis:
        debugRots = []

        for i in range(4):
            allPoints = []
            allPoints += thisPiece.sides[0].Points
            # for side in thisPiece.sides:
            #     allPoints += side.Points
            debugRots.append(
                draw_gradient_contours(
                    thisPiece.puzzle_piece,
                    allPoints,
                    "Rot" + str(i + 1),
                    False,
                    False,
                )
            )
            thisPiece.rotate_piece_deep()
        show_all(debugRots, "Rotations", 5, 1, True, True)

    return thisPiece


def main():
    puzzle_name = "jigsaw3"
    pieces, _ = load_puzzle_pieces(os.path.join(shuffledPath, puzzle_name))

    for piece in pieces:
        segmentSides(piece, True, 4, 3, 10)


# Runs only if called as main file
if __name__ == "__main__":
    main()
