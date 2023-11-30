import cv2
import csv
import numpy as np
import os
import glob

from utils import (
    find_bounding_box,
    rotate_image_easy,
    rotate_image,
    read_puzzle_pieces_info,
    load_puzzle_pieces,
)

# Paths
shuffledPath = "Puzzles/Shuffled/"
solvedPath = "Puzzles/Solved/"


def contour_to_DirectionalVector(contour):
    # Calculate the directional vectors
    directionalVectors = np.diff(contour, axis=0)

    # Append the first point to the end to close the contour and compute the last vector
    closed_contour = np.vstack((contour, contour[0]))
    last_vector = np.diff(closed_contour[-2:], axis=0)

    # Append the last vector to the list of directional vectors
    directionalVectors = np.vstack((directionalVectors, last_vector))

    # print(directionalVectors)
    return directionalVectors


def rotate_DirectionalVector(directionalVectors, angle_flag):
    # Define a mapping from OpenCV rotation constants to angles in degrees
    rotation_angles = {
        cv2.ROTATE_90_CLOCKWISE: -90,
        cv2.ROTATE_180: 180,
        cv2.ROTATE_90_COUNTERCLOCKWISE: 90,
    }

    # Map the OpenCV rotation constant to an angle in degrees
    angle_deg = rotation_angles[angle_flag]

    # Convert angle to radians for numpy
    angle_rad = np.deg2rad(angle_deg)

    # Create the rotation matrix
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # Apply the rotation matrix to each directional vector
    rotated_vectors = np.dot(directionalVectors, rotation_matrix)

    # Now, rotated_vectors contains the directional vectors rotated by the specified angle.
    return rotated_vectors

def find_match(dirVec, dirVecList):
    dirVec *= -1 # Invert the vector to find the match
    tolerance = 0.04 

    for index, directional_vector in enumerate(dirVecList):
        match = True
        match_index = -1

        for i, vec in enumerate(directional_vector):
            

    return match, match_index

def find_edges(dirVecList, pieceInfo):
    dirVecEdges = []
    edgePieceInfo = []

    num_samples = 8 
    tolerance = 0.02 
    percentFlat = 0.15

    for index, directional_vector in enumerate(dirVecList):
        sample_indices = np.linspace(0, len(directional_vector) - 1, num_samples, dtype=int)
        aligned_count = 0
        x_edge_start = -1
        y_edge_start = -1

        for i in sample_indices:
            vec = directional_vector[i]
            angle_with_x = np.arctan2(vec[1], vec[0]) * 180 / np.pi
            angle_with_y = np.arctan2(vec[0], vec[1]) * 180 / np.pi

            if abs(angle_with_x) <= tolerance * 90 or abs(angle_with_y) <= tolerance * 90:
                aligned_count += 1
                if x_edge_start == -1 and abs(angle_with_x) <= tolerance * 90:
                    x_edge_start = i
                if y_edge_start == -1 and abs(angle_with_y) <= tolerance * 90:
                    y_edge_start = i

        if (aligned_count / num_samples) >= percentFlat:
            dirVecEdges.append(directional_vector)
            edgePieceInfo.append({
                'index': len(dirVecEdges) - 1,
                'return_index': index,
                'x-edge start': x_edge_start,
                'y-edge start': y_edge_start
            })

    return dirVecEdges, edgePieceInfo


def solve_puzzle(puzzle_name):
    pieces = load_puzzle_pieces(os.path.join(shuffledPath, puzzle_name))

    # TODO

    # Save the solved puzzle
    print(f"Saving solved puzzle... {puzzle_name}_solved.png")
    cv2.imwrite(os.path.join(solvedPath, f"{puzzle_name}_solved.png"), solvedPuzzle)
    return solvedPuzzle


# # Example usage
# puzzle_name = "jigsaw2"  # replace with actual puzzle name
# sorted_pieces = solve_puzzle(puzzle_name)
# Loop through each puzzle directory in shuffledPath
for puzzle_folder in os.listdir(shuffledPath):
    puzzle_folder_path = os.path.join(shuffledPath, puzzle_folder)
    # Check if it's a directory
    if os.path.isdir(puzzle_folder_path):
        print(f"Solving puzzle: {puzzle_folder}")
        sorted_pieces = solve_puzzle(puzzle_folder)
