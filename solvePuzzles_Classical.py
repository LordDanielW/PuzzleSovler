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

from definePiece import segmentSides

# Paths
shuffledPath = "Puzzles/Shuffled/"
solvedPath = "Puzzles/Solved/"


# TODO gradient decent and wiggling
def distance_squared_average(array1, array2):
    array1 = np.array(array1)
    array2 = np.array(array2)
    min_length = min(len(array1), len(array2))
    truncated_array1 = array1[:min_length]
    truncated_array2 = array2[:min_length]
    squared_diff = np.square(truncated_array1 - truncated_array2)
    avg_squared_distance = np.mean(squared_diff)
    return avg_squared_distance


def findBestMatches(histogram, histograms_to_compare, pieces_index):
    reversed_histogram = histogram[::-1]
    distance_squared_dict = {}  # Initialize an empty dictionary

    for i, piece_histograms in enumerate(histograms_to_compare):
        for j, side_histo in enumerate(piece_histograms):
            if side_histo != "Flat_Edge":
                dis_sqrd = distance_squared_average(reversed_histogram, side_histo)
                # Store the distance squared in the dictionary with the key as a tuple
                # print(dis_sqrd)
                distance_squared_dict[(pieces_index + i, j)] = dis_sqrd

    sorted_distance_squared_dict = dict(
        sorted(distance_squared_dict.items(), key=lambda item: item[1])[:5]
    )
    print(sorted_distance_squared_dict)
    return sorted_distance_squared_dict


def write_histogram_scores_to_csv(piecesInfo, filename):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        # Writing the header
        writer.writerow(["Piece Index", "Side Index", "Histogram Scores"])

        # Iterating through each piece and its histogram scores
        for piece_index, pInfo in enumerate(piecesInfo):
            if "histogram_score" in pInfo:
                for side_index, score in pInfo["histogram_score"].items():
                    # Writing the piece index, side index, and the histogram score
                    writer.writerow([piece_index, side_index, score])


# def findBestPuzzle():


def solve_puzzle(puzzle_name):
    pieces, piecesInfo = load_puzzle_pieces(os.path.join(shuffledPath, puzzle_name))
    histograms_to_compare = []
    averaged_tolerance = 1

    for i, piece in enumerate(pieces):
        side_histograms = segmentSides(piece, False, 4, 3)

        # Flag Flat Edges to avoid compare, while still keeping consistent spacing
        for j, histo in enumerate(side_histograms):
            if np.all(histo == 0) or np.mean(histo) < averaged_tolerance:
                side_histograms[j] = "Flat_Edge"

        piecesInfo[i]["side_histograms"] = side_histograms
        if i > 0:
            histograms_to_compare.append(side_histograms)

    for i, pInfo in enumerate(piecesInfo[:-1]):
        for ii, histogram in enumerate(pInfo["side_histograms"]):
            if histogram != "Flat_Edge":
                # piecesInfo[i][ii]["histogram_score"] = findBestMatches(
                #     histogram, histograms_to_compare, i
                # )
                findBestMatches(histogram, histograms_to_compare, i)

        histograms_to_compare.pop(0)

    # write_histogram_scores_to_csv(piecesInfo, "histogram_scores.csv")

    # # Save the solved puzzle
    # print(f"Saving solved puzzle... {puzzle_name}_solved.png")
    # cv2.imwrite(os.path.join(solvedPath, f"{puzzle_name}_solved.png"), solvedPuzzle)
    # return solvedPuzzle


# Example usage
puzzle_name = "jigsaw1"  # replace with actual puzzle name
sorted_pieces = solve_puzzle(puzzle_name)

# # Loop through each puzzle directory in shuffledPath
# for puzzle_folder in os.listdir(shuffledPath):
#     puzzle_folder_path = os.path.join(shuffledPath, puzzle_folder)
#     # Check if it's a directory
#     if os.path.isdir(puzzle_folder_path):
#         print(f"Solving puzzle: {puzzle_folder}")
#         sorted_pieces = solve_puzzle(puzzle_folder)
