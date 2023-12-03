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

from puzzleClass import PieceInfo, PuzzleInfo, SideInfo, SideMatch

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


def findBestMatches(histogram, pieces_to_compare, pieces_index):
    pieces_to_compare: [PieceInfo]
    reversed_histogram = histogram[::-1]
    all_side_matches = []

    for piece in pieces_to_compare:
        piece: PieceInfo
        for side in piece.sides:
            side: SideInfo
            if side.isEdge == False:
                dis_sqrd = distance_squared_average(reversed_histogram, side.Histogram)
                side_match = SideMatch()
                side_match.piece_index = pieces_index
                side_match.side_index = side.side_Index
                side_match.histogram_score = dis_sqrd
                side.side_matches.append(side_match)

    sorted_side_matches = sorted(
        all_side_matches, key=lambda match: match.histogram_score
    )[:5]

    return sorted_side_matches


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


def solve_puzzle(puzzle_name, averaged_tolerance=9):
    raw_pieces, piecesInfo = load_puzzle_pieces(os.path.join(shuffledPath, puzzle_name))

    puzzle = PuzzleInfo()
    pieces_to_compare: [PieceInfo] = []

    for i, raw_piece in enumerate(raw_pieces):
        piece: PieceInfo = segmentSides(raw_piece, False, 4, 3)
        piece.piece_Index = i
        piece.piece_name = piecesInfo[i]["piece_name"]
        puzzle.pieces.append(piece)
        if i > 0:
            pieces_to_compare.append(piece)

    for i, tPiece in enumerate(puzzle.pieces[:-1]):
        tPiece: PieceInfo
        for ii, tSide in enumerate(tPiece.sides):
            tSide: SideInfo
            if tSide.isEdge == False:
                tSide.side_Matches = findBestMatches(
                    tSide.Histogram, pieces_to_compare, i
                )

        pieces_to_compare.pop(0)

    #  write_histogram_scores_to_csv(piecesInfo, "histogram_scores.csv")

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
