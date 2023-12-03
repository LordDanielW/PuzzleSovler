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

from puzzleClass import PieceInfo, PuzzleInfo, SideInfo, SideMatch, PuzzleSolve

from definePiece import segmentSides

# Paths
shuffledPath = "Puzzles/Shuffled/"
solvedPath = "Puzzles/Solved/"


def distance_squared_average(array1, array2, shift_range=3):
    array1 = np.array(array1)
    array2 = np.array(array2)
    min_length = min(len(array1), len(array2))

    min_distance = float("inf")
    best_shift = 0

    # Try shifting array2 within the range [-shift_range, shift_range]
    for shift in range(-shift_range, shift_range + 1):
        shifted_array2 = np.roll(array2, shift)
        truncated_array1 = array1[:min_length]
        truncated_array2 = shifted_array2[:min_length]
        squared_diff = np.square(truncated_array1 - truncated_array2)
        avg_squared_distance = np.mean(squared_diff)

        if avg_squared_distance < min_distance:
            min_distance = avg_squared_distance
            best_shift = shift

    return min_distance, best_shift


def findBestMatches(histogram, pieces_to_compare, pieces_index):
    pieces_to_compare: [PieceInfo]
    reversed_histogram = histogram[::-1]
    all_side_matches = []

    for piece in pieces_to_compare:
        piece: PieceInfo
        for side in piece.sides:
            side: SideInfo
            if side.isEdge == False:
                dis_sqrd, shift = distance_squared_average(
                    reversed_histogram, side.Histogram
                )
                side_match = SideMatch()
                side_match.piece_index = pieces_index
                side_match.side_index = side.side_Index
                side_match.histogram_score = dis_sqrd
                side_match.histogram_shift = shift
                side.side_matches.append(side_match)

    sorted_side_matches = sorted(
        all_side_matches, key=lambda match: match.histogram_score
    )[:5]

    return sorted_side_matches


# Function to find the best match for a given side
def find_best_match_for_side(side, pieces):
    for side_match in side.side_matches:
        for piece in pieces:
            if piece.piece_Index == side_match.piece_index:
                return piece, side_match


def findBestPuzzle(puzzleSolve, piece, pieces_left):
    puzzleSolve: PuzzleSolve
    piece: PieceInfo
    pieces_left: [PieceInfo]

    if not pieces_left:
        return puzzleSolve

    y_piece_index, x_piece_index = puzzleSolve.find_piece(piece.piece_Index)

    # Place piece for side 2 (bottom)
    if not piece.sides[2].isEdge:
        best_match_piece, best_match = find_best_match_for_side(
            piece.sides[2], pieces_left
        )
        if best_match_piece:
            # Remove matched piece from pieces_left and update puzzleSolve
            y_pieces_left = pieces_left.copy()
            y_pieces_left.remove(best_match_piece)
            best_match_piece.rotate_sides()  # Rotate to align the matched side correctly
            while best_match_piece.sides[0].side_Index != best_match.side_index:
                best_match_piece.rotate_sides()

            # Add the piece to the puzzle matrix and update the score
            new_y_piece_index = y_piece_index + 1
            puzzleSolve.puzzle_matrix[
                new_y_piece_index, x_piece_index
            ] = best_match_piece
            puzzleSolve.puzzle_score += best_match.histogram_score

            # Recursive call with the next piece
            findBestPuzzle(puzzleSolve, best_match_piece, y_pieces_left)

    # Place piece for side 1 (right)
    if not piece.sides[1].isEdge:
        best_match_piece, best_match = find_best_match_for_side(
            piece.sides[1], pieces_left
        )
        if best_match_piece:
            # Remove matched piece from pieces_left and update puzzleSolve
            pieces_left.remove(best_match_piece)
            best_match_piece.rotate_sides()  # Rotate to align the matched side correctly
            while best_match_piece.sides[3].side_Index != best_match.side_index:
                best_match_piece.rotate_sides()

            # Add the piece to the puzzle matrix and update the score
            new_x_piece_index = x_piece_index + 1
            puzzleSolve.puzzle_matrix[
                y_piece_index, new_x_piece_index
            ] = best_match_piece
            puzzleSolve.puzzle_score += best_match.histogram_score

            # Recursive call with the next piece
            findBestPuzzle(puzzleSolve, best_match_piece, pieces_left)

    return puzzleSolve


# def findBestBorder(start_piece, corner_pieces, edge_pieces):
#     start_piece: PieceInfo
#     corner_pieces: [PieceInfo]
#     edge_pieces: [PieceInfo]


def solve_puzzle(puzzle_name):
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

    for tPiece in puzzle.pieces[:-1]:
        tPiece: PieceInfo
        for tSide in tPiece.sides:
            tSide: SideInfo
            if tSide.isEdge == False:
                tSide.side_matches = findBestMatches(
                    tSide.Histogram, pieces_to_compare, tPiece.piece_Index
                )

        pieces_to_compare.pop(0)

    corner_pieces = [PieceInfo()]
    edge_pieces = [PieceInfo()]
    middle_pieces = [PieceInfo()]
    for piece in puzzle.pieces:
        if piece.isCorner:
            corner_pieces.append(piece)
            edge_pieces.append(piece)
        elif piece.isEdge:
            edge_pieces.append(piece)
        else:
            middle_pieces.append(piece)

    puzzleSolve = PuzzleSolve()
    first_piece = corner_pieces[0]
    counter = 0
    while not (first_piece.sides[3].isEdge and first_piece.sides[0].isEdge):
        if counter == 4:
            break
        first_piece.rotate_sides()
        counter += 1
    puzzleSolve.pieces.append(0, 0, first_piece)
    puzzle_pieces_left = puzzle.pieces
    bestSolve = findBestPuzzle(first_piece, puzzleSolve, puzzle_pieces_left.copy())

    write_placement_to_csv(bestSolve, "puzzle_placement.csv")
    # # Save the solved puzzle
    # print(f"Saving solved puzzle... {puzzle_name}_solved.png")
    # cv2.imwrite(os.path.join(solvedPath, f"{puzzle_name}_solved.png"), solvedPuzzle)
    # return solvedPuzzle


def write_placement_to_csv(puzzleSolve, filename):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["y_piece_index", "x_piece_index", "piece_index"])

        for placement in puzzleSolve.placement_details:
            writer.writerow(placement)


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
