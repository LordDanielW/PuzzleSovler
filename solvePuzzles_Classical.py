import cv2
import csv
import numpy as np
import os
import matplotlib.pyplot as plt
import glob

from utils import (
    find_bounding_box,
    rotate_image_easy,
    rotate_image,
    read_puzzle_pieces_info,
    load_puzzle_pieces,
    draw_gradient_contours,
    plot_histogram,
    generate_spaced_colors,
    draw_segmented_contours,
    draw_segmented_contours2,
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


def findBestMatches(current_piece, side_Index, pieces_to_compare):
    current_piece: PieceInfo
    pieces_to_compare: [PieceInfo]
    side = current_piece.sides[side_Index]
    reversed_histogram = np.array(side.Histogram)[::-1]
    inv_rev_hist = np.array(reversed_histogram) * -1
    all_side_matches = []

    best_dis_sqrd = 1000000000
    best_match = None
    best_piece = None
    best_side = None

    for piece in pieces_to_compare:
        piece: PieceInfo
        for side in piece.sides:
            side: SideInfo
            if side.isEdge == False:
                dis_sqrd, shift = distance_squared_average(inv_rev_hist, side.Histogram)
                side_match = SideMatch()
                side_match.piece_index = piece.piece_Index
                side_match.side_index = side.side_Index
                side_match.histogram_score = dis_sqrd
                side_match.histogram_shift = shift
                side.side_matches.append(side_match)
                if dis_sqrd < best_dis_sqrd:
                    best_dis_sqrd = dis_sqrd
                    best_match = side.Histogram
                    best_piece = piece
                    best_side = side

    # if best_piece:
    #     plot_all(
    #         inv_rev_hist,
    #         best_match,
    #         current_piece.puzzle_piece,
    #         side.Points,
    #         best_piece.puzzle_piece,
    #         best_side.Points,
    #     )

    sorted_side_matches = sorted(
        all_side_matches, key=lambda match: match.histogram_score
    )[:5]

    return sorted_side_matches


def plot_all(
    reversed_histogram,
    best_match_histogram,
    current_puzzle_piece,
    current_points,
    best_puzzle_piece,
    best_points,
):
    draw_segmented_contours2(current_puzzle_piece, current_points, "current_piece")
    draw_segmented_contours2(best_puzzle_piece, best_points, "best_piece")

    plt.figure(figsize=(10, 6))
    plt.plot(reversed_histogram, label="Reversed Histogram")
    plt.plot(best_match_histogram, label="Best Match Histogram")
    plt.title("Histogram Comparison")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


# Function to find the best match for a given side
def find_best_match_for_side(side, pieces):
    for side_match in side.side_matches:
        for piece in pieces:
            if piece.piece_Index == side_match.piece_index:
                return piece, side_match
    return None, None


def findBestPuzzle(puzzleSolve, piece, pieces_left):
    puzzleSolve: PuzzleSolve
    piece: PieceInfo
    pieces_left: [PieceInfo]

    returnSolved = []

    if not pieces_left:
        return puzzleSolve

    y_piece_index, x_piece_index = puzzleSolve.find_piece(piece)

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
            puzzleSolve.add_piece(new_y_piece_index, x_piece_index, best_match_piece)
            puzzleSolve.puzzle_score += best_match.histogram_score

            # Recursive call with the next piece

            new_puzzle_solve = findBestPuzzle(
                puzzleSolve, best_match_piece, y_pieces_left
            )
            if new_puzzle_solve:
                returnSolved.append(new_puzzle_solve)

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
            puzzleSolve.add_piece(y_piece_index, new_x_piece_index, best_match_piece)
            puzzleSolve.puzzle_score += best_match.histogram_score

            # Recursive call with the next piece

            new_puzzle_solve = findBestPuzzle(
                puzzleSolve, best_match_piece, pieces_left
            )
            if new_puzzle_solve:
                returnSolved.append(new_puzzle_solve)

    if returnSolved:
        # Return the PuzzleSolve instance with the smallest puzzle_score
        return min(returnSolved, key=lambda ps: ps.puzzle_score)
    else:
        # Return None if no further solutions were found
        return None


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
                    tPiece, tSide.side_Index, pieces_to_compare
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
            print("Broken Corner")
            break
        first_piece.rotate_sides()
        counter += 1
    puzzleSolve.add_piece(0, 0, first_piece)
    puzzle_pieces_left = puzzle.pieces
    bestSolve = findBestPuzzle(puzzleSolve, first_piece, puzzle_pieces_left.copy())

    write_placement_to_csv(bestSolve, "puzzle_placement.csv")
    # # Save the solved puzzle
    # print(f"Saving solved puzzle... {puzzle_name}_solved.png")
    # cv2.imwrite(os.path.join(solvedPath, f"{puzzle_name}_solved.png"), solvedPuzzle)
    # return solvedPuzzle


def write_placement_to_csv(puzzleSolve, filename):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["y_piece_index", "x_piece_index", "piece_index"])

        for (y, x), piece in puzzleSolve.pieces.items():
            writer.writerow([y, x, piece.piece_Index])


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
