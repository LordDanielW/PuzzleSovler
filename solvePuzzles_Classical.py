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
)

from drawUtils import (
    draw_gradient_contours,
    plot_histogram,
    draw_segmented_contours,
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
    current_side = current_piece.sides[side_Index]
    reversed_histogram = np.array(current_side.Histogram)[::-1]
    inv_rev_hist = np.array(reversed_histogram) * -1
    all_side_matches = []

    for piece in pieces_to_compare:
        piece: PieceInfo
        for side in piece.sides:
            side: SideInfo
            if not side.isEdge:
                dis_sqrd, shift = distance_squared_average(inv_rev_hist, side.Histogram)
                side_match = SideMatch()
                side_match.piece_index = piece.piece_Index
                side_match.side_index = side.side_Index
                side_match.histogram_score = dis_sqrd
                side_match.histogram_shift = shift
                all_side_matches.append(side_match)

    sorted_side_matches = sorted(
        all_side_matches, key=lambda match: match.histogram_score
    )  # [:5]

    # # Plot for each of the top 5 matches
    # for match in sorted_side_matches:
    #     matched_piece = next(
    #         (p for p in pieces_to_compare if p.piece_Index == match.piece_index), None
    #     )
    #     matched_side = matched_piece.sides[match.side_index] if matched_piece else None
    #     if matched_side:
    #         plot_all(
    #             inv_rev_hist,
    #             matched_side.Histogram,
    #             current_piece.puzzle_piece,
    #             current_side.Points,
    #             matched_piece.puzzle_piece,
    #             matched_side.Points,
    #         )

    return sorted_side_matches


def plot_all(
    reversed_histogram,
    match_histogram,
    current_puzzle_piece,
    current_points,
    match_puzzle_piece,
    match_points,
):
    draw_segmented_contours(current_puzzle_piece, current_points, "current_piece")
    draw_segmented_contours(match_puzzle_piece, match_points, "match_piece")

    plt.figure(figsize=(10, 6))
    plt.plot(reversed_histogram, label="Reversed Histogram")
    plt.plot(match_histogram, label="Match Histogram")
    plt.title("Histogram Comparison")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


def find_best_match_for_side(side, pieces, max_matches=5):
    matches = []
    for side_match in side.side_matches:
        if len(matches) >= max_matches:
            break  # Stop when we reach the maximum number of matches

        for piece in pieces:
            if piece.piece_Index == side_match.piece_index:
                matches.append((piece, side_match))
                break  # Found a match for this side match, move to the next

    return matches


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
        matches = find_best_match_for_side(piece.sides[2], pieces_left)

        for best_match_piece, best_match in matches:
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
        matches = find_best_match_for_side(piece.sides[1], pieces_left)

        for best_match_piece, best_match in matches:
            x_pieces_left = pieces_left.copy()
            x_pieces_left.remove(best_match_piece)
            best_match_piece.rotate_sides()  # Rotate to align the matched side correctly
            while best_match_piece.sides[3].side_Index != best_match.side_index:
                best_match_piece.rotate_sides()

            # Add the piece to the puzzle matrix and update the score
            new_x_piece_index = x_piece_index + 1
            puzzleSolve.add_piece(y_piece_index, new_x_piece_index, best_match_piece)
            puzzleSolve.puzzle_score += best_match.histogram_score

            # Recursive call with the next piece
            new_puzzle_solve = findBestPuzzle(
                puzzleSolve, best_match_piece, x_pieces_left
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


def findAPuzzle(puzzleSolve, last_piece, pieces_left):
    puzzleSolve: PuzzleSolve
    last_piece: PieceInfo
    pieces_left: [PieceInfo]

    if not pieces_left:
        return puzzleSolve

    # Assuming the last_piece is placed at (0,0) initially
    if not puzzleSolve.pieces:
        puzzleSolve.add_piece(0, 0, last_piece)
        puzzleSolve.update_score(puzzleSolve.puzzle_score + 100)

    last_x, last_y = puzzleSolve.find_piece(last_piece)
    # width, height = puzzleSolve.width, puzzleSolve.height
    width, height = 8, 9

    # Try placing pieces to the right and bottom of the last placed piece
    for w in range(width):
        for h in range(height):
            if w == 0 and h == 0:
                continue
            puzzleSolve.add_piece(h, w, pieces_left[0])
            puzzleSolve.update_score(puzzleSolve.puzzle_score + 100)
            pieces_left.remove(pieces_left[0])

    return puzzleSolve


def space_puzzle(puzzleSolve):
    puzzleSolve: PuzzleSolve

    # Find the maximum extent of the puzzle
    max_y = max(y for (y, x) in puzzleSolve.pieces.keys())
    max_x = max(x for (y, x) in puzzleSolve.pieces.keys())

    for y in range(max_y + 1):
        for x in range(max_x + 1):
            if (y, x) in puzzleSolve.pieces:
                current_piece = puzzleSolve.pieces[(y, x)]
                left_x, top_y = 0, 0

                # Check for a piece to the left
                if (y, x - 1) in puzzleSolve.pieces:
                    left_neighbor = puzzleSolve.pieces[(y, x - 1)]
                    left_x = left_neighbor.right_x

                # Check for a piece above
                if (y - 1, x) in puzzleSolve.pieces:
                    top_neighbor = puzzleSolve.pieces[(y - 1, x)]
                    top_y = top_neighbor.bottom_y

                # Calculate the bounding rectangle for each contour
                _, _, width, height = cv2.boundingRect(current_piece.puzzle_piece)

                # Set right_x and bottom_y for the current piece
                current_piece.left_x = left_x
                current_piece.top_y = top_y
                current_piece.right_x = left_x + width
                current_piece.bottom_y = top_y + height


def generate_solution_CSV(puzzleSolve, filename):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        # Writing the header
        writer.writerow(
            [
                "piece_Index",
                "top_y",
                "left_x",
                "bottom_y",
                "right_x",
                "angle",
                "piece_name",
            ]
        )

        # Iterating through puzzle pieces and writing their data
        for _, piece in puzzleSolve.pieces.items():
            writer.writerow(
                [
                    piece.piece_Index,
                    piece.top_y,
                    piece.left_x,
                    piece.bottom_y,
                    piece.right_x,
                    piece.angle,
                    piece.piece_name,
                ]
            )


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

    corner_pieces: [PieceInfo] = []
    edge_pieces: [PieceInfo] = []
    middle_pieces: [PieceInfo] = []

    for piece in puzzle.pieces:
        if piece.isCorner:
            corner_pieces.append(piece)
            edge_pieces.append(piece)
        elif piece.isEdge:
            edge_pieces.append(piece)
        else:
            middle_pieces.append(piece)

    for c_piece in corner_pieces:
        contours = []
        for side in c_piece.sides:
            if side.isEdge:
                contours.append(side.Points)
        # draw_segmented_contours(c_piece.puzzle_piece, contours, "corner_piece")
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    for e_piece in edge_pieces:
        contours = []
        for side in e_piece.sides:
            if side.isEdge:
                contours.append(side.Points)
        # draw_segmented_contours(e_piece.puzzle_piece, contours, "edge_piece")
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    puzzleSolve = PuzzleSolve()
    last_piece = corner_pieces[0]
    # pieces_left = []
    pieces_left = puzzle.pieces
    pieces_left.remove(last_piece)
    a_solution = findAPuzzle(puzzleSolve, last_piece, pieces_left)
    space_puzzle(a_solution)
    generate_solution_CSV(a_solution, "Puzzles\Shuffled\jigsaw1\puzzle_placement.csv")
    # good_solutions = []
    # best_solution = None
    # for corner_piece in corner_pieces:
    #     puzzleSolve = PuzzleSolve()

    #     # Orient the corner piece correctly
    #     counter = 0
    #     while not (corner_piece.sides[3].isEdge and corner_piece.sides[0].isEdge):
    #         if counter == 4:
    #             print("Broken Corner")
    #             break
    #         corner_piece.rotate_sides()
    #         counter += 1

    #     puzzleSolve.add_piece(0, 0, corner_piece)
    #     puzzle_pieces_left = edge_pieces.copy()
    #     puzzle_pieces_left.remove(corner_piece)
    #     bestSolve = findBestPuzzle(puzzleSolve, corner_piece, puzzle_pieces_left)

    #     if bestSolve:
    #         good_solutions.append(bestSolve)

    # # Return the best solution based on the puzzle score
    # if good_solutions:
    #     best_solution = min(good_solutions, key=lambda ps: ps.puzzle_score)

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
