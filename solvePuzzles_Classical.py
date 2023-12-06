import cv2
import csv
import numpy as np
import os
import matplotlib.pyplot as plt
import glob

from utilsMath import distance_squared_average, rotate_image_easy

from utilsLoad import load_puzzle_pieces, load_metadata
from utilsDraw import (
    draw_gradient_contours,
    plot_histogram,
    draw_segmented_contours,
    scale_piece,
    show_all,
)
from utilsMath import distance_squared_average
from classPuzzle import PieceInfo, PuzzleInfo, SideInfo, SideMatch, PuzzleSolve
from definePiece import segmentSides

# Paths
shuffledPath = "Puzzles/Shuffled/"
solvedPath = "Puzzles/Solved/"


def findBestMatches(
    current_piece,
    side_Index,
    pieces_to_compare,
    index_of_sides_to_compare,
    debugVis=False,
    count_matches=0,
):
    current_piece: PieceInfo
    pieces_to_compare: [PieceInfo]
    current_side = current_piece.sides[side_Index]
    reversed_histogram = np.array(current_side.Histogram)[::-1]
    inv_rev_hist = np.array(reversed_histogram) * -1
    all_side_matches = []

    for piece in pieces_to_compare:
        piece: PieceInfo
        for side_index in index_of_sides_to_compare:
            side = piece.sides[side_index]
            side: SideInfo
            if not side.isEdge:
                count_matches += 1
                dis_sqrd, shift = distance_squared_average(inv_rev_hist, side.Histogram)
                side_match = SideMatch()
                side_match.piece_index = piece.piece_Index
                side_match.side_index = side.side_Index
                side_match.histogram_score = dis_sqrd
                side_match.histogram_shift = shift
                all_side_matches.append(side_match)

    sorted_side_matches = sorted(
        all_side_matches, key=lambda match: match.histogram_score
    )

    if debugVis:
        # Plot for each of the top 5 matches
        for i, match in enumerate(sorted_side_matches[:5]):
            matched_piece = next(
                (p for p in pieces_to_compare if p.piece_Index == match.piece_index),
                None,
            )
            matched_side = (
                matched_piece.sides[match.side_index] if matched_piece else None
            )
            if matched_side:
                plot_all(
                    inv_rev_hist,
                    matched_side.Histogram,
                    current_piece.puzzle_piece,
                    current_side.Points,
                    matched_piece.puzzle_piece,
                    matched_side.Points,
                )

    return sorted_side_matches, count_matches


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


def find_best_match_for_side(side, pieces, max_matches=3):
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


def verify_edges_corners(corner_pieces, edge_pieces):
    corner_images = []
    for c_piece in corner_pieces:
        contours = []
        for side in c_piece.sides:
            if side.isEdge:
                contours.append(side.Points)
        corner_images.append(
            draw_segmented_contours(
                c_piece.puzzle_piece, contours, "corner_piece", False, False
            )
        )
    show_all(corner_images, "Corner Pieces", 5, 1, False, True)

    edge_images = []
    for e_piece in edge_pieces:
        contours = []
        for side in e_piece.sides:
            if side.isEdge:
                contours.append(side.Points)
        edge_images.append(
            draw_segmented_contours(
                e_piece.puzzle_piece, contours, "edge_piece", False, False
            )
        )
    show_all(edge_images, "Edge Pieces", 5, 0.5, True, True)


def solve_middle_pieces(puzzleSolve, middle_pieces, debugVis=False):
    # Loop through the middle area, excluding the edges
    for y in range(1, puzzleSolve.metadata.yn - 1):
        for x in range(1, puzzleSolve.metadata.xn - 1):
            current_piece_above = puzzleSolve.pieces[(y - 1, x)]
            current_piece_left = puzzleSolve.pieces[(y, x - 1)]

            # Find the best match for the bottom side of the piece above and the right side of the piece to the left
            above_side_matches, _ = findBestMatches(
                current_piece_above, 2, middle_pieces, [0], False, 0
            )
            left_side_matches, _ = findBestMatches(
                current_piece_left, 1, middle_pieces, [3], False, 0
            )

            # Find the best matching piece that matches both above and left conditions
            best_match_piece = None
            for above_match in above_side_matches:
                for left_match in left_side_matches:
                    if above_match.piece_index == left_match.piece_index:
                        best_match_piece = next(
                            (
                                p
                                for p in middle_pieces
                                if p.piece_Index == above_match.piece_index
                            ),
                            None,
                        )
                        break
                if best_match_piece:
                    break

            # If a match is found, rotate and add the piece
            if best_match_piece:
                # Rotate the piece to align correctly with the pieces above and to the left
                while best_match_piece.sides[0].side_Index != above_match.side_index:
                    best_match_piece.rotate_piece_deep()
                while best_match_piece.sides[3].side_Index != left_match.side_index:
                    best_match_piece.rotate_piece_deep()

                # Add the piece to the puzzle matrix
                puzzleSolve.add_piece(y, x, best_match_piece, debugVis)

                # Remove the piece from the middle pieces list
                middle_pieces.remove(best_match_piece)

    return puzzleSolve


def solve_edge_pieces(puzzleSolve, current_piece, edge_pieces, debugVis=False):
    # Grab all edges align them to edge = 0
    t_edges = []
    for edge_piece in edge_pieces:
        edge_piece: PieceInfo
        while not (edge_piece.sides[0].isEdge):
            edge_piece.rotate_piece_deep()
        aligned_edges.append(edge_piece)

    edge_images = []
    for e_piece in aligned_edges:
        contours = []
        for side in e_piece.sides:
            if side.isEdge:
                contours.append(side.Points)
        edge_images.append(
            draw_segmented_contours(
                e_piece.puzzle_piece, contours, "edge_piece", False, False
            )
        )
    if debugVis:
        show_all(edge_images, "Edge Pieces", 5, 0.5, True, True)

    # Solve Top Edge
    current_piece = current_piece
    for _ in range(puzzleSolve.metadata.xn - 2):
        # Find best matches for the current piece's right side
        current_piece.sides[1].side_matches, _ = findBestMatches(
            current_piece,
            1,
            aligned_edges,
            [3],
            False,
            0,
        )

        # If there's no match found, break the loop
        if not current_piece.sides[1].side_matches:
            break

        # Find best match for side[1] of current piece
        best_match_piece, _ = find_best_match_for_side(
            current_piece.sides[1], aligned_edges, 1
        )[0]

        # Add the best matching piece to the puzzle matrix on the right side
        puzzleSolve.add_piece(
            0, puzzleSolve.find_piece(current_piece)[1] + 1, best_match_piece, debugVis
        )

        # Remove the best matching piece from the list of pieces to compare
        aligned_edges.remove(best_match_piece)

        # Update the current piece to the best matching piece for the next iteration
        current_piece = best_match_piece

    return puzzleSolve, current_piece, edge_pieces


def solve_corner_piece(
    puzzleSolve, current_piece, corner_pieces, corner_index, debugVis=False
):
    # The indexes for sides that are edges in corner pieces, in clockwise order starting from top-left
    corner_sides = [(0, 3), (0, 1), (1, 2), (2, 3)]
    corner_compares = [(3, 1), (1, 3), (2, 0), (2, 0)]
    corner_coridinates = [
        (0, 0),
        (0, puzzleSolve.metadata.xn - 1),
        (puzzleSolve.metadata.yn - 1, puzzleSolve.metadata.xn - 1),
        (puzzleSolve.metadata.yn - 1, 0),
    ]

    tSide = corner_sides[corner_index]
    tComp = corner_compares[corner_index]
    tCoord = corner_coridinates[corner_index]

    # Orient the corner pieces
    for corner_piece in corner_pieces:
        # Orient the corner so that the edges are on sides[0] and sides[1]
        counter = 0
        while not (
            corner_piece.sides[tSide[0]].isEdge and corner_piece.sides[tSide[1]].isEdge
        ):
            if counter >= 4:
                print("Broken Corner")
                break
            corner_piece.rotate_piece_deep()
            counter += 1

    # Find best matches for the current piece's right side
    current_piece.sides[tComp[0]].side_matches, _ = findBestMatches(
        current_piece,
        tComp[0],
        corner_pieces,
        [tComp[1]],
        False,
        0,
    )

    # If there's no match found, break the loop
    if not current_piece.sides[tComp[0]].side_matches:
        raise Exception("No match found for top right corner piece.")

    # Find best match for side[1] of current piece
    best_match_piece, best_match = find_best_match_for_side(
        current_piece.sides[tComp[0]], corner_pieces, 1
    )[0]

    # Add the best matching piece to the puzzle matrix on the right side
    puzzleSolve.add_piece(tCoord[0], tCoord[1], best_match_piece, debugVis)

    # Remove the best matching piece from the list of pieces to compare
    corner_pieces.remove(best_match_piece)

    return puzzleSolve, current_piece, corner_pieces


def solve_puzzle(puzzle_name, debugVis=True):
    raw_pieces, piecesInfo = load_puzzle_pieces(os.path.join(shuffledPath, puzzle_name))
    meta_data = load_metadata(
        os.path.join(shuffledPath, puzzle_name, "puzzle_meta_data.json")
    )

    puzzle = PuzzleInfo()
    pieces_to_compare: [PieceInfo] = []

    # Define All Pieces
    for i, raw_piece in enumerate(raw_pieces):
        # For Debug i == 0
        piece: PieceInfo = segmentSides(raw_piece, False, 4, 3)
        piece.piece_Index = i
        piece.piece_name = piecesInfo[i]["piece_name"]
        # piece.puzzle_piece = rotate_image_easy(piece.puzzle_piece, 1)
        puzzle.pieces.append(piece)
        if i > 0:
            pieces_to_compare.append(piece)

    count_matches = 0
    # Find matches for Sides
    for i, tPiece in enumerate(puzzle.pieces[:-1]):
        tPiece: PieceInfo
        for tSide in tPiece.sides:
            tSide: SideInfo
            if tSide.isEdge == False:
                # For Debug i == 0
                tSide.side_matches, count_matches = findBestMatches(
                    tPiece,
                    tSide.side_Index,
                    pieces_to_compare,
                    [0, 1, 2, 3],
                    False,
                    count_matches,
                )

        pieces_to_compare.pop(0)

    # Verify Matches
    print("Matches Found: ", count_matches)

    corner_pieces: [PieceInfo] = []
    edge_pieces: [PieceInfo] = []
    middle_pieces: [PieceInfo] = []

    # Find Edges / Corners / Middle Pieces
    for piece in puzzle.pieces:
        if piece.isCorner:
            corner_pieces.append(piece)
        elif piece.isEdge:
            edge_pieces.append(piece)
        else:
            middle_pieces.append(piece)

    # # Verify Edges / Corners
    # if debugVis:
    #     verify_edges_corners(corner_pieces, edge_pieces)

    # Lets Solve a Puzzle
    puzzleSolve = PuzzleSolve(meta_data)
    first_piece: PieceInfo = corner_pieces[0]
    corner_pieces.remove(first_piece)

    # Orient the corner piece to top left
    counter = 0
    while not (first_piece.sides[3].isEdge and first_piece.sides[0].isEdge):
        if counter == 4:
            print("Broken Corner")
            break
        first_piece.rotate_piece_deep()
        counter += 1

    puzzleSolve.add_piece(0, 0, first_piece, True)

    current_piece = first_piece

    # debugVis = True

    # Grab all edges align them to edge = 0
    aligned_edges = []
    for edge_piece in edge_pieces:
        edge_piece: PieceInfo
        while not (edge_piece.sides[0].isEdge):
            edge_piece.rotate_piece_deep()
        aligned_edges.append(edge_piece)

    edge_images = []
    for e_piece in aligned_edges:
        contours = []
        for side in e_piece.sides:
            if side.isEdge:
                contours.append(side.Points)
        edge_images.append(
            draw_segmented_contours(
                e_piece.puzzle_piece, contours, "edge_piece", False, False
            )
        )
    if debugVis:
        show_all(edge_images, "Edge Pieces", 5, 0.5, True, True)

    # Solve Top Edge

    puzzleSolve, current_piece, aligned_edges = solve_edge_pieces(
        puzzleSolve, current_piece, aligned_edges, debugVis
    )

    current_piece = first_piece
    for _ in range(puzzleSolve.metadata.xn - 2):
        # Find best matches for the current piece's right side
        current_piece.sides[1].side_matches, _ = findBestMatches(
            current_piece,
            1,
            aligned_edges,
            [3],
            False,
            0,
        )

        # If there's no match found, break the loop
        if not current_piece.sides[1].side_matches:
            break

        # Find best match for side[1] of current piece
        best_match_piece, _ = find_best_match_for_side(
            current_piece.sides[1], aligned_edges, 1
        )[0]

        # Add the best matching piece to the puzzle matrix on the right side
        puzzleSolve.add_piece(
            0, puzzleSolve.find_piece(current_piece)[1] + 1, best_match_piece, debugVis
        )

        # Remove the best matching piece from the list of pieces to compare
        aligned_edges.remove(best_match_piece)

        # Update the current piece to the best matching piece for the next iteration
        current_piece = best_match_piece

    # Solve Top Right Corner
    puzzleSolve, current_piece, corner_pieces = solve_corner_piece(
        puzzleSolve, current_piece, corner_pieces, 1, debugVis
    )

    # Solve Left Edge
    debugVis = True
    # Rotate all aligned edges so that the edge is on sides[1]
    for edge_piece in aligned_edges:
        while not (edge_piece.sides[3].isEdge):
            edge_piece.rotate_piece_deep()

    current_piece = first_piece
    for _ in range(puzzleSolve.metadata.yn - 2):
        # Find best matches for the current piece's right side
        current_piece.sides[2].side_matches, _ = findBestMatches(
            current_piece,
            1,
            aligned_edges,
            [0],
            False,
            0,
        )

        # If there's no match found, break the loop
        if not current_piece.sides[2].side_matches:
            break

        # Find best match for side[1] of current piece
        best_match_piece, _ = find_best_match_for_side(
            current_piece.sides[2], aligned_edges, 1
        )[0]

        # Add the best matching piece to the puzzle matrix on the right side
        puzzleSolve.add_piece(
            puzzleSolve.find_piece(current_piece)[0] + 1,
            0,
            best_match_piece,
            debugVis,
        )

        # Remove the best matching piece from the list of pieces to compare
        aligned_edges.remove(best_match_piece)

        # Update the current piece to the best matching piece for the next iteration
        current_piece = best_match_piece

    # Solve Middle Pieces
    puzzleSolve = solve_middle_pieces(puzzleSolve, middle_pieces, debugVis)

    # Solve Bottom Right Corner
    puzzleSolve, current_piece, corner_pieces = solve_corner_piece(
        puzzleSolve, current_piece, corner_pieces, 3, debugVis
    )

    # puzzleSolve = PuzzleSolve()
    # last_piece = corner_pieces[0]
    # # pieces_left = []
    # pieces_left = puzzle.pieces
    # pieces_left.remove(last_piece)
    # a_solution = findAPuzzle(puzzleSolve, last_piece, pieces_left)
    # space_puzzle(a_solution)
    # generate_solution_CSV(a_solution, "Puzzles\Shuffled\jigsaw1\puzzle_placement.csv")

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
    #     puzzle_pieces_left = puzzle.pieces
    #     puzzle_pieces_left.remove(corner_piece)
    #     bestSolve = findBestPuzzle(puzzleSolve, corner_piece, puzzle_pieces_left)

    #     if bestSolve:
    #         good_solutions.append(bestSolve)

    # if good_solutions:
    #     best_solution = min(good_solutions, key=lambda ps: ps.puzzle_score)

    # space_puzzle(best_solution)
    # generate_solution_CSV(best_solution, "Puzzles\Shuffled\jigsaw1\solution.csv")


# Example usage
puzzle_name = "jigsaw1"  # replace with actual puzzle name
sorted_pieces = solve_puzzle(puzzle_name, False)

# # Loop through each puzzle directory in shuffledPath
# for puzzle_folder in os.listdir(shuffledPath):
#     puzzle_folder_path = os.path.join(shuffledPath, puzzle_folder)
#     # Check if it's a directory
#     if os.path.isdir(puzzle_folder_path):
#         print(f"Solving puzzle: {puzzle_folder}")
#         sorted_pieces = solve_puzzle(puzzle_folder)
