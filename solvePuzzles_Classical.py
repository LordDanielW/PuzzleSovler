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


def findBestDoubleMatches(piece3, piece0, pieces_to_compare):
    piece3: PieceInfo
    piece0: PieceInfo

    reversed_histogram3 = np.array(piece3.sides[1].Histogram)[::-1]
    inv_rev_hist3 = np.array(reversed_histogram3) * -1

    reversed_histogram0 = np.array(piece0.sides[2].Histogram)[::-1]
    inv_rev_hist0 = np.array(reversed_histogram0) * -1

    pieces_to_compare: [PieceInfo]
    all_side_matches = []

    for piece in pieces_to_compare:
        piece: PieceInfo
        for side_index in range(4):
            side3 = piece.sides[side_index]
            side0 = piece.sides[(side_index + 1) % 3]
            side3: SideInfo
            side0: SideInfo

            dis_sqrd3, shift = distance_squared_average(
                np.array(inv_rev_hist3),
                np.array(side3.Histogram),
            )

            dis_sqrd0, shift = distance_squared_average(
                np.array(inv_rev_hist0),
                np.array(side0.Histogram),
            )

            side_match = SideMatch()
            side_match.piece_index = piece.piece_Index
            side_match.side_index = side_index
            side_match.histogram_score = dis_sqrd3 + dis_sqrd0
            side_match.histogram_shift = shift
            all_side_matches.append(side_match)

    sorted_side_matches = sorted(
        all_side_matches, key=lambda match: match.histogram_score
    )

    return sorted_side_matches


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
                side_match.side_index = side_index
                side_match.histogram_score = dis_sqrd
                side_match.histogram_shift = shift
                all_side_matches.append(side_match)

    sorted_side_matches = sorted(
        all_side_matches, key=lambda match: match.histogram_score
    )

    if debugVis:
        # Plot for each of the top 3 matches
        for i, match in enumerate(sorted_side_matches[:3]):
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


def find_best_match_for_side2(side, current_piece, pieces, max_matches=3):
    matches = []
    for side_match in side.side_matches:
        if len(matches) >= max_matches:
            break  # Stop when we reach the maximum number of matches

        for piece in pieces:
            if piece.piece_Index == side_match.piece_index:
                matches.append((piece, side_match))
                break  # Found a match for this side match, move to the next

    for piece, side_match in matches:
        piece: PieceInfo
        side_match: SideMatch

        combine_images_left_right(
            current_piece.puzzle_piece,
            current_piece.corners,
            piece.puzzle_piece,
            piece.corners,
        )

    return matches


def find_corners(piece_img):
    # Find the coordinates of all white pixels
    y_coords, x_coords = np.where(piece_img == 255)
    white_pixels = np.column_stack((x_coords, y_coords))

    # If there are no white pixels, return None or handle as needed
    if white_pixels.size == 0:
        return None

    # Calculate the distance of each white pixel to the four corners of the image
    top_left_dist = np.sum((white_pixels - [0, 0]) ** 2, axis=1)
    top_right_dist = np.sum((white_pixels - [piece_img.shape[1], 0]) ** 2, axis=1)
    bottom_right_dist = np.sum(
        (white_pixels - [piece_img.shape[1], piece_img.shape[0]]) ** 2, axis=1
    )
    bottom_left_dist = np.sum((white_pixels - [0, piece_img.shape[0]]) ** 2, axis=1)

    # Identify the white pixel closest to each corner
    top_left = white_pixels[np.argmin(top_left_dist)]
    top_right = white_pixels[np.argmin(top_right_dist)]
    bottom_right = white_pixels[np.argmax(bottom_right_dist)]
    bottom_left = white_pixels[np.argmin(bottom_left_dist)]

    # Create a contour array
    corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)

    return corners.reshape((-1, 1, 2))


def combine_images_left_right(img1, corners1, img2, corners2):
    # Find the top-rightmost white pixel in img1 and the top-leftmost white pixel in img2
    # corners1 = find_corners(img1)
    # corners2 = find_corners(img2)

    debugImgs = []

    debugImgs.append(
        draw_gradient_contours(img1, [corners1[1]], "Corners", False, False)
    )
    debugImgs.append(
        draw_gradient_contours(img2, [corners2[0]], "Corners", False, False)
    )

    show_all(debugImgs, "Image 1 and 2", 3, 1, True, True)

    print(f"Corners 1: {corners1[1]}")
    print(f"Corners 2: {corners2[0]}")

    # Calculate the offset for placing img2 next to img1
    offset_x = corners1[1][0][0] + 1 - corners2[0][0][0]
    offset_y = corners1[1][0][1] - corners2[0][0][1]
    y1_diff = 0
    y2_diff = 0
    if offset_y < 0:
        y1_diff = abs(offset_y)
    else:
        y2_diff = abs(offset_y)

    print(f"Offset X: {offset_x}")
    print(f"Y1 Diff: {y1_diff}")
    print(f"Y2 Diff: {y2_diff}")

    # Create a new image large enough to hold both images
    combined_height = max(img1.shape[0] + y1_diff, img2.shape[0] + y2_diff)
    combined_width = offset_x + img2.shape[1]
    combined_img = np.zeros((combined_height, combined_width), dtype=img1.dtype)

    # Place img1 in the combined image
    combined_img[y1_diff : y1_diff + img1.shape[0], : img1.shape[1]] = img1

    # Place img2 in the combined image at the calculated offset
    mask = img2 == 255

    combined_img[y2_diff : y2_diff + img2.shape[0], offset_x:combined_width][
        mask
    ] = img2[mask]

    # # Calculate the region where img2 is placed in the combined image
    # img2_region = combined_img[
    #     y2_diff : y2_diff + img2.shape[0], offset_x:combined_width
    # ]

    # # Count overlapping white pixels
    # # The overlapping region is where img1 and img2 potentially intersect
    # # We consider a pixel as overlapping if it is white (255) in both images
    # overlap = np.sum(
    #     (img1[y1_diff : y1_diff + img2.shape[0], : img2.shape[1]] == 255)
    #     & (img2_region == 255)
    # )

    # print(f"Number of overlapping white pixels: {overlap}")

    # Distance squared between bottom-right corner of img1 and bottom-left corner of img2
    distance_squared = (corners1[2][0][0] - corners2[3][0][0] - offset_x) ** 2 + (
        (corners1[2][0][1] + y1_diff) - (corners2[3][0][1] + y2_diff)
    ) ** 2

    img_list = [img1, img2, combined_img]
    show_all(img_list, "Combined Images", 3, 1, True, True)

    print(f"Overlap: {overlap}")
    print(f"Distance Squared: {distance_squared}")

    return overlap, distance_squared


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
    show_all(edge_images, "Edge Pieces", 5, 0.5, False, True)


def solve_middle_pieces(puzzleSolve, middle_pieces, debugVis=False):
    # Loop through the middle area, excluding the edges
    for y in range(1, puzzleSolve.metadata.yn - 1):
        for x in range(1, puzzleSolve.metadata.xn - 1):
            current_piece_above = puzzleSolve.pieces[(y - 1, x)]
            current_piece_left = puzzleSolve.pieces[(y, x - 1)]
            current_piece_above: PieceInfo
            current_piece_left: PieceInfo

            # Match Top and Left sides
            current_piece_left.sides[1].side_matches = findBestDoubleMatches(
                current_piece_left, current_piece_above, middle_pieces
            )

            if not current_piece_left.sides[1]:
                raise Exception("No match found for middle piece.")

            # Find best match for side[1] of current piece
            best_match_piece, best_match = find_best_match_for_side(
                current_piece_left.sides[1], middle_pieces, 1
            )[0]

            # Rotate the piece to match the top and left sides
            for _ in range(3 - best_match.side_index):
                best_match_piece.rotate_piece_deep()

            # Add the piece to the puzzle matrix
            puzzleSolve.add_piece(y, x, best_match_piece, debugVis)

            # Remove the piece from the middle pieces list
            middle_pieces.remove(best_match_piece)

    return puzzleSolve


def solve_edge_pieces(
    puzzleSolve, current_piece, edge_pieces, edge_index, debugVis=False
):
    # The indexes for sides that are edges in corner pieces, in clockwise order starting from top-left
    edge_compares = [(1, 3), (2, 0), (1, 3), (2, 0)]
    eComp = edge_compares[edge_index]
    y_add = 0 if edge_index == 0 or edge_index == 2 else 1
    x_add = 0 if edge_index == 1 or edge_index == 3 else 1

    t_range = puzzleSolve.metadata.xn - 2 if x_add == 1 else puzzleSolve.metadata.yn - 2

    # Grab all edges align them to edge = edge_index
    t_edges = []
    for edge_piece in edge_pieces:
        edge_piece: PieceInfo
        while not (edge_piece.sides[edge_index].isEdge):
            edge_piece.rotate_piece_deep()
        t_edges.append(edge_piece)

    if debugVis and edge_index == 0:
        edge_images = []
        for e_piece in t_edges:
            contours = []
            contours.append(e_piece.sides[eComp[1]].Points)
            edge_images.append(
                draw_segmented_contours(
                    e_piece.puzzle_piece, contours, "edge_piece", False, False
                )
            )
        show_all(edge_images, "Edge Pieces", 5, 0.5, True, True)

    # Solve Edge
    for i in range(t_range):
        show_debug = i == 0 and debugVis and edge_index == 0
        # Find best matches for the current piece
        current_piece.sides[eComp[0]].side_matches, _ = findBestMatches(
            current_piece,
            eComp[0],
            t_edges,
            [eComp[1]],
            show_debug,
            0,
        )

        # If there's no match found, break the loop
        if not current_piece.sides[eComp[0]].side_matches:
            raise ValueError(f"No matches for piece {current_piece.piece_name}.")

        # Find best match for side[1] of current piece

        # best_match_piece, _ = find_best_match_for_side2(
        #     current_piece.sides[eComp[0]], current_piece, t_edges, 3
        # )[0]

        best_match_piece, _ = find_best_match_for_side(
            current_piece.sides[eComp[0]], t_edges, 3
        )[0]

        # Add the best matching piece to the puzzle matrix on the right side
        last_y, last_x = puzzleSolve.find_piece(current_piece)
        puzzleSolve.add_piece(
            last_y + y_add, last_x + x_add, best_match_piece, debugVis
        )

        # Remove the best matching piece from the list of pieces to compare
        t_edges.remove(best_match_piece)

        # Update the current piece to the best matching piece for the next iteration
        current_piece = best_match_piece

    return puzzleSolve, current_piece, t_edges


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

    current_piece = best_match_piece

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
        t_Debug = debugVis and i == 0
        piece: PieceInfo = segmentSides(raw_piece, t_Debug, 4, 3, 18)
        piece.piece_Index = i
        piece.piece_name = piecesInfo[i]["piece_name"]
        # piece.puzzle_piece = rotate_image_easy(piece.puzzle_piece, 1)
        puzzle.pieces.append(piece)
        if i > 0:
            pieces_to_compare.append(piece)

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

    # Verify Edges / Corners
    if debugVis:
        verify_edges_corners(corner_pieces, edge_pieces)
        middle_imgs = []
        for m_piece in middle_pieces:
            middle_imgs.append(m_piece.puzzle_piece)
        show_all(middle_imgs, "Middle Pieces", 5, 0.5, True, True)

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

    # Copy the edge pieces to a new list
    # aligned_edges = [edge_piece for edge_piece in edge_pieces]

    # Solve Top Edge
    puzzleSolve, current_piece, edge_pieces = solve_edge_pieces(
        puzzleSolve, current_piece, edge_pieces, 0, debugVis
    )

    # Solve Top Right Corner
    puzzleSolve, current_piece, corner_pieces = solve_corner_piece(
        puzzleSolve, current_piece, corner_pieces, 1, debugVis
    )
    top_right_corner = current_piece

    # Solve Left Edge
    current_piece = first_piece
    puzzleSolve, current_piece, edge_pieces = solve_edge_pieces(
        puzzleSolve, current_piece, edge_pieces, 3, debugVis
    )

    # Solve Middle Pieces
    puzzleSolve = solve_middle_pieces(puzzleSolve, middle_pieces, debugVis)

    # Solve Bottom Left Corner
    puzzleSolve, current_piece, corner_pieces = solve_corner_piece(
        puzzleSolve, current_piece, corner_pieces, 3, debugVis
    )

    # Solve Bottom Edge
    puzzleSolve, current_piece, edge_pieces = solve_edge_pieces(
        puzzleSolve, current_piece, edge_pieces, 2, debugVis
    )

    # Solve the Right Edge
    current_piece = top_right_corner
    puzzleSolve, current_piece, edge_pieces = solve_edge_pieces(
        puzzleSolve, current_piece, edge_pieces, 1, debugVis
    )

    # Solve Bottom Right Corner
    puzzleSolve, current_piece, corner_pieces = solve_corner_piece(
        puzzleSolve, current_piece, corner_pieces, 2, True
    )

    # Print Puzzle Score
    print(f"Puzzle Error: {puzzleSolve.puzzle_score}")
    generate_solution_CSV(
        puzzleSolve, f"Puzzles/Solved/{puzzle_name}_classic_solved.csv"
    )


# Example usage
# puzzle_name = "jigsaw3"  # replace with actual puzzle name
# sorted_pieces = solve_puzzle(puzzle_name, True)

# Loop through each puzzle directory in shuffledPath
for i, puzzle_folder in enumerate(os.listdir(shuffledPath)):
    puzzle_folder_path = os.path.join(shuffledPath, puzzle_folder)
    # Check if it's a directory
    if os.path.isdir(puzzle_folder_path):
        print(f"Solving puzzle: {puzzle_folder}")
        sorted_pieces = solve_puzzle(puzzle_folder, i == 0)
