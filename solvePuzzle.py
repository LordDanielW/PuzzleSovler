import cv2
import csv
import numpy as np
import os
import glob

# Paths
shuffledPath = "Puzzles/Shuffled/"
solvedPath = "Puzzles/Solved/"


def read_puzzle_pieces_info(csv_file):
    puzzlePiecesInfo = []
    with open(csv_file, "r") as file:
        csvreader = csv.reader(file)
        next(csvreader)  # Skip the header
        for row in csvreader:
            puzzlePiecesInfo.append(
                {
                    "index": int(row[0]),
                    "top_y": int(row[1]),
                    "left_x": int(row[2]),
                }
            )
    return puzzlePiecesInfo


def load_puzzle_pieces(puzzle_folder):
    puzzlePieces = []
    i = 0
    while True:
        filepath = os.path.join(puzzle_folder, f"piece_{i}.png")
        if os.path.exists(filepath):
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                puzzlePieces.append(img)
            i += 1
        else:
            break
    return puzzlePieces


def apply_opposite_rotation(image, angle):
    if angle == cv2.ROTATE_90_CLOCKWISE:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == cv2.ROTATE_180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    else:
        return image  # No rotation needed


def solve_puzzle(puzzle_name):
    puzzle_pieces_info_file = os.path.join(
        shuffledPath, puzzle_name, "puzzle_pieces_info.csv"
    )
    pieceInfo = read_puzzle_pieces_info(puzzle_pieces_info_file)

    pieces = load_puzzle_pieces(os.path.join(shuffledPath, puzzle_name))
    sortedPuzzlePieces = [None] * len(pieces)

    # Create an empty image large enough to place all pieces
    # Assuming the dimensions based on the world coordinates
    solve_width = max([p.shape[1] for p in pieces]) + max(
        pI["left_x"] for pI in pieceInfo
    )
    solve_height = max([p.shape[0] for p in pieces]) + max(
        pI["top_y"] for pI in pieceInfo
    )

    solvedPuzzle = np.zeros(
        (solve_height, solve_width), dtype=np.uint8
    )  # Adjust the size as needed

    for i, piece in enumerate(pieces):
        y, x = pieceInfo[i]["top_y"], pieceInfo[i]["left_x"]
        h, w = piece.shape

        # Create a mask where white pixels are 255 (or true) and others are 0 (or false)
        mask = piece == 255

        # Use the mask to only copy the white pixels onto the solvedPuzzle
        solvedPuzzle[y : y + h, x : x + w][mask] = piece[mask]

        # cv2.imshow("Puzzle", solvedPuzzle)
        # cv2.waitKey(0)
    # Save the solved puzzle
    print(f"Saving solved puzzle... {puzzle_name}_solved.png")
    cv2.imwrite(os.path.join(solvedPath, f"{puzzle_name}_solved.png"), solvedPuzzle)
    return solvedPuzzle

    # for info, piece in zip(puzzlePiecesInfo, puzzlePieces):
    #     i, world_x, world_y, angle = info
    #     rotated_piece = apply_opposite_rotation(piece, angle)

    #     # Place the piece onto the Puzzle image at its world coordinates
    #     h, w = rotated_piece.shape
    #     Puzzle_h, Puzzle_w = Puzzle.shape

    #     # Ensure the piece fits within the Puzzle boundaries
    #     end_x = min(world_x + w, Puzzle_w)
    #     end_y = min(world_y + h, Puzzle_h)

    #     # Adjust the size of the rotated piece if it goes beyond the Puzzle boundaries
    #     adjusted_piece = rotated_piece[: end_y - world_y, : end_x - world_x]

    #     # Place the adjusted piece onto the Puzzle image
    #     Puzzle[world_y:end_y, world_x:end_x] = adjusted_piece

    #     # Store the piece in sortedPuzzlePieces according to its original index
    #     sortedPuzzlePieces[i] = adjusted_piece

    # # Save the solved puzzle
    # print(f"Saving solved puzzle... {puzzle_name}_solved.png")
    # cv2.imwrite(os.path.join(solvedPath, f"{puzzle_name}_solved.png"), Puzzle)

    # return sortedPuzzlePieces


# Example usage
puzzle_name = "jigsaw"  # replace with actual puzzle name
sorted_pieces = solve_puzzle(puzzle_name)
