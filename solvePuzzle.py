import cv2
import csv
import numpy as np
import os
import glob

from utils import find_bounding_box, rotate_image_easy, rotate_image

# Paths
shuffledPath = "Puzzles/Shuffled/"
solvedPath = "Puzzles/Solved/"


def read_puzzle_pieces_info(csv_file):
    puzzlePiecesInfo = []
    with open(csv_file, "r") as file:
        csvreader = csv.DictReader(file)
        for row in csvreader:
            # Convert all values to integers if needed
            info = {
                key: int(value) if value.isdigit() else value
                for key, value in row.items()
            }
            puzzlePiecesInfo.append(info)
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
        return rotate_image_easy(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == cv2.ROTATE_180:
        return rotate_image_easy(image, cv2.ROTATE_180)
    elif angle == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return rotate_image_easy(image, cv2.ROTATE_90_CLOCKWISE)
    else:
        return image  # No rotation needed


def solve_puzzle(puzzle_name):
    puzzle_pieces_info_file = os.path.join(
        shuffledPath, puzzle_name, "puzzle_pieces_info.csv"
    )
    pieceInfo = read_puzzle_pieces_info(puzzle_pieces_info_file)
    pieces = load_puzzle_pieces(os.path.join(shuffledPath, puzzle_name))

    # Unshuffle  #  Turns out this is uneeded, as piece and piece info are still matched
    # pieces = [None] * len(shuffledPieces)
    # pieceInfo = [None] * len(shuffledPieceInfo)
    # for i, info in enumerate(shuffledPieceInfo):
    #     originalIndex = info["index"]  # The original index of the puzzle piece
    #     pieces[originalIndex] = shuffledPieces[i]
    #     pieceInfo[originalIndex] = info

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

    #
    for i, piece in enumerate(pieces):
        y, x, angle = (
            pieceInfo[i]["top_y"],
            pieceInfo[i]["left_x"],
            pieceInfo[i]["angle"],
        )

        rotated_piece = apply_opposite_rotation(piece, angle)

        h, w = rotated_piece.shape

        # Create a mask where white pixels are 255 (or true) and others are 0 (or false)
        mask = rotated_piece == 255
        # Use the mask to only copy the white pixels onto the solvedPuzzle
        solvedPuzzle[y : y + h, x : x + w][mask] = rotated_piece[mask]

    # Save the solved puzzle
    print(f"Saving solved puzzle... {puzzle_name}_solved.png")
    cv2.imwrite(os.path.join(solvedPath, f"{puzzle_name}_solved.png"), solvedPuzzle)
    return solvedPuzzle


# Example usage
puzzle_name = "jigsaw"  # replace with actual puzzle name
sorted_pieces = solve_puzzle(puzzle_name)
