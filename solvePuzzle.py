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


def apply_opposite_rotation(image, angle):
    if angle == 90:
        return rotate_image_easy(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        return rotate_image_easy(image, cv2.ROTATE_180)
    elif angle == 270:
        return rotate_image_easy(image, cv2.ROTATE_90_CLOCKWISE)
    else:
        return image  # No rotation needed


def solve_puzzle(puzzle_name, info_filename="puzzle_pieces_info.csv", save_name=""):
    puzzle_pieces_info_file = os.path.join(shuffledPath, puzzle_name, info_filename)

    pieceInfo = read_puzzle_pieces_info(puzzle_pieces_info_file)
    pieces, _ = load_puzzle_pieces(os.path.join(shuffledPath, puzzle_name))

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
    print(f"Saving solved puzzle... {puzzle_name}_solved{save_name}.png")
    cv2.imwrite(
        os.path.join(solvedPath, f"{puzzle_name}_solved{save_name}.png"), solvedPuzzle
    )
    return solvedPuzzle


def main():
    # solve_puzzle("jigsaw1", "puzzle_placement.csv", "placement")
    loop = True
    # loop = False

    if loop:
        # Loop through each puzzle directory in shuffledPath
        for puzzle_folder in os.listdir(shuffledPath):
            puzzle_folder_path = os.path.join(shuffledPath, puzzle_folder)
            # Check if it's a directory
            if os.path.isdir(puzzle_folder_path):
                print(f"Solving puzzle: {puzzle_folder}")
                sorted_pieces = solve_puzzle(puzzle_folder)
    else:
        # Example usage
        puzzle_name = "jigsaw1"  # replace with actual puzzle name

        for rot_num in ["0", "90", "180", "270"]:
            info_filename = f"Rotation_{rot_num}.csv"
            save_name = f"_Rot_{rot_num}"
            sorted_pieces = solve_puzzle(puzzle_name, info_filename, save_name)


if __name__ == "__main__":
    main()
