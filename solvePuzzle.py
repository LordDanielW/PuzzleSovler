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
                [int(row[0]), int(row[1]), int(row[2]), int(row[3])]
            )
    return puzzlePiecesInfo


def load_puzzle_pieces(puzzle_folder):
    puzzlePieces = []
    for filepath in sorted(glob.glob(os.path.join(puzzle_folder, "piece_*.png"))):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            puzzlePieces.append(img)
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
    puzzlePiecesInfo = read_puzzle_pieces_info(puzzle_pieces_info_file)

    puzzlePieces = load_puzzle_pieces(os.path.join(shuffledPath, puzzle_name))
    sortedPuzzlePieces = [None] * len(puzzlePieces)

    # Create an empty image large enough to place all pieces
    # Assuming the dimensions based on the world coordinates
    Puzzle = np.zeros((3000, 3000), dtype=np.uint8)  # Adjust the size as needed

    for info, piece in zip(puzzlePiecesInfo, puzzlePieces):
        i, world_x, world_y, angle = info
        rotated_piece = apply_opposite_rotation(piece, angle)

        # Place the piece onto the Puzzle image at its world coordinates
        h, w = rotated_piece.shape
        Puzzle_h, Puzzle_w = Puzzle.shape

        # Ensure the piece fits within the Puzzle boundaries
        end_x = min(world_x + w, Puzzle_w)
        end_y = min(world_y + h, Puzzle_h)

        # Adjust the size of the rotated piece if it goes beyond the Puzzle boundaries
        adjusted_piece = rotated_piece[: end_y - world_y, : end_x - world_x]

        # Place the adjusted piece onto the Puzzle image
        Puzzle[world_y:end_y, world_x:end_x] = adjusted_piece

        # Store the piece in sortedPuzzlePieces according to its original index
        sortedPuzzlePieces[i] = adjusted_piece

    # Save the solved puzzle
    print(f"Saving solved puzzle... {puzzle_name}_solved.png")
    cv2.imwrite(os.path.join(solvedPath, f"{puzzle_name}_solved.png"), Puzzle)

    return sortedPuzzlePieces


# Example usage
puzzle_name = "jigsaw1"  # replace with actual puzzle name
sorted_pieces = solve_puzzle(puzzle_name)
