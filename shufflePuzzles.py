import cv2
import csv
import numpy as np
import os
import random
import glob
from scipy import ndimage

from utils import find_bounding_box, rotate_image_easy, rotate_image

debugVisuals = False
originalPath = "Puzzles/Original/"
shuffledPath = "Puzzles/Shuffled/"
solvedPath = "Puzzles/Solved/"


def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_image(file_path):
    # Read in the image.
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: Could not read the image from {file_path}")
        return None  # You could handle the error or exception according to your project's requirements.

    return img


def find_contours(img):
    inverted = 255 - img
    # if debugVisuals:
    #     cv2.imshow("Inverted", inverted)

    _, thresh = cv2.threshold(inverted, 25, 255, cv2.THRESH_BINARY)
    # if debugVisuals:
    #     cv2.imshow("Thresholded", thresh)

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Sort contours by size (number of points in contour) in ascending order
    contours_sorted = sorted(contours, key=lambda x: len(x), reverse=False)

    # Remove the largest contour (the background)
    contours_sorted.pop()

    return contours_sorted


def create_puzzle_pieces(img):
    pieces = []
    piecesInfo = []
    puzzle_meta_data = []

    # This finds the puzzle pieces based on contours
    contoursFinal = find_contours(img)

    for i, contour in enumerate(contoursFinal):
        # Calculate the bounding rectangle for each contour
        x, y, w, h = cv2.boundingRect(contour)
        # Create blank BGR image
        piece = np.zeros((h, w, 3), dtype=np.uint8)
        # Draw contour on piece
        contour_offset = contour - [x, y]
        # Draw filled contour in white
        cv2.drawContours(
            piece, [contour_offset], -1, (255, 255, 255), thickness=cv2.FILLED
        )
        # Draw contour outline in black
        cv2.drawContours(piece, [contour_offset], -1, (0, 0, 0), thickness=1)
        # Convert to grayscale
        piece = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
        # Convert to binary
        _, piece = cv2.threshold(piece, 125, 255, cv2.THRESH_BINARY)

        # A mapping from cv2 rotation constants to degrees
        rotation_to_angle = {
            cv2.ROTATE_90_CLOCKWISE: 90,
            cv2.ROTATE_180: 180,
            cv2.ROTATE_90_COUNTERCLOCKWISE: 270,
        }

        # Easy rotation
        cv2_rotation_constant = random.choice(
            list(rotation_to_angle.keys())
        )  # select a random rotation constant
        angle_in_degrees = rotation_to_angle[
            cv2_rotation_constant
        ]  # convert the constant to degrees
        rotatedPiece = rotate_image_easy(piece, cv2_rotation_constant)

        # Hard rotation
        # angle = random.randint(0, 359)  # select a random angle between 0 and 359
        # rotatedPiece = rotate_image(piece, angle)

        pieces.append(rotatedPiece)

        # Store the position and size information in piece_info
        piecesInfo.append(
            {
                "index": i,
                "top_y": y,
                "left_x": x,
                "bottom_y": y + h - 1,  # Subtract 1 because pixel indices start at 0
                "right_x": x + w - 1,  # Subtract 1 because pixel indices start at 0
                "angle": angle_in_degrees,
            }
        )

    # Shuffle the puzzlePiecesInfo array
    random.shuffle(piecesInfo)
    puzzlePiecesShuffled = [None] * len(pieces)
    # Shuffle the puzzle pieces based on the shuffled information
    for i, info in enumerate(piecesInfo):
        originalIndex = info["index"]  # The original index of the puzzle piece
        puzzlePiecesShuffled[i] = pieces[originalIndex]

        # Add Piece Name here after shuffling
        info["piece_name"] = f"piece_{i}.png"

    # Generate Meta Data
    img_height, img_width = img.shape[:2]
    puzzle_meta_data.append(
        {
            "num_pieces": len(pieces),
            "img_height": img_height,
            "img_width": img_width,
        }
    )

    return puzzlePiecesShuffled, piecesInfo, puzzle_meta_data


def save_puzzle_pieces(puzzlePieces, puzzlePiecesInfo, puzzle_meta_data, puzzle_name):
    current_shuffled_path = os.path.join(shuffledPath, puzzle_name)
    ensure_directory_exists(current_shuffled_path)

    # Write puzzlePiecesInfo to a CSV file
    csv_filename = os.path.join(current_shuffled_path, "puzzle_pieces_info.csv")
    with open(csv_filename, "w", newline="") as csvfile:
        # Check if there's at least one piece of info to write
        if puzzlePiecesInfo:
            # Use the keys of the first dictionary as the header
            header = puzzlePiecesInfo[0].keys()
            csvwriter = csv.DictWriter(csvfile, fieldnames=header)
            csvwriter.writeheader()  # Write the header

            # Write the rows based on the dictionary values
            for pI in puzzlePiecesInfo:
                csvwriter.writerow(pI)

    # Write puzzle_meta_data to a CSV file
    csv_filename = os.path.join(current_shuffled_path, "puzzle_meta_data.csv")
    with open(csv_filename, "w", newline="") as csvfile:
        # Check if there's at least one piece of info to write
        if puzzle_meta_data:
            # Use the keys of the first dictionary as the header
            header = puzzle_meta_data[0].keys()
            csvwriter = csv.DictWriter(csvfile, fieldnames=header)
            csvwriter.writeheader()  # Write the header

            # Write the rows based on the dictionary values
            for pI in puzzle_meta_data:
                csvwriter.writerow(pI)

    # Save each puzzle piece as an image
    for i, piece in enumerate(puzzlePieces):
        piece_filename = f"piece_{i}.png"  # Saving as .png
        piece_path = os.path.join(current_shuffled_path, piece_filename)
        cv2.imwrite(piece_path, piece)


def main():
    ensure_directory_exists(shuffledPath)

    # Loop through all the puzzles in the original folder
    for file_path in glob.glob(os.path.join(originalPath, "*.png")):
        # Read in a puzzle
        img = read_image(file_path)
        if img is None:
            return  # If the image was not read properly, skip this iteration

        # Generate puzzle pieces
        puzzlePieces, puzzlePiecesInfo, puzzle_meta_data = create_puzzle_pieces(img)

        # Save puzzle pieces
        puzzle_name = os.path.splitext(os.path.basename(file_path))[0]
        save_puzzle_pieces(
            puzzlePieces, puzzlePiecesInfo, puzzle_meta_data, puzzle_name
        )

        print(f"Shuffled {puzzle_name}")

    # Wait
    if debugVisuals:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
