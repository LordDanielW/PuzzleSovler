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

    # Filter out contours that are too large or too small relative to the average size
    aveSize = sum([len(c) for c in contours]) / len(contours)
    contoursFinal = [
        c for c in contours if len(c) < aveSize * 2 and len(c) > aveSize / 4
    ]

    return contoursFinal


def create_puzzle_pieces(img):
    pieces = []
    piecesShuffled = []
    piecesInfo = []

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

        # Easy rotation
        angle = random.choice(
            [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
        )  # select a random angle from 0, 90, 180, 270
        rotatedPiece = rotate_image_easy(piece, angle)

        pieces.append(rotatedPiece)

        # Store the position and size information in piece_info
        piecesInfo.append(
            {
                "index": i,
                "top_y": y,
                "left_x": x,
                "angle": angle,
            }
        )
    return pieces, piecesInfo

    #
    #   Apply the rotation to singlePiece

    # Hard rotation
    # angle = random.randint(0, 359)  # select a random angle between 0 and 359
    # rotatedPiece = rotate_image(singlePiece, angle)

    # if debugVisuals:
    #     cv2.imshow(f"Piece {i}", rotatedPiece)
    #     cv2.waitKey(0)
    #     print(f"Angle of rotation: {angle}")

    #
    #   Add the piece info to the puzzlePiecesInfo array

    #     puzzlePiecesInfo.append([i, world_x, world_y, 0])
    #     # puzzlePieces.append(rotatedPiece)
    #     puzzlePieces.append(singlePiece)

    # return puzzlePieces, puzzlePiecesInfo

    # # Shuffle the puzzlePiecesInfo array
    # random.shuffle(puzzlePiecesInfo)
    # puzzlePiecesShuffled = [None] * len(puzzlePieces)
    # # Shuffle the puzzle pieces based on the shuffled information
    # for i, info in enumerate(puzzlePiecesInfo):
    #     originalIndex = info[0]  # The original index of the puzzle piece
    #     puzzlePiecesShuffled[i] = puzzlePieces[originalIndex]

    # return puzzlePieces, puzzlePiecesInfo


def save_puzzle_pieces(puzzlePieces, puzzlePiecesInfo, puzzle_name):
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
        puzzlePieces, puzzlePiecesInfo = create_puzzle_pieces(img)

        # Save puzzle pieces
        puzzle_name = os.path.splitext(os.path.basename(file_path))[0]
        save_puzzle_pieces(puzzlePieces, puzzlePiecesInfo, puzzle_name)

    # Wait
    if debugVisuals:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
