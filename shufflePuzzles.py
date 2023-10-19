import cv2
import numpy as np
import os
import random
import glob
from scipy import ndimage

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


def process_image(img):
    inverted = 255 - img
    if debugVisuals:
        cv2.imshow("Inverted", inverted)
    _, thresh = cv2.threshold(inverted, 25, 255, cv2.THRESH_BINARY)
    if debugVisuals:
        cv2.imshow("Thresholded", thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return contours


def filter_contours(contours):
    aveSize = sum([len(c) for c in contours]) / len(contours)
    contoursFinal = [
        c for c in contours if len(c) < aveSize * 2 and len(c) > aveSize / 4
    ]
    return contoursFinal


def find_bounding_box(image):
    # For grayscale image, we don't compare against a 3-channel color.
    # Instead, we simply look for pixels that are not black (i.e., not 0).
    pts = np.argwhere(image > 0)  # Assuming black pixels have value 0.

    # Find the minimum and maximum x and y coordinates.
    min_y, min_x = pts.min(axis=0)
    max_y, max_x = pts.max(axis=0)

    # Create a box around those points.
    # Adding a border of 5 pixels around the non-black area.
    min_y = max(min_y - 5, 0)
    min_x = max(min_x - 5, 0)
    max_y = min(max_y + 5, image.shape[0])
    max_x = min(max_x + 5, image.shape[1])

    # Return the coordinates of the box.
    return [(min_x, min_y), (max_x, max_y)]


def rotate_image(image, angle):
    # Define the color of the border and the size of the border.
    background_color = (0, 0, 0)  # black
    border_size = max(image.shape)  # You may adjust this size as needed.

    # Create an expanded image by adding a border around the original image.
    expanded_image = cv2.copyMakeBorder(
        image,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=background_color,
    )

    # Using scipy's ndimage to rotate the image within the expanded border.
    # 'reshape=False' ensures the output shape is the same as the input image.
    # 'cval=0' sets the background to black where there are no image data.
    # 'order=1' (bilinear interpolation) is used here, but you can adjust as needed.
    rotated_image = ndimage.rotate(
        expanded_image, angle, reshape=False, cval=0, order=1
    )

    # Find the bounding box of the non-black areas.
    box = find_bounding_box(rotated_image)

    # Crop the image using the bounding box.
    cropped_image = rotated_image[box[0][1] : box[1][1], box[0][0] : box[1][0]]

    return cropped_image


def create_puzzle_pieces(contoursFinal):
    puzzlePieces = []
    padding = 5

    for contour in contoursFinal:
        x, y, w, h = cv2.boundingRect(contour)
        contour = contour - [x - padding, y - padding]
        singlePiece = np.zeros((h + 2 * padding, w + 2 * padding, 3), dtype=np.uint8)
        cv2.drawContours(
            singlePiece, [contour], -1, (255, 255, 255), thickness=cv2.FILLED
        )
        singlePiece = cv2.cvtColor(singlePiece, cv2.COLOR_BGR2GRAY)
        _, singlePiece = cv2.threshold(singlePiece, 125, 255, cv2.THRESH_BINARY)
        puzzlePieces.append(singlePiece)

        if debugVisuals:
            cv2.imshow(f"Piece {len(puzzlePieces)}", singlePiece)
            cv2.waitKey(0)

    # Adding random rotation and shuffling
    for i, piece in enumerate(puzzlePieces):
        # Decide the rotation angle
        angle = random.randint(0, 359)  # select a random angle between 0 and 359

        # Now, apply the rotation
        puzzlePieces[i] = rotate_image(piece, angle)

        if debugVisuals:
            cv2.imshow(f"Piece Rotated {i}", puzzlePieces[i])
            cv2.waitKey(0)

    random.shuffle(puzzlePieces)
    return puzzlePieces


def save_puzzle_pieces(puzzlePieces, puzzle_name):
    current_shuffled_path = os.path.join(shuffledPath, puzzle_name)
    ensure_directory_exists(current_shuffled_path)

    for i, piece in enumerate(puzzlePieces):
        piece_filename = f"piece_{i}.png"  # Modified to save as .png
        piece_path = os.path.join(current_shuffled_path, piece_filename)
        cv2.imwrite(piece_path, piece)


def handle_puzzle(file_path):
    # Reading and processing the image
    img = read_image(file_path)
    if img is None:
        return  # If the image was not read properly, skip this iteration

    contours = process_image(img)
    contoursFinal = filter_contours(contours)

    # Creating and saving puzzle pieces
    puzzlePieces = create_puzzle_pieces(contoursFinal)
    puzzle_name = os.path.splitext(os.path.basename(file_path))[0]
    save_puzzle_pieces(puzzlePieces, puzzle_name)


def main():
    ensure_directory_exists(shuffledPath)

    for file_path in glob.glob(os.path.join(originalPath, "*.png")):
        handle_puzzle(file_path)

    if debugVisuals:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
