import cv2
import csv
import json
import os

from classPuzzle import MetaData


def save_metadata(metadata, file_path):
    """
    Saves MetaData object to a file in JSON format.

    :param metadata: MetaData object to be saved.
    :param file_path: Path of the file where metadata will be saved.
    """
    with open(file_path, "w") as file:
        json.dump(metadata.to_dict(), file)


def load_metadata(file_path):
    """
    Loads MetaData from a file.

    :param file_path: Path of the file from which metadata will be loaded.
    :return: MetaData object.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
        return MetaData(**data)


def save_puzzle_pieces_info(puzzlePiecesInfo, file_folder, file_name):
    ensure_directory_exists(file_folder)
    file_path = os.path.join(file_folder, file_name)
    # Write puzzlePiecesInfo to a CSV file
    with open(file_path, "w", newline="") as csvfile:
        # Check if there's at least one piece of info to write
        if puzzlePiecesInfo:
            # Use the keys of the first dictionary as the header
            header = puzzlePiecesInfo[0].keys()
            csvwriter = csv.DictWriter(csvfile, fieldnames=header)
            csvwriter.writeheader()  # Write the header

            # Write the rows based on the dictionary values
            for pI in puzzlePiecesInfo:
                csvwriter.writerow(pI)


def load_puzzle_pieces_info(csv_file):
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
    pieceInfo = []
    i = 0
    while True:
        filepath = os.path.join(puzzle_folder, f"piece_{i}.png")
        if os.path.exists(filepath):
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                puzzlePieces.append(img)
                pieceInfo.append({"piece_name": f"piece_{i}.png"})
            i += 1
        else:
            break
    return puzzlePieces, pieceInfo


def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
