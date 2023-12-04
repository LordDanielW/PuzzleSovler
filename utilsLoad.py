import cv2
import csv
import os

from puzzleClass import MetaData


def read_metadata(file_path):
    with open(file_path, mode="r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            metadata = MetaData(
                _seed=int(row["_seed"]),
                _tabsize=int(row["_tabsize"]),
                _jitter=int(row["_jitter"]),
                xn=int(row["xn"]),
                yn=int(row["yn"]),
                width=int(row["width"]),
                height=int(row["height"]),
            )
            return metadata


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
