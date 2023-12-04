import cv2
import csv
import numpy as np
import os
import random
import glob
from scipy import ndimage

from utils import (
    find_bounding_box,
    rotate_image_easy,
    rotate_image,
    read_puzzle_pieces_info,
    ensure_directory_exists,
)


def save_rotated_piece_info(puzzlePiecesInfo, save_path):
    ensure_directory_exists(os.path.dirname(save_path))

    # Write puzzlePiecesInfo to a CSV file
    with open(save_path, "w", newline="") as csvfile:
        # Check if there's at least one piece of info to write
        if puzzlePiecesInfo:
            # Use the keys of the first dictionary as the header
            header = puzzlePiecesInfo[0].keys()
            csvwriter = csv.DictWriter(csvfile, fieldnames=header)
            csvwriter.writeheader()  # Write the header

            # Write the rows based on the dictionary values
            for pI in puzzlePiecesInfo:
                csvwriter.writerow(pI)


def rotate_piece(piece, width):
    new_top_y = width - piece["right_x"]
    new_left_x = piece["top_y"]
    new_bottom_y = width - piece["left_x"]
    new_right_x = piece["bottom_y"]

    new_angle = (piece["angle"] + 90) % 360

    return {
        "top_y": new_top_y,
        "left_x": new_left_x,
        "right_x": new_right_x,
        "bottom_y": new_bottom_y,
        "angle": new_angle,
    }


def calculate_score(piece_metric, piece_solution):
    # Calculate the difference in position and angle (using squared difference)
    dy = pow(piece_metric["top_y"] - piece_solution["top_y"], 2)
    dx = pow(piece_metric["left_x"] - piece_solution["left_x"], 2)
    da = pow(piece_metric["angle"] - piece_solution["angle"], 2)
    # Combine the differences into a single score
    score = dx + dy + da
    return score


def score_puzzle(
    puzzle_meta_data_file, metric_file, solution_file, output_file, puzzle_name
):
    # Read puzzle meta data and piece information from CSV files
    puzzle_meta_data = read_puzzle_pieces_info(puzzle_meta_data_file)[0]
    img_width = puzzle_meta_data["img_width"] + 1
    img_height = puzzle_meta_data["img_height"] + 1

    metric_pieces = read_puzzle_pieces_info(metric_file)
    solution_pieces = read_puzzle_pieces_info(solution_file)

    best_score = float("inf")
    # Iterate for 4 rotations (0, 90, 180, 270 degrees)
    for rotation in range(4):
        total_score = 0

        if rotation == 0:
            rotated_metric_pieces = metric_pieces
        elif rotation == 1 or rotation == 3:
            rotated_metric_pieces = [
                rotate_piece(piece, img_width) for piece in metric_pieces
            ]
        elif rotation == 2:
            rotated_metric_pieces = [
                rotate_piece(piece, img_height) for piece in metric_pieces
            ]

        for rotated_metric, solution in zip(rotated_metric_pieces, solution_pieces):
            total_score += calculate_score(rotated_metric, solution)

        # Save rotated metric piece info
        rotation_file_path = (
            f"Puzzles/Shuffled/{puzzle_name}/Rotation_{rotation * 90}.csv"
        )
        save_rotated_piece_info(rotated_metric_pieces, rotation_file_path)

        best_score = min(best_score, total_score)

        # Update metric pieces for the next iteration
        metric_pieces = rotated_metric_pieces

    # Write the best score to a CSV file
    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Best Score"])
        writer.writerow([best_score])

    print(f"Best score: {best_score}")


def main():
    puzzle_name = "jigsaw1"

    puzzle_meta_data_file_path = (
        "Puzzles/Shuffled/" + puzzle_name + "/puzzle_meta_data.csv"
    )
    metric_file_path = "Puzzles/Shuffled/" + puzzle_name + "/puzzle_pieces_info.csv"
    solution_file_path = "Puzzles/Solved/puzzle_placement.csv"
    # solution_file_path = "Puzzles/Shuffled/" + puzzle_name + "/puzzle_pieces_info.csv"
    score_output_file_path = "Puzzles/Solved/" + puzzle_name + "_grade.csv"

    score_puzzle(
        puzzle_meta_data_file_path,
        metric_file_path,
        solution_file_path,
        score_output_file_path,
        puzzle_name,
    )


if __name__ == "__main__":
    main()
