import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import csv
import os
from definePiece import segmentSides

from utilsLoad import load_metadata, load_puzzle_pieces, load_puzzle_pieces_info
from scorePuzzle import score_puzzle
from classPuzzle import PuzzleInfo, PieceInfo
from utilsMath import rotate_image_easy

from sklearn.model_selection import train_test_split

shuffledPath = "Puzzles/Shuffled/"
puzzle_name = "jigsaw1"


# Define your neural network architecture here
class PuzzleSolverNN(nn.Module):
    def __init__(self):
        super(PuzzleSolverNN, self).__init__()
        # Example architecture: simple CNN for image processing
        self.conv1 = nn.Conv2d(
            1, 16, kernel_size=3, stride=1, padding=1
        )  # Adjust in_channels based on your image format
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(
            64 * 8 * 8, 512
        )  # Adjust the input features of fc1 based on the output of the last conv layer
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation function, assuming a regression problem
        return x


# Custom dataset for loading puzzle pieces
class PuzzleDataset(Dataset):
    def __init__(self, puzzle_pieces, targets=None, transform=None):
        self.puzzle_pieces = puzzle_pieces
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.puzzle_pieces)

    def __getitem__(self, idx):
        piece = self.puzzle_pieces[idx]
        if self.transform:
            piece = self.transform(piece)

        # If targets are provided, return both image and target
        if self.targets is not None:
            target = self.targets[idx]
            return piece, target
        else:
            return piece


# Define a transform to resize the images
transform = transforms.Compose(
    [
        transforms.ToPILImage(),  # Convert numpy array or tensor to PIL Image
        transforms.Resize((224, 224)),  # Resize to a fixed size, e.g., 224x224
        transforms.ToTensor(),  # Convert PIL Image back to tensor
    ]
)


def save_predicted_data(outputs, file_path):
    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Convert outputs to numpy array if it's a tensor
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.cpu().detach().numpy()

    # Assuming outputs is a 2D array where each row corresponds to a puzzle piece
    # and contains [top_y, left_x, bottom_y, right_x, angle]
    with open(file_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write the header
        writer.writerow(["top_y", "left_x", "bottom_y", "right_x", "angle"])

        # Write each row of outputs
        for row in outputs:
            writer.writerow(row)


def process_outputs(outputs):
    # Assuming outputs is a tensor of shape (batch_size, num_features)
    # where num_features corresponds to [top_y, left_x, bottom_y, right_x, angle]
    processed_outputs = []

    # Convert outputs to numpy array for easier processing
    outputs = outputs.cpu().detach().numpy()

    for output in outputs:
        # Post-process each output as needed, e.g., rounding or scaling
        # Adjust this part based on how your network's output correlates to the actual values
        top_y, left_x, bottom_y, right_x, angle = output

        # Example: rounding off the values
        top_y, left_x, bottom_y, right_x = map(
            round, [top_y, left_x, bottom_y, right_x]
        )
        angle = round(angle) % 360  # Ensuring angle is within 0-359 degrees

        processed_outputs.append([top_y, left_x, bottom_y, right_x, angle])

    return processed_outputs


def evaluate_model(
    model, dataloader, puzzle_meta_data_file_path, solution_file_path, puzzle_name
):
    model.eval()
    total_score = 0
    with torch.no_grad():
        for batch in dataloader:
            images, _ = batch
            outputs = model(images)

            # Process outputs to match the format expected by score_puzzle
            processed_outputs = process_outputs(outputs)

            # Save the processed outputs temporarily to use with score_puzzle
            temp_output_path = "temp_predicted.csv"
            save_predicted_data(processed_outputs, temp_output_path)

            # Calculate the score
            score = score_puzzle(
                puzzle_meta_data_file_path,
                temp_output_path,
                solution_file_path,
                None,  # Output file not needed here
                puzzle_name,
            )
            total_score += score

    return total_score / len(dataloader)


def train(
    model,
    train_dataloader,
    val_dataloader,
    criterion,
    optimizer,
    epochs,
    puzzle_meta_data_file_path,
    solution_file_path,
    puzzle_name,
):
    for epoch in range(epochs):
        model.train()
        for batch in train_dataloader:
            images, targets = batch
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Evaluate the model at the end of each epoch
        avg_score = evaluate_model(
            model,
            val_dataloader,
            puzzle_meta_data_file_path,
            solution_file_path,
            puzzle_name,
        )
        print(f"Epoch {epoch+1}, Average Score: {avg_score}")


def solve_puzzle_with_nn(model, puzzle_pieces):
    model.eval()
    solved_puzzle = []

    with torch.no_grad():
        for piece_tensor in puzzle_pieces:
            # Ensure the piece tensor is in the correct shape for the model
            if len(piece_tensor.shape) == 3:
                piece_tensor = piece_tensor.unsqueeze(
                    0
                )  # Add batch dimension if needed

            # Predict the position and orientation of each piece
            output = model(piece_tensor)

            # Convert the model output to a readable format (e.g., numpy array)
            predicted_data = output.squeeze().cpu().numpy()

            # Interpret the output (assuming the output is [top_y, left_x, bottom_y, right_x, angle])
            top_y, left_x, bottom_y, right_x, angle = predicted_data

            # Store or process the predicted data as needed
            solved_puzzle.append(
                {
                    "top_y": top_y,
                    "left_x": left_x,
                    "bottom_y": bottom_y,
                    "right_x": right_x,
                    "angle": angle,
                }
            )

    return solved_puzzle


def split_data(data):
    images = [item[0] for item in data]
    targets = [item[1] for item in data]
    return images, targets


def main():
    # Load puzzle pieces and metadata
    raw_pieces, piecesInfo = load_puzzle_pieces(os.path.join(shuffledPath, puzzle_name))
    meta_data = load_metadata(
        os.path.join(shuffledPath, puzzle_name, "puzzle_meta_data.json")
    )

    # Initialize PuzzleInfo
    puzzle = PuzzleInfo()
    processed_pieces = []  # List to store processed tensors

    # Define and process all pieces
    for i, raw_piece in enumerate(raw_pieces):
        piece = PieceInfo()
        # Segment sides of the piece (modify segmentSides as needed)
        piece = segmentSides(raw_piece, False, 4, 3)
        piece.piece_Index = i
        piece.piece_name = piecesInfo[i]["piece_name"]
        puzzle.pieces.append(piece)

        # Convert and normalize the piece image
        raw_piece = np.array(piece.puzzle_piece)
        piece_tensor = torch.from_numpy(raw_piece).float() / 255.0
        if len(piece_tensor.shape) == 2:
            piece_tensor = piece_tensor.unsqueeze(0)
        processed_pieces.append(piece_tensor)

    # Split the data into training and validation sets
    train_data, val_data = train_test_split(
        processed_pieces, test_size=0.2, random_state=42
    )

    # Split the data into images and targets
    train_images, train_targets = split_data(train_data)
    val_images, val_targets = split_data(val_data)

    # Create datasets for training and validation
    train_dataset = PuzzleDataset(train_images, train_targets, transform=transform)
    val_dataset = PuzzleDataset(val_images, val_targets, transform=transform)

    # Create dataloaders for training and validation
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize the neural network, criterion, and optimizer
    model = PuzzleSolverNN()
    criterion = nn.MSELoss()  # Example loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define file paths for puzzle metadata and solution
    puzzle_meta_data_file_path = os.path.join(
        shuffledPath, puzzle_name, "puzzle_meta_data.json"
    )
    solution_file_path = os.path.join("Puzzles/Solved", f"{puzzle_name}_solved.csv")

    # Train the model
    epochs = 10
    train(
        model,
        train_dataloader,
        val_dataloader,  # Replace with actual validation dataloader
        criterion,
        optimizer,
        epochs,
        puzzle_meta_data_file_path,
        solution_file_path,
        puzzle_name,
    )

    # Optionally, you can use score_puzzle to evaluate the solution
    # score = score_puzzle(...)


if __name__ == "__main__":
    main()
