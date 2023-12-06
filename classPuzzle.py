import cv2
import numpy as np

# *** Note: don't inclue utilsLoad for circular dependency ***

from utilsMath import distance_squared_average, rotate_image_easy, rotate_points_list
from utilsDraw import scale_piece


class SideMatch:
    def __init__(self):
        self.piece_index = 0
        self.side_index = 0
        self.histogram_score = 0
        self.histogram_shift = 0


class SideInfo:
    def __init__(self):
        self.side_Index = 0
        self.Histogram = []
        self.Points = []
        self.isEdge = False
        self.start_corner_index = 0
        self.end_corner_index = 0
        self.side_matches = [SideMatch()]


class PieceInfo:
    def __init__(self):
        self.piece_Index = 0
        self.piece_name = ""
        self.sides = [
            SideInfo(),
            SideInfo(),
            SideInfo(),
            SideInfo(),
        ]  # Four SideInfo instances
        self.rotate_sides_angle = 0

        self.puzzle_piece = []
        self.puzzle_contours_all = []
        self.puzzle_sampled_contours = []

        self.isCorner = False
        self.isEdge = False

        self.top_y = 0
        self.left_x = 0
        self.bottom_y = 0
        self.right_x = 0
        self.angle = 0

    # TODO: Validate this function
    # def rotate_quick(self):
    #     self.sides = self.sides[-1:] + self.sides[:-1]
    #     self.angle = (self.angle + 90) % 360
    #     self.puzzle_piece = rotate_image_easy(
    #         self.puzzle_piece, cv2.ROTATE_90_CLOCKWISE
    #     )

    def rotate_piece_deep(self):
        height, width = self.puzzle_piece.shape[:2]

        # Rotating edgePoints using rotate_points_list
        for side in self.sides:
            side.Points = rotate_points_list(side.Points, width, height)

        # Rotate start and end corner indices
        for side in self.sides:
            side.start_corner_index = (side.start_corner_index + 1) % 4
            side.end_corner_index = (side.end_corner_index + 1) % 4

        # Rotate Histograms, Points, start_corner_index, end_corner_index
        self.sides = self.sides[-1:] + self.sides[:-1]

        # Rotate puzzle piece image
        self.puzzle_piece = cv2.rotate(self.puzzle_piece, cv2.ROTATE_90_CLOCKWISE)

        # Rotate puzzle_contours_all and puzzle_sampled_contours
        self.puzzle_contours_all = [
            rotate_points_list(contour, width, height)
            for contour in self.puzzle_contours_all
        ]
        self.puzzle_sampled_contours = [
            rotate_points_list(contour, width, height)
            for contour in self.puzzle_sampled_contours
        ]

        # Update the angles and positions
        self.angle = (self.angle + 90) % 360
        self.top_y = width - self.right_x
        self.left_x = self.top_y
        self.bottom_y = width - self.left_x
        self.right_x = self.bottom_y

    def rotate_to_top(self):
        # Find the side which should be the top
        # max_x2_minus_yxAll = float("inf")
        # top_side_index = 0

        # for i, side in enumerate(self.sides):
        #     x1, y1 = self.puzzle_sampled_contours[side.start_corner_index][0]
        #     x2, y2 = self.puzzle_sampled_contours[side.end_corner_index][0]
        #     x2_minus_yx = x2 - y2 - x1 - y1

        #     if (x2_minus_yx) < max_x2_minus_yxAll:
        #         top_side_index = i

        # Identify the side with the minimum average y-value which will be considered the top side.
        min_avg_y = float("inf")
        top_side_index = -1

        for i, side in enumerate(self.sides):
            avg_y = np.mean([pt[0][1] for pt in side.Points])
            if avg_y < min_avg_y:
                min_avg_y = avg_y
                top_side_index = i
        print(f"Top side index: {top_side_index}")

        # Rotate the piece until the identified top side is in the 0th index
        counter = 0
        while top_side_index != 0:
            counter += 1

            print(f"Rotating piece {counter} times")
            # Rotate Histograms, Points, start_corner_index, end_corner_index
            self.sides = self.sides[-1:] + self.sides[:-1]

            top_side_index = (top_side_index + 1) % 4


class PuzzleInfo:
    def __init__(self):
        self.puzzle_name = ""
        self.pieces = []
        self.width = 0
        self.height = 0


class MetaData:
    def __init__(self, _seed, _tabsize, _jitter, xn, yn, width, height):
        self._seed = _seed
        self._tabsize = _tabsize
        self._jitter = _jitter
        self.xn = xn
        self.yn = yn
        self.width = width
        self.height = height

    def __str__(self):
        return f"MetaData(_seed={self._seed}, _tabsize={self._tabsize}, _jitter={self._jitter}, xn={self.xn}, yn={self.yn}, width={self.width}, height={self.height})"

    def to_dict(self):
        return {
            "_seed": self._seed,
            "_tabsize": self._tabsize,
            "_jitter": self._jitter,
            "xn": self.xn,
            "yn": self.yn,
            "width": self.width,
            "height": self.height,
        }


class PuzzleSolve:
    def __init__(self, metadata):
        if not isinstance(metadata, MetaData):
            raise ValueError("metadata must be an instance of MetaData class")
        self.metadata = metadata
        # self.pieces = {}  # Dictionary to hold pieces with keys as [y, x]
        self.pieces = {
            (y, x): None for y in range(metadata.yn - 1) for x in range(metadata.xn - 1)
        }
        self.puzzle_score = 0  # Initialize puzzle score

    def add_piece(self, y, x, piece, showPuzzle=False):
        """Add a piece to the puzzle at coordinates [y, x]."""
        if not isinstance(piece, PieceInfo):
            raise ValueError("piece must be an instance of PieceInfo class")
        self.pieces[(y, x)] = piece

        # Update piece coordinates
        if y == 0 and x == 0:
            piece.top_y = 0
            piece.left_x = 0
            piece.bottom_y = piece.puzzle_piece.shape[0]
            piece.right_x = piece.puzzle_piece.shape[1]
        else:
            if y != 0:
                piece.top_y = self.pieces[(y - 1, x)].bottom_y
                piece.bottom_y = piece.top_y + piece.puzzle_piece.shape[0]
            else:
                piece.top_y = 0
                piece.bottom_y = piece.puzzle_piece.shape[0]
            if x != 0:
                piece.left_x = self.pieces[(y, x - 1)].right_x
                piece.right_x = piece.left_x + piece.puzzle_piece.shape[1]
            else:
                piece.left_x = 0
                piece.right_x = piece.puzzle_piece.shape[1]

        # Update puzzle score
        if y == 0 and x == 0:
            self.puzzle_score = 0.0
        else:
            if y != 0:
                y_dist_sqrd, _ = distance_squared_average(
                    piece.sides[0].Points,
                    self.pieces[(y - 1, x)].sides[2].Points,
                )
                self.puzzle_score = self.puzzle_score + y_dist_sqrd
            if x != 0:
                x_dist_sqrd, _ = distance_squared_average(
                    piece.sides[3].Points,
                    self.pieces[(y, x - 1)].sides[1].Points,
                )
                self.puzzle_score = self.puzzle_score + x_dist_sqrd

        if showPuzzle:
            self.show_puzzle()

    def find_piece(self, search_piece):
        """Find and return the coordinates of a piece that matches the search_piece's piece_index."""
        for (y, x), piece in self.pieces.items():
            if piece is None:
                continue
            if piece.piece_Index == search_piece.piece_Index:
                return (y, x)
        return None  # Return None if no matching piece is found

    def show_puzzle(self):
        """Show the puzzle image."""
        puzzle_image = self.generate_puzzle_image()
        scale_piece(puzzle_image, "Puzzle", 0.5, True, True)

    def generate_puzzle_image(self):
        error_margin = 2.5
        puzzle_image = np.zeros(
            (
                int(self.metadata.height * error_margin),
                int(self.metadata.width * error_margin),
            ),
            dtype=np.uint8,
        )

        for (y, x), piece in self.pieces.items():
            if piece is None:
                continue

            add_piece = piece.puzzle_piece
            mask = add_piece == 255

            # Debugging shapes and coordinates
            subarray_shape = puzzle_image[
                piece.top_y : piece.bottom_y, piece.left_x : piece.right_x
            ].shape
            mask_shape = mask.shape
            print(f"Subarray shape: {subarray_shape}, Mask shape: {mask_shape}")

            # Ensure the shapes match before assignment
            if subarray_shape == mask_shape:
                puzzle_image[
                    piece.top_y : piece.bottom_y, piece.left_x : piece.right_x
                ][mask] = add_piece[mask]
            else:
                print("Mismatch in shapes detected")

            # Debug: Check dimensions
            print(f"Mask shape: {mask.shape}, Puzzle piece shape: {add_piece.shape}")
            print(
                f"Top_y: {piece.top_y}, Bottom_y: {piece.bottom_y}, Left_x: {piece.left_x}, Right_x: {piece.right_x}"
            )
            print(
                f"Subarray shape: {puzzle_image[piece.top_y : piece.bottom_y, piece.left_x : piece.right_x].shape}"
            )

            puzzle_image[piece.top_y : piece.bottom_y, piece.left_x : piece.right_x][
                mask
            ] = add_piece[mask]

        return puzzle_image
