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

    def rotate_sides(self):
        self.sides = self.sides[-1:] + self.sides[:-1]
        self.angle = (self.angle + 90) % 360


class PuzzleInfo:
    def __init__(self):
        self.puzzle_name = ""
        self.pieces = []
        self.width = 0
        self.height = 0

class PuzzleSolve:
    def __init__(self):

        rows = 100  # Number of rows
        cols = 100  # Number of columns
        self.puzzle_matrix = [[None for _ in range(cols)] for _ in range(rows)]
        # self.puzzle_matrix = [,,] # y_piece_index, x_piece_index, piece
        self.puzzle_score = 0

    def find_piece(self, piece_index):
        for y_piece_index, y_piece in enumerate(self.puzzle_matrix):
            for x_piece_index, piece in enumerate(y_piece):
                if piece.piece_Index == piece_index:
                    return y_piece_index, x_piece_index
        return -1, -1

    
