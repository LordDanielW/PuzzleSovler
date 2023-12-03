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

    def rotate_sides(self):
        self.sides = self.sides[-1:] + self.sides[:-1]
        self.angle = (self.angle + 90) % 360


class PuzzleInfo:
    def __init__(self):
        self.puzzle_name = ""
        self.pieces = []
        self.width = 0
        self.height = 0

class PieceSolve:
    x_piece_index = 0
    y_piece_index = 0
    piece = PieceInfo()

class PuzzleSolve:
    def __init__(self):
        self.pieces = {}  # Dictionary to hold pieces with keys as [y, x]
        self.puzzle_score = 0  # Initialize puzzle score

    def add_piece(self, y, x, piece):
        """Add a piece to the puzzle at coordinates [y, x]."""
        if not isinstance(piece, PieceInfo):
            raise ValueError("piece must be an instance of PieceInfo class")
        self.pieces[(y, x)] = piece

    def find_piece(self, search_piece):
        """Find and return the coordinates of a piece that matches the search_piece's piece_index."""
        for (y, x), piece in self.pieces.items():           
            if piece.piece_Index == search_piece.piece_Index:
                return (y, x)
        return None  # Return None if no matching piece is found

    def update_score(self, score):
        """Update the puzzle score."""
        self.puzzle_score = score

