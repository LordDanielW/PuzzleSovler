class SideMatch:
    def __init__(self):
        self.piece_index = 0
        self.side_index = 0
        self.histogram_score = 0

class SideInfo:
    def __init__(self):
        self.side_Index = 0
        self.Histogram = []
        self.isEdge = False
        self.start_corner_index = 0
        self.end_corner_index = 0
        self.side_matches = SideMatch()


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


class PuzzleInfo:
    def __init__(self):
        self.puzzle_name = ""
        self.pieces = []
        self.width = 0
        self.height = 0
