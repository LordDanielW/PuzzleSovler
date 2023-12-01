# PuzzleSovler

## To Run

- To download the puzzles from the website

  `importPuzzles.py`

- To convert the SVG files to PNG

  - Move puzzles to the folder Puzzles/SVG
  - Then Run

    `convert_SVG_PNG.py`

- To shuffle the puzzles

  `shufflePuzzles.py`

- To solve the puzzles

      solvePuzzles_StuffMadeHere.py
      solvePuzzles_Tom.py
      solvePuzzles_Nerual_Basic.py
      solvePuzzles_Nerual_Advanced.py

- To check the puzzles

  `checkPuzzles.py`

- To train the neural network

  `train_Nerual_Basic.py`
  `train_Nerual_Advanced.py`

## Puzzle Folder Hierarchy

Puzzles/

- SVG/
- Original/
- Shuffled/
- Solved/
  - StuffMadeHere/
  - Tom/
  - Basic/
  - Advanced/
- Training/
  - Basic/
  - Advanced/

## Installation

      python -m venv sessionFP
      sessionFP\Scripts\activate
      python.exe -m pip install --upgrade pip
      pip install -r requirements.txt

for SVG to PNG only:
https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases

### added to requirements.txt (for pip install -r requirements.txt)

pip install cairosvg

pip install selenium

pip install numpy

pip install opencv-python

pip install scipy

