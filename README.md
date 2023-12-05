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

      solvePuzzles.py
      solvePuzzles_Classical.py

- To score the puzzles

  `scorePuzzle.py`

## Puzzle Folder Hierarchy

Puzzles/

- SVG/
- Original/
- Shuffled/
  - jigSaw1/
  - jigSaw2/
  - etc.
- Solved/

## Installation

      python -m venv sessionFP
      sessionFP\Scripts\activate
      python.exe -m pip install --upgrade pip
      pip install -r requirements.txt

for SVG to PNG only:
https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases

for pytorch only:
https://developer.nvidia.com/cuda-zone

### added to requirements.txt (for pip install -r requirements.txt)

pip install cairosvg  
pip install selenium  
pip install numpy  
pip install opencv-python  
pip install scipy  
pip install matplotlib
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install scikit-learn
