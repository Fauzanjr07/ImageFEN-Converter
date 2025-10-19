# ConverterChess — Image to FEN

Convert a chessboard image into a FEN string using a CNN classifier (OCR-style) that recognizes the piece in each square. Includes training and inference scripts.

## Features
- Board detection and perspective warp from arbitrary photos (OpenCV)
- 8x8 square splitting and batch classification
- Trainable CNN (ResNet18) on your dataset (PyTorch + TorchVision)
- FEN assembly with proper empty-square compression
- Windows-friendly helper scripts (.cmd)

## Project structure
- `src/converter/` — Core library code (board detection, dataset, model, training, prediction)
- `data/` — Put your training images here (empty folders provided)
- `weights/` — Trained model checkpoints and metadata
- `scripts/` — Convenience wrappers for Windows shell

## Install
1) Install Python 3.9–3.12
2) Create a virtual environment (recommended) and install deps:

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If PyTorch fails to install from pip, follow the official instructions for your CUDA/CPU platform:
https://pytorch.org/get-started/locally/

## Prepare your data
This project trains a classifier on single-square images, one of 13 classes:
- empty
- white_pawn, white_knight, white_bishop, white_rook, white_queen, white_king
- black_pawn, black_knight, black_bishop, black_rook, black_queen, black_king

Place images (PNG/JPG) into the provided folders under:
- `data/train/<class>/...`
- `data/val/<class>/...` (validation set)
- `data/test/<class>/...` (optional test set)

Tip: Use `src/converter/boards.py` and `src/converter/predict.py --debug-save-squares` to auto-split a board photo into 64 squares you can hand-label into the above folders for bootstrapping.

## Train
Train a ResNet18 classifier on your square images:

```cmd
scripts\train.cmd --data-root data --epochs 10 --batch-size 64 --lr 0.0005 --img-size 224 --out-dir weights
```

Common flags:
- `--data-root` folder containing `train/` and `val/`
- `--epochs` number of training epochs
- `--batch-size` per-GPU batch size (this project assumes CPU or single GPU)
- `--img-size` square image size for the network (default 224)
- `--out-dir` where to save checkpoints and class mapping

Outputs:
- `weights/best.pt` — best checkpoint by validation accuracy
- `weights/classes.json` — class names in training order

## Predict (Image -> FEN)
Given a photo, detect the board, split into 64 squares, classify each square, and output FEN:

```cmd
scripts\predict.cmd --image path\to\board.jpg --weights weights\best.pt --classes weights\classes.json
```

Options:
- `--flip` force a 180° rotation if your board is upside down
- `--save-overlays` save an annotated visualization next to the input
- `--debug-save-squares` save the 64 cropped squares to a folder for inspection/labeling

The script prints the FEN to stdout. It assumes the warped board’s top-left is a dark square (a8). The pipeline tries to auto-rotate, but you can override with `--flip`.

## Notes and tips
- Data quality matters a lot: ensure consistent framing, focus, and lighting. Larger datasets yield better generalization.
- If board detection struggles, try placing the board centrally with clear contrast from background.
- Start training with a few dozen squares per class, then iteratively improve via hard examples collected from predictions.

## License
This project is provided as-is. You own your data and trained models.
