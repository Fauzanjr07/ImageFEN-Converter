"""
Labeling GUI for ImageFEN-Converter
===================================

Two workflows:
1) squares mode  : You already have 64 cropped squares (e.g., from --debug-save-squares).
                   The tool shows each image; press a hotkey to assign a class.
2) board mode    : Provide a top-down board image (already rectified). The tool splits it
                   into an 8x8 grid, shows one square at a time; you label & it saves crops.

Hotkeys (default):
  0 or e      -> empty
  p r n b q k -> white pawn/rook/knight/bishop/queen/king
  P R N B Q K -> black pawn/rook/knight/bishop/queen/king

Other keys:
  SPACE       -> next (if you want to skip labeling a square in board mode)
  u           -> undo last saved crop (only current session)
  [ / ]       -> previous / next image (squares mode)
  g           -> toggle grid overlay (board mode)
  ESC         -> quit (progress saved)

Outputs:
  Saves images into: <output_root>/<split>/<class_name>/...
  Where split is 'train' or 'val' depending on --val-split probability.
  Also logs to a session CSV under <output_root>/label_log_<timestamp>.csv

Usage examples:
  # 1) Label existing 64-square crops
  python labeler.py --mode squares --input_dir debug_squares --output_root data --val-split 0.1

  # 2) Split a top-down board image into 8x8 and label each square
  python labeler.py --mode board --board_img path/to/board.jpg --output_root data --val-split 0.1

Requirements: opencv-python, numpy, pillow (optional)
"""

import argparse
import os
import sys
import csv
import time
import random
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np

# =====================
# Configurable classes
# =====================
CLASS_NAMES: List[str] = [
    "empty",
    "white_pawn", "white_rook", "white_knight", "white_bishop", "white_queen", "white_king",
    "black_pawn", "black_rook", "black_knight", "black_bishop", "black_queen", "black_king",
]

# Hotkey -> class_name mapping
HOTKEYS = {
    ord('0'): "empty",
    ord('e'): "empty",
    # white
    ord('p'): "white_pawn",
    ord('r'): "white_rook",
    ord('n'): "white_knight",
    ord('b'): "white_bishop",
    ord('q'): "white_queen",
    ord('k'): "white_king",
    # black (uppercase)
    ord('P'): "black_pawn",
    ord('R'): "black_rook",
    ord('N'): "black_knight",
    ord('B'): "black_bishop",
    ord('Q'): "black_queen",
    ord('K'): "black_king",
}

HELP_TEXT = " | ".join([
    "0/e=empty",
    "p/r/n/b/q/k=white",
    "P/R/N/B/Q/K=black",
    "u=undo",
    "[ ]=prev/next (squares)",
    "g=grid (board)",
    "SPACE=skip/next",
    "ESC=quit",
])


# ===============
# Utility helpers
# ===============

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def draw_text(img: np.ndarray, text: str, org=(8, 24), scale=0.6, color=(255, 255, 255)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def save_labeled(sample_img: np.ndarray, out_root: Path, cls: str, val_split: float,
                 board_meta: Optional[str], source_name: str, row: Optional[int], col: Optional[int]) -> Path:
    split = 'val' if random.random() < val_split else 'train'
    out_dir = out_root / split / cls
    ensure_dir(out_dir)
    ts = int(time.time() * 1000)
    base = f"{cls}_{ts}"
    if row is not None and col is not None:
        base += f"_r{row}c{col}"
    fname = out_dir / f"{base}.png"
    cv2.imwrite(str(fname), sample_img)
    return fname


def write_log(log_csv: Path, rows: List[List[str]]):
    header_needed = not log_csv.exists()
    with log_csv.open('a', newline='') as f:
        writer = csv.writer(f)
        if header_needed:
            writer.writerow(["timestamp", "split", "class", "filepath", "source", "board_meta", "row", "col"])
        for r in rows:
            writer.writerow(r)


# ================
# Squares workflow
# ================

def run_squares_mode(input_dir: Path, output_root: Path, val_split: float):
    # Collect image files
    img_paths = sorted([p for p in input_dir.rglob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}])
    if not img_paths:
        print(f"No images found in {input_dir}")
        return

    idx = 0
    undo_stack: List[Path] = []
    log_csv = output_root / f"label_log_{int(time.time())}.csv"

    cv2.namedWindow('labeler', cv2.WINDOW_NORMAL)

    while 0 <= idx < len(img_paths):
        path = img_paths[idx]
        img = cv2.imread(str(path))
        if img is None:
            idx += 1
            continue
        vis = img.copy()
        draw_text(vis, f"[{idx+1}/{len(img_paths)}] {path.name}")
        draw_text(vis, HELP_TEXT, org=(8, vis.shape[0]-10), scale=0.6)
        cv2.imshow('labeler', vis)
        key = cv2.waitKey(0) & 0xFFFF

        if key == 27:  # ESC
            break
        elif key in (ord('u'), ord('U')):
            if undo_stack and undo_stack[-1].exists():
                try:
                    undo_stack[-1].unlink()
                except Exception:
                    pass
                undo_stack.pop()
        elif key == ord('['):
            idx = max(0, idx - 1)
        elif key == ord(']'):
            idx = min(len(img_paths) - 1, idx + 1)
        elif key in HOTKEYS:
            cls = HOTKEYS[key]
            saved = save_labeled(img, output_root, cls, val_split, board_meta=None,
                                 source_name=str(path), row=None, col=None)
            undo_stack.append(saved)
            write_log(
                log_csv,
                [[int(time.time()), saved.parts[-3], cls, str(saved), str(path), "", "", ""]]
            )
            idx += 1
        else:
            # ignore unknown key, stay on same image
            pass

    cv2.destroyAllWindows()


# ==============
# Board workflow
# ==============

def split_board_equal(img: np.ndarray, rows: int = 8, cols: int = 8) -> List[Tuple[int, int, np.ndarray]]:
    h, w = img.shape[:2]
    cell_h = h // rows
    cell_w = w // cols
    squares = []
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * cell_h, (r + 1) * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w
            crop = img[y0:y1, x0:x1]
            squares.append((r, c, crop))
    return squares


def draw_grid(vis: np.ndarray, rows: int = 8, cols: int = 8):
    h, w = vis.shape[:2]
    for r in range(1, rows):
        y = r * (h // rows)
        cv2.line(vis, (0, y), (w, y), (0, 255, 255), 1)
    for c in range(1, cols):
        x = c * (w // cols)
        cv2.line(vis, (x, 0), (x, h), (0, 255, 255), 1)


def run_board_mode(board_img_path: Path, output_root: Path, val_split: float, rows: int, cols: int):
    img = cv2.imread(str(board_img_path))
    if img is None:
        print(f"Failed to read board image: {board_img_path}")
        return

    squares = split_board_equal(img, rows=rows, cols=cols)
    total = len(squares)
    idx = 0
    show_grid = True
    undo_stack: List[Path] = []
    log_csv = output_root / f"label_log_{int(time.time())}.csv"

    cv2.namedWindow('board', cv2.WINDOW_NORMAL)
    cv2.namedWindow('square', cv2.WINDOW_AUTOSIZE)

    while 0 <= idx < total:
        r, c, crop = squares[idx]
        # Board view with highlight
        vis = img.copy()
        if show_grid:
            draw_grid(vis, rows, cols)
        # highlight current cell
        h, w = img.shape[:2]
        cell_h, cell_w = h // rows, w // cols
        x0, y0 = c * cell_w, r * cell_h
        x1, y1 = x0 + cell_w, y0 + cell_h
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)
        draw_text(vis, f"Square r{r} c{c}  [{idx+1}/{total}]  {board_img_path.name}")
        draw_text(vis, HELP_TEXT, org=(8, vis.shape[0]-10), scale=0.6)

        cv2.imshow('board', vis)
        cv2.imshow('square', crop)
        key = cv2.waitKey(0) & 0xFFFF

        if key == 27:  # ESC
            break
        elif key == ord('g'):
            show_grid = not show_grid
        elif key in (ord('u'), ord('U')):
            if undo_stack and undo_stack[-1].exists():
                try:
                    undo_stack[-1].unlink()
                except Exception:
                    pass
                undo_stack.pop()
        elif key == 32:  # SPACE -> skip/next
            idx = min(total - 1, idx + 1)
        elif key in HOTKEYS:
            cls = HOTKEYS[key]
            saved = save_labeled(crop, output_root, cls, val_split,
                                 board_meta=f"{board_img_path.name}",
                                 source_name=str(board_img_path), row=r, col=c)
            undo_stack.append(saved)
            write_log(
                log_csv,
                [[int(time.time()), saved.parts[-3], cls, str(saved), str(board_img_path), f"r{r}c{c}", r, c]]
            )
            idx = min(total - 1, idx + 1)
        else:
            # ignore unknown key
            pass

    cv2.destroyAllWindows()


# ======
#  Main
# ======

def parse_args():
    ap = argparse.ArgumentParser(description="Labeling GUI for ImageFEN-Converter")
    ap.add_argument('--mode', choices=['squares', 'board'], required=True,
                    help='squares: label pre-cropped squares; board: split a top-down board image into 8x8 and label')
    ap.add_argument('--input_dir', type=str, default=None, help='Directory of square images (for squares mode)')
    ap.add_argument('--board_img', type=str, default=None, help='Path to board image (for board mode)')
    ap.add_argument('--output_root', type=str, default='data', help='Output dataset root (default: data)')
    ap.add_argument('--val-split', type=float, default=0.1, help='Probability to send a sample to val split')
    ap.add_argument('--rows', type=int, default=8, help='Rows for board split (default 8)')
    ap.add_argument('--cols', type=int, default=8, help='Cols for board split (default 8)')
    return ap.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.output_root)

    if args.mode == 'squares':
        if not args.input_dir:
            print('--input_dir is required for squares mode')
            sys.exit(1)
        run_squares_mode(Path(args.input_dir), out_root, args.val_split)

    elif args.mode == 'board':
        if not args.board_img:
            print('--board_img is required for board mode')
            sys.exit(1)
        run_board_mode(Path(args.board_img), out_root, args.val_split, rows=args.rows, cols=args.cols)


if __name__ == '__main__':
    main()
