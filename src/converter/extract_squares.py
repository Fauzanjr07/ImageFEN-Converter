import argparse
import os

import cv2

from .boards import warp_board, maybe_autoflip, split_squares, save_grid_debug, save_squares_debug


def parse_args():
    p = argparse.ArgumentParser(description="Extract 64 square crops from a board image")
    p.add_argument("--image", required=True)
    p.add_argument("--out-dir", required=False, help="Directory to save squares (default next to image)")
    p.add_argument("--flip", action="store_true", help="Force 180-degree rotation")
    p.add_argument("--save-grid", action="store_true", help="Save grid overlay image")
    return p.parse_args()


def main():
    args = parse_args()
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    board = warp_board(img)
    board = maybe_autoflip(board)
    if args.flip:
        board = cv2.rotate(board, cv2.ROTATE_180)

    squares = split_squares(board)
    base = os.path.splitext(args.image)[0]
    out_dir = args.out_dir or (base + "_squares")
    save_squares_debug(squares, out_dir)

    if args.save_grid:
        save_grid_debug(board, base + "_grid.png")

    print(f"Saved 64 squares to: {out_dir}")


if __name__ == "__main__":
    main()
