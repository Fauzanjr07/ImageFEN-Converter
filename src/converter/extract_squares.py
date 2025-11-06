import argparse
import os

import cv2

from .boards import warp_board, maybe_autoflip, split_squares, save_grid_debug, save_squares_debug, crop_board_margin


def parse_args():
    p = argparse.ArgumentParser(description="Extract 64 square crops from a board image")
    p.add_argument("--image", required=True)
    p.add_argument("--out-dir", required=False, help="Directory to save squares (default next to image)")
    p.add_argument("--flip", action="store_true", help="Force 180-degree rotation")
    p.add_argument("--board-size", type=int, default=0, help="Warped board resolution (pixels). 0=auto (8 * 224)")
    p.add_argument("--skip-warp", action="store_true", help="Skip perspective warp (use full image)")
    p.add_argument("--no-autoflip", action="store_true", help="Disable dark-square heuristic 180° auto-flip")
    p.add_argument("--crop-percent", type=float, default=0.0, help="Crop a uniform margin (0.0-0.45)")
    p.add_argument("--save-grid", action="store_true", help="Save grid overlay image")
    return p.parse_args()


def main():
    args = parse_args()
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    # Use a higher warp size so squares keep detail before resizing in later steps
    default_img_size = 224
    warp_size = (8 * default_img_size) if (args.board_size is None or int(args.board_size) <= 0) else int(args.board_size)
    board = img.copy() if args.skip_warp else warp_board(img, size=warp_size)
    if not args.no_autoflip:
        board = maybe_autoflip(board)
    if args.flip:
        board = cv2.rotate(board, cv2.ROTATE_180)

    board = crop_board_margin(board, args.crop_percent)

    squares = split_squares(board)
    base = os.path.splitext(args.image)[0]
    out_dir = args.out_dir or (base + "_squares")
    save_squares_debug(squares, out_dir)

    if args.save_grid:
        save_grid_debug(board, base + "_grid.png")

    print(f"Saved 64 squares to: {out_dir}")


if __name__ == "__main__":
    main()
