import argparse
import json
import os
from typing import List

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

from .boards import warp_board, maybe_autoflip, split_squares, save_grid_debug, save_squares_debug
from .dataset import load_classes_json, DEFAULT_CLASSES
from .model import build_model, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Predict FEN from a board image")
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--weights", required=True, help="Path to model checkpoint .pt")
    p.add_argument("--classes", required=False, help="Path to classes.json (if absent, default order is used)")
    p.add_argument("--flip", action="store_true", help="Force 180-degree rotation")
    p.add_argument("--save-overlays", action="store_true", help="Save visualizations next to the image")
    p.add_argument("--debug-save-squares", action="store_true", help="Save cropped 64 squares for inspection")
    p.add_argument("--img-size", type=int, default=224)
    return p.parse_args()


def to_tensor(img_bgr: np.ndarray, img_size: int):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    tfm = T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tfm(pil)


def classes_to_fen(classes: List[str]) -> str:
    # Map class name to FEN char
    cmap = {
        "white_pawn": "P", "white_knight": "N", "white_bishop": "B", "white_rook": "R", "white_queen": "Q", "white_king": "K",
        "black_pawn": "p", "black_knight": "n", "black_bishop": "b", "black_rook": "r", "black_queen": "q", "black_king": "k",
        "empty": "1",
    }
    rows = []
    for r in range(8):
        fen_row = ""
        empty_run = 0
        for c in range(8):
            name = classes[r * 8 + c]
            ch = cmap.get(name, "1")
            if ch == "1":
                empty_run += 1
            else:
                if empty_run > 0:
                    fen_row += str(empty_run)
                    empty_run = 0
                fen_row += ch
        if empty_run > 0:
            fen_row += str(empty_run)
        rows.append(fen_row)
    return "/".join(rows)


def main():
    args = parse_args()

    # Resolve class names: prefer explicit --classes, else read from checkpoint, else default
    class_names = None
    if args.classes and os.path.exists(args.classes):
        class_names = load_classes_json(args.classes)
    if class_names is None:
        try:
            state = torch.load(args.weights, map_location="cpu")
            if isinstance(state, dict) and "classes" in state and isinstance(state["classes"], list):
                class_names = state["classes"]
        except Exception:
            class_names = None
    if class_names is None:
        class_names = DEFAULT_CLASSES

    # Load image and detect board
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    board = warp_board(img)
    board = maybe_autoflip(board)
    if args.flip:
        board = cv2.rotate(board, cv2.ROTATE_180)

    if args.save_overlays:
        save_grid_debug(board, os.path.splitext(args.image)[0] + "_grid.png")

    squares = split_squares(board)
    if args.debug_save_squares:
        out_dir = os.path.splitext(args.image)[0] + "_squares"
        save_squares_debug(squares, out_dir)

    # Build model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(num_classes=len(class_names))
    model = load_checkpoint(model, args.weights, device)

    # Classify squares
    tensors = torch.stack([to_tensor(sq, args.img_size) for sq in squares]).to(device)
    with torch.no_grad():
        logits = model(tensors)
        pred = logits.argmax(dim=1).cpu().numpy().tolist()

    pred_names = [class_names[i] for i in pred]
    fen = classes_to_fen(pred_names)

    if args.save_overlays:
        # Draw predicted piece letters on each square
        overlay = board.copy()
        h, w = overlay.shape[:2]
        sq_h, sq_w = h // 8, w // 8
        # Map classes to display letters
        disp_map = {
            "white_pawn": "P", "white_knight": "N", "white_bishop": "B", "white_rook": "R", "white_queen": "Q", "white_king": "K",
            "black_pawn": "p", "black_knight": "n", "black_bishop": "b", "black_rook": "r", "black_queen": "q", "black_king": "k",
            "empty": "",
        }
        for idx, name in enumerate(pred_names):
            r, c = divmod(idx, 8)
            ch = disp_map.get(name, "")
            if not ch:
                continue
            y0, x0 = r * sq_h, c * sq_w
            # Put text roughly centered
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.8
            thickness = 2
            (tw, th), _ = cv2.getTextSize(ch, font, scale, thickness)
            cx, cy = x0 + sq_w // 2, y0 + sq_h // 2
            org = (int(cx - tw / 2), int(cy + th / 2))
            # Outline for contrast
            cv2.putText(overlay, ch, org, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            color = (0, 255, 0) if ch.isupper() else (0, 200, 255)
            cv2.putText(overlay, ch, org, font, scale, color, thickness, cv2.LINE_AA)
        out_path = os.path.splitext(args.image)[0] + "_pred.png"
        cv2.imwrite(out_path, overlay)

    print(fen)


if __name__ == "__main__":
    main()
