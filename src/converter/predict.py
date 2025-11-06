import argparse
import json
import os
import csv
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

from .boards import warp_board, warp_board_h, maybe_autoflip, split_squares, save_grid_debug, save_squares_debug, crop_board_margin
from .dataset import load_classes_json, DEFAULT_CLASSES
from .model import build_model, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Predict FEN from a board image")
    # Single file (back-compat) or directory mode
    p.add_argument("--image", required=False, help="Path to input image (single file mode)")
    p.add_argument("--input", required=False, help="Path to an image file or a directory of images")
    p.add_argument("--output-dir", required=False, help="Directory to write outputs (overlays, FEN, CSV)")
    p.add_argument("--recursive", action="store_true", help="Recurse subfolders when --input is a directory")
    p.add_argument("--weights", required=True, help="Path to model checkpoint .pt")
    p.add_argument("--classes", required=False, help="Path to classes.json (if absent, default order is used)")
    p.add_argument("--force-classes", action="store_true", help="Force using --classes even if checkpoint contains its own class list")
    p.add_argument("--flip", action="store_true", help="Force 180-degree rotation")
    p.add_argument("--skip-warp", action="store_true", help="Skip perspective warp (use full image)")
    p.add_argument("--no-autoflip", action="store_true", help="Disable dark-square heuristic 180° auto-flip")
    p.add_argument("--crop-percent", type=float, default=0.0, help="Crop a uniform margin (0.0-0.45) before splitting squares")
    p.add_argument("--board-size", type=int, default=0, help="Warped board resolution (pixels). 0=auto (8 * img-size)")
    p.add_argument("--overlay-on-original", action="store_true", help="Also save overlay projected back to the original image when warping")
    p.add_argument("--save-overlays", action="store_true", help="Save visualizations next to the image")
    p.add_argument("--debug-save-squares", action="store_true", help="Save cropped 64 squares for inspection")
    p.add_argument("--try-orientations", action="store_true", help="Print FEN for vertical/horizontal/180 flips to help debugging orientation")
    p.add_argument("--topk", type=int, default=0, help="If >0, print top-k predictions per square (e.g., 3)")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--white-lowercase", action="store_true", help="Use lowercase letters for WHITE and uppercase for BLACK in overlays and FEN")
    p.add_argument(
        "--fen-orientation",
        choices=["as-is", "flip-v", "flip-h", "rot-180"],
        default="as-is",
        help="Orientation used to build the final FEN (default: as-is)",
    )
    p.add_argument("--auto-orientation", action="store_true", help="Try all orientations and choose one with heuristic score (does not affect overlays)")
    p.add_argument("--prefer-empty", action="store_true", help="Post-process: if top prob is low but 'empty' prob is reasonable, choose empty")
    p.add_argument("--empty-threshold", type=float, default=0.0, help="If top prob < this and --prefer-empty, consider empty")
    p.add_argument("--empty-min", type=float, default=0.0, help="If 'empty' prob >= this and --prefer-empty, choose empty")
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


def classes_to_fen(classes: List[str], white_lowercase: bool = False) -> str:
    # Map class name to FEN char
    if white_lowercase:
        # Non-standard by request: white = lowercase, black = UPPERCASE
        cmap = {
            "white_pawn": "p", "white_knight": "n", "white_bishop": "b", "white_rook": "r", "white_queen": "q", "white_king": "k",
            "black_pawn": "P", "black_knight": "N", "black_bishop": "B", "black_rook": "R", "black_queen": "Q", "black_king": "K",
            "empty": "1",
        }
    else:
        # Standard FEN: white = UPPERCASE, black = lowercase
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

    # Resolve I/O mode
    input_path = args.input or args.image
    if not input_path:
        raise SystemExit("Please provide --input <file|dir> or --image <file>.")
    is_dir_mode = os.path.isdir(input_path)
    out_root = args.output_dir
    if out_root:
        os.makedirs(out_root, exist_ok=True)

    # Resolve class names: prefer checkpoint -> classes.json -> default (unless --force-classes)
    class_names = None
    ckpt_classes = None
    file_classes = None
    state = None
    if os.path.exists(args.weights):
        try:
            state = torch.load(args.weights, map_location="cpu")
            if isinstance(state, dict) and "classes" in state and isinstance(state["classes"], list):
                ckpt_classes = state["classes"]
        except Exception:
            state = None

    if args.classes and os.path.exists(args.classes):
        try:
            file_classes = load_classes_json(args.classes)
        except Exception:
            file_classes = None

    if args.force_classes and file_classes:
        class_names = file_classes
    else:
        class_names = ckpt_classes or file_classes or DEFAULT_CLASSES
    if state is not None and ckpt_classes is not None and file_classes is not None and ckpt_classes != file_classes and not args.force_classes:
        print("[warn] classes.json differs from checkpoint classes; using checkpoint order. Pass --force-classes to override.")

    # Warp size and model
    warp_size = (8 * args.img_size) if (args.board_size is None or int(args.board_size) <= 0) else int(args.board_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(num_classes=len(class_names))
    model = load_checkpoint(model, args.weights, device)

    # Helpers
    def valid_image(p: str) -> bool:
        ext = os.path.splitext(p)[1].lower()
        return ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def iter_images(path: str):
        if os.path.isdir(path):
            if args.recursive:
                for root, _, files in os.walk(path):
                    for f in files:
                        fp = os.path.join(root, f)
                        if valid_image(fp):
                            yield fp, os.path.relpath(fp, path)
            else:
                for f in os.listdir(path):
                    fp = os.path.join(path, f)
                    if os.path.isfile(fp) and valid_image(fp):
                        yield fp, os.path.basename(fp)
        else:
            yield path, os.path.basename(path)

    def ensure_parent(p: str):
        os.makedirs(os.path.dirname(p), exist_ok=True)

    def base_out_paths(img_path: str, rel_name: str) -> Tuple[str, str]:
        if out_root:
            base_no_ext = os.path.splitext(os.path.join(out_root, os.path.splitext(rel_name)[0]))[0]
        else:
            base_no_ext = os.path.splitext(img_path)[0]
        ensure_parent(base_no_ext + "._")
        return base_no_ext, os.path.dirname(base_no_ext)

    rows_for_csv: List[List[str]] = []
    last_fen: str = ""

    for img_path, rel_name in iter_images(input_path):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[warn] Could not read image: {img_path}")
            continue

        base_no_ext, _ = base_out_paths(img_path, rel_name)

        # Board extraction
        H = None
        orig_img = img.copy()
        if args.skip_warp:
            board = img.copy()
        else:
            if args.overlay_on_original:
                board, H, _, _ = warp_board_h(img, size=warp_size)
            else:
                board = warp_board(img, size=warp_size)

        if not args.no_autoflip:
            board = maybe_autoflip(board)
        if args.flip:
            board = cv2.rotate(board, cv2.ROTATE_180)

        board = crop_board_margin(board, args.crop_percent)

        if args.save_overlays:
            save_grid_debug(board, base_no_ext + "_grid.png")

        squares = split_squares(board)
        if args.debug_save_squares:
            save_squares_debug(squares, base_no_ext + "_squares")

        # Predict
        tensors = torch.stack([to_tensor(sq, args.img_size) for sq in squares]).to(device)
        with torch.no_grad():
            logits = model(tensors)
            pred = logits.argmax(dim=1).cpu().numpy().tolist()
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred_confs = [float(probs[i, cls]) for i, cls in enumerate(pred)]

        if args.prefer_empty and "empty" in class_names:
            empty_idx = class_names.index("empty")
            for i in range(len(pred)):
                top_idx = pred[i]
                top_prob = probs[i, top_idx]
                empty_prob = probs[i, empty_idx]
                if top_prob < float(args.empty_threshold) and empty_prob >= float(args.empty_min):
                    pred[i] = empty_idx

        pred_names = [class_names[i] for i in pred]

        if args.topk and args.topk > 0 and not is_dir_mode:
            k = min(args.topk, len(class_names))
            topk_idx = np.argsort(-probs, axis=1)[:, :k]
            print("Per-square top-{} predictions (idx: class(prob)):".format(k))
            for i in range(len(topk_idx)):
                items = [f"{class_names[j]}({probs[i, j]:.2f})" for j in topk_idx[i]]
                print(f"{i:02d}: " + ", ".join(items))

        rows_grid = [pred_names[r * 8:(r + 1) * 8] for r in range(8)]
        fen_as_is = classes_to_fen([x for row in rows_grid for x in row], white_lowercase=args.white_lowercase)
        fen_options = {
            "as-is": fen_as_is,
            "flip-v": classes_to_fen([x for row in reversed(rows_grid) for x in row], white_lowercase=args.white_lowercase),
            "flip-h": classes_to_fen([x for row in rows_grid for x in reversed(row)], white_lowercase=args.white_lowercase),
            "rot-180": classes_to_fen([x for row in reversed([list(reversed(r)) for r in rows_grid]) for x in row], white_lowercase=args.white_lowercase),
        }

        if args.try_orientations and not is_dir_mode:
            print("FEN (as-is):        ", fen_options["as-is"]) 
            print("FEN (flip vertical):", fen_options["flip-v"]) 
            print("FEN (flip horizontal):", fen_options["flip-h"]) 
            print("FEN (rotate 180):   ", fen_options["rot-180"]) 

        if args.auto_orientation:
            def score(orient):
                idxs = []
                for r in range(8):
                    for c in range(8):
                        if orient == "as-is":
                            rr, cc = r, c
                        elif orient == "flip-v":
                            rr, cc = 7 - r, c
                        elif orient == "flip-h":
                            rr, cc = r, 7 - c
                        else:
                            rr, cc = 7 - r, 7 - c
                        idxs.append(rr * 8 + cc)
                empty_idx = class_names.index("empty") if "empty" in class_names else None
                empty_score = 0.0
                conf_score = 0.0
                for r in range(2, 6):
                    for c in range(8):
                        i_after = r * 8 + c
                        i_before = idxs[i_after]
                        if empty_idx is not None:
                            empty_score += probs[i_before, empty_idx]
                        conf_score += float(probs[i_before].max())
                return (empty_score, conf_score)
            best_orient = max(["as-is", "flip-v", "flip-h", "rot-180"], key=score)
            fen = fen_options[best_orient]
        else:
            fen = fen_options[args.fen_orientation]

        # Save FEN per image
        fen_txt_path = base_no_ext + ".fen.txt"
        with open(fen_txt_path, "w", encoding="utf-8") as ftxt:
            ftxt.write(fen + "\n")
        rows_for_csv.append([img_path, fen])

        # Overlays
        if args.save_overlays:
            overlay = board.copy()
            anno = np.zeros_like(board)
            h, w = overlay.shape[:2]
            sq_h, sq_w = h // 8, w // 8
            if args.white_lowercase:
                disp_map = {"white_pawn":"p","white_knight":"n","white_bishop":"b","white_rook":"r","white_queen":"q","white_king":"k","black_pawn":"P","black_knight":"N","black_bishop":"B","black_rook":"R","black_queen":"Q","black_king":"K","empty":""}
            else:
                disp_map = {"white_pawn":"P","white_knight":"N","white_bishop":"B","white_rook":"R","white_queen":"Q","white_king":"K","black_pawn":"p","black_knight":"n","black_bishop":"b","black_rook":"r","black_queen":"q","black_king":"k","empty":""}
            for idx, name in enumerate(pred_names):
                r, c = divmod(idx, 8)
                ch = disp_map.get(name, "")
                y0, x0 = r * sq_h, c * sq_w
                idx_org = (x0 + 6, y0 + 16)
                if not ch:
                    cv2.putText(overlay, str(idx), idx_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 2, cv2.LINE_AA)
                    cv2.putText(overlay, str(idx), idx_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(anno, str(idx), idx_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(anno, str(idx), idx_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    continue
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 0.8
                thickness = 2
                (tw, th), _ = cv2.getTextSize(ch, font, scale, thickness)
                cx, cy = x0 + sq_w // 2, y0 + sq_h // 2
                org = (int(cx - tw / 2), int(cy + th / 2))
                cv2.putText(overlay, ch, org, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                color = (0, 255, 0) if ch.isupper() else (0, 200, 255)
                cv2.putText(overlay, ch, org, font, scale, color, thickness, cv2.LINE_AA)
                cv2.putText(anno, ch, org, font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                cv2.putText(anno, ch, org, font, scale, color, thickness, cv2.LINE_AA)
                cv2.putText(overlay, str(idx), idx_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 2, cv2.LINE_AA)
                cv2.putText(overlay, str(idx), idx_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(anno, str(idx), idx_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(anno, str(idx), idx_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                conf = pred_confs[idx] if idx < len(pred_confs) else 0.0
                conf_txt = f"{int(round(conf * 100))}%"
                (cw, chh), _ = cv2.getTextSize(conf_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                conf_org = (x0 + sq_w - cw - 6, y0 + sq_h - 6)
                cv2.putText(overlay, conf_txt, conf_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(overlay, conf_txt, conf_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(anno, conf_txt, conf_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(anno, conf_txt, conf_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imwrite(base_no_ext + "_pred.png", overlay)
            if (not args.skip_warp) and args.overlay_on_original and H is not None:
                Hinv = np.linalg.inv(H)
                oh, ow = orig_img.shape[:2]
                anno_back = cv2.warpPerspective(anno, Hinv, (ow, oh))
                mask = cv2.cvtColor(anno_back, cv2.COLOR_BGR2GRAY)
                _, mask_bin = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask_bin)
                bg = cv2.bitwise_and(orig_img, orig_img, mask=mask_inv)
                fg = cv2.bitwise_and(anno_back, anno_back, mask=mask_bin)
                blended = cv2.add(bg, fg)
                cv2.imwrite(base_no_ext + "_pred_orig.png", blended)

        last_fen = fen

    # Write summary CSV if batch mode or output-dir is specified
    if out_root and rows_for_csv:
        csv_path = os.path.join(out_root, "predictions.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["image", "fen"])
            w.writerows(rows_for_csv)

    # Print the FEN in single-file mode for back-compat
    if not is_dir_mode and last_fen:
        print(last_fen)


if __name__ == "__main__":
    main()
