from __future__ import annotations
import os
from typing import List, Tuple
import cv2
import numpy as np

# --- Board detection and warping ---
# Finds the largest 4-point contour, assumes it's the board, and warps to 512x512.

def find_largest_quad(image_gray):
    blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 180)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_quad = None
    best_area = 0
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > best_area:
                best_area = area
                best_quad = approx
    return best_quad


def order_corners(pts: np.ndarray) -> np.ndarray:
    # pts shape: (4,1,2) or (4,2)
    pts = pts.reshape(-1, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right
    ordered[1] = pts[np.argmin(diff)]  # top-right
    ordered[3] = pts[np.argmax(diff)]  # bottom-left
    return ordered


def warp_board(image_bgr: np.ndarray, size: int = 512) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    quad = find_largest_quad(gray)
    if quad is None:
        raise RuntimeError("Could not detect board contour")
    src = order_corners(quad)
    dst = np.array([[0,0],[size-1,0],[size-1,size-1],[0,size-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image_bgr, M, (size, size))
    return warped


def maybe_autoflip(warped: np.ndarray) -> np.ndarray:
    # Heuristic: top-left should be a dark square. Check brightness of 4 corners squares.
    h, w = warped.shape[:2]
    sq = h // 8
    tl = warped[0:sq, 0:sq]
    tr = warped[0:sq, w - sq:w]
    bl = warped[h - sq:h, 0:sq]
    br = warped[h - sq:h, w - sq:w]
    def mean_brightness(img):
        return img.mean()
    tl_b, tr_b, bl_b, br_b = map(mean_brightness, [tl, tr, bl, br])
    # If top-left is the brightest, assume flipped and rotate 180
    if tl_b > max(tr_b, bl_b, br_b):
        return cv2.rotate(warped, cv2.ROTATE_180)
    return warped


def split_squares(board_bgr: np.ndarray) -> List[np.ndarray]:
    h, w = board_bgr.shape[:2]
    sq_h, sq_w = h // 8, w // 8
    squares = []
    for r in range(8):
        for c in range(8):
            y0, y1 = r * sq_h, (r + 1) * sq_h
            x0, x1 = c * sq_w, (c + 1) * sq_w
            squares.append(board_bgr[y0:y1, x0:x1].copy())
    return squares


def save_grid_debug(board_bgr: np.ndarray, out_path: str) -> None:
    dbg = board_bgr.copy()
    h, w = dbg.shape[:2]
    for i in range(1, 8):
        y = i * h // 8
        x = i * w // 8
        cv2.line(dbg, (0, y), (w, y), (0, 255, 0), 1)
        cv2.line(dbg, (x, 0), (x, h), (0, 255, 0), 1)
    cv2.imwrite(out_path, dbg)


def save_squares_debug(squares: List[np.ndarray], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for i, sq in enumerate(squares):
        r, c = divmod(i, 8)
        cv2.imwrite(os.path.join(out_dir, f"r{r}_c{c}.png"), sq)
