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


def warp_board_h(image_bgr: np.ndarray, size: int = 512):
    """
    Warp the detected board and also return the homography and corner points.
    Returns: warped, M (3x3), src(4x2), dst(4x2)
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    quad = find_largest_quad(gray)
    if quad is None:
        raise RuntimeError("Could not detect board contour")
    src = order_corners(quad)
    dst = np.array([[0,0],[size-1,0],[size-1,size-1],[0,size-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image_bgr, M, (size, size))
    return warped, M, src, dst


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


def crop_board_margin(board_bgr: np.ndarray, pct: float) -> np.ndarray:
    """
    Optionally crop a uniform margin around the board before splitting squares.
    Useful for images that include coordinates/labels or borders around the 8x8 area.

    pct: 0.0..0.45 fraction of the min(board height, width) to trim from each side.
    """
    if pct is None or pct <= 0:
        return board_bgr
    h, w = board_bgr.shape[:2]
    s = min(h, w)
    m = int(round(s * float(pct)))
    m = max(0, min(m, (s // 2) - 1))
    if m <= 0:
        return board_bgr
    # keep square by cropping equally on all sides based on min dimension
    y0, y1 = m, h - m
    x0, x1 = m, w - m
    cropped = board_bgr[y0:y1, x0:x1]
    # ensure result is still divisible by 8 to avoid rounding artifacts when splitting
    ch, cw = cropped.shape[:2]
    trim_h = ch % 8
    trim_w = cw % 8
    if trim_h or trim_w:
        y1 = y1 - (trim_h if trim_h else 0)
        x1 = x1 - (trim_w if trim_w else 0)
        cropped = board_bgr[y0:y1, x0:x1]
    return cropped


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
