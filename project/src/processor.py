"""Minimal image processor: grayscale + morphological gradient.

This module provides basic image scanning, loading, grayscale conversion,
saving, and a morphological gradient filter.

Run with: python -m src.main from the project/ directory.
"""

from pathlib import Path
from typing import List, Optional
import cv2
import numpy as np

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}



def scan_images(input_dir: Path) -> List[Path]:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        print(f"[scan] Input directory does not exist: {input_dir}")
        return []
    images: List[Path] = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXT]
    images.sort()
    print(f"[scan] Found {len(images)} image(s) in {input_dir}")
    return images


def load_image(path: Path) -> Optional[np.ndarray]:
    try:
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[load] Failed to read image (corrupted or unsupported): {path}")
            return None
        return img
    except Exception as e:
        print(f"[load] Exception reading {path}: {e}")
        return None


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.copy()
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def morphological_gradient(img: np.ndarray, kernel_size: int = 3, shape: int = cv2.MORPH_RECT) -> np.ndarray:
    """Compute the morphological gradient of `img`.

    Args:
        img: Grayscale or color image. If color, it will be converted to grayscale.
        kernel_size: Size of the square structuring element (odd positive int).
        shape: Structuring element shape (cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS).

    Returns:
        Gradient image as a single-channel uint8 ndarray.
    """
    if img.ndim != 2:
        img = to_grayscale(img)
    ksize = max(1, int(kernel_size))
    if ksize % 2 == 0:
        ksize += 1
    kernel = cv2.getStructuringElement(shape, (ksize, ksize))
    grad = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    return grad


def mirror_horizontal(img: np.ndarray) -> np.ndarray:
    """Return a horizontally mirrored copy of `img`."""
    return cv2.flip(img, 1)


def half_mirror(img: np.ndarray, side: str = "left") -> np.ndarray:
    """Mirror half of the image onto the other half.

    Args:
        img: Input image (grayscale or color).
        side: Which half to mirror from: "left" mirrors the left half onto the right,
              "right" mirrors the right half onto the left.

    Returns:
        The half-mirrored image as the same dtype and shape as `img`.
    """
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    out = img.copy()
    h, w = out.shape[:2]
    mid = w // 2
    if side == "left":
        left = out[:, :mid].copy()
        left_flipped = np.fliplr(left)
        out[:, w - mid :] = left_flipped
    else:
        right = out[:, w - mid :].copy()
        right_flipped = np.fliplr(right)
        out[:, :mid] = right_flipped
    return out




def symmetry_analysis(img: np.ndarray, threshold: float = 0.9) -> dict:
    """Analyze horizontal symmetry of `img`.

    Returns a dict with keys: `score` (0..1) and `symmetric` (bool).
    The score is 1 - normalized mean absolute difference between the image and its
    horizontal mirror. Closer to 1 means more symmetric.
    """
    if img.ndim != 2:
        img = to_grayscale(img)
    mirrored = mirror_horizontal(img)
    # Ensure same dtype
    if img.dtype != mirrored.dtype:
        mirrored = mirrored.astype(img.dtype)
    diff = cv2.absdiff(img, mirrored)
    mean_diff = float(diff.mean())
    score = max(0.0, 1.0 - (mean_diff / 255.0))
    return {"score": score, "symmetric": score >= float(threshold)}


def unsharp_mask(img: np.ndarray, amount: float = 1.0, radius: float = 1.0, threshold: int = 0) -> np.ndarray:
    """Apply Unsharp Masking to sharpen `img`.

    Args:
        img: Input image (grayscale or color).
        amount: Strength of the sharpening (1.0 = add 100% of the mask).
        radius: Gaussian blur sigma used to create the mask.
        threshold: Minimum difference (0..255) to consider for sharpening.

    Returns:
        Sharpened image with the same dtype as input.
    """
    if radius <= 0:
        return img.copy()

    img_f = img.astype(np.float32)
    # Use Gaussian blur with sigma=radius; kernel size 0 lets OpenCV compute it
    blurred = cv2.GaussianBlur(img_f, (0, 0), float(radius))
    mask = img_f - blurred

    if threshold > 0:
        # Compute absolute mask (grayscale) to threshold edges
        gray_mask = to_grayscale(np.clip(mask, 0, 255).astype(np.uint8)).astype(np.int32)
        strong = gray_mask > int(threshold)
        # Broadcast strong mask to image shape if necessary
        if img.ndim == 3:
            strong = np.repeat(strong[:, :, np.newaxis], img.shape[2], axis=2)
        mask = mask * strong

    sharp = img_f + (mask * float(amount))
    sharp = np.clip(sharp, 0, 255).astype(img.dtype)
    return sharp


def thermal_map(img: np.ndarray, colormap: str = "jet", clip_percentiles=(2, 98), overlay: bool = False, alpha: float = 0.6) -> np.ndarray:
    """Create a thermal-style heatmap from `img`.

    This synthesizes a thermal visualization by mapping image intensity to a
    false-color colormap. Optionally overlays the heatmap onto the original image.

    Args:
        img: Input image (grayscale or color).
        colormap: One of 'jet', 'hot', 'inferno', 'magma'.
        clip_percentiles: Tuple (low, high) percentiles to clip intensity for contrast.
        overlay: If True, overlay the heatmap onto the original image with `alpha`.
        alpha: Blend factor when overlaying (0..1).

    Returns:
        BGR uint8 heatmap image.
    """
    # Choose cv2 colormap constant
    cmap_map = {
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "inferno": cv2.COLORMAP_INFERNO,
        "magma": cv2.COLORMAP_MAGMA,
    }
    cmap = cmap_map.get(colormap.lower(), cv2.COLORMAP_JET)

    # Compute intensity
    if img.ndim == 2:
        gray = img.copy()
    else:
        gray = to_grayscale(img)

    # Clip percentiles to enhance contrast
    try:
        lo, hi = clip_percentiles
        lo_v = np.percentile(gray, float(lo))
        hi_v = np.percentile(gray, float(hi))
        if hi_v > lo_v:
            gray_clipped = np.clip(gray, lo_v, hi_v)
        else:
            gray_clipped = gray
    except Exception:
        gray_clipped = gray

    # Normalize to 0..255
    norm = cv2.normalize(gray_clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat = cv2.applyColorMap(norm, cmap)

    if overlay and img.ndim == 3:
        # Resize heat to match original if necessary
        heat_rgb = heat
        overlayed = cv2.addWeighted(img, 1.0 - float(alpha), heat_rgb, float(alpha), 0)
        return overlayed

    return heat


def ar_glove_overlay(img: np.ndarray, glove_img: np.ndarray, landmarks: List[tuple], blend_alpha: float = 0.85) -> np.ndarray:
    """Overlay a glove texture onto a hand using 21 hand landmarks.

    Args:
        img: BGR image to draw onto.
        glove_img: Glove texture image (BGR or BGRA) to warp onto the hand.
        landmarks: Sequence of 21 (x, y) coordinates in pixel coords or normalized (0..1).
        blend_alpha: Maximum alpha blend (0..1) applied where the glove is present.

    Returns:
        Composited BGR image (same dtype as `img`).

    Notes:
        - This uses a perspective transform to map the glove image to the hand's
          minimum-area rectangle, and masks with the convex hull of the landmarks
          so the glove follows the hand shape.
    """
    if not landmarks or len(landmarks) < 4:
        return img.copy()

    h, w = img.shape[:2]
    pts = np.array(landmarks, dtype=np.float32)
    # If landmarks are normalized (<=1.0), scale to image coords
    if pts.max() <= 1.0:
        pts[:, 0] *= float(w)
        pts[:, 1] *= float(h)

    # Ensure shape (N,2)
    pts = pts.reshape((-1, 2))

    # Convex hull of the hand landmarks -> mask region
    hull = cv2.convexHull(pts.astype(np.int32))
    hull_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(hull_mask, hull, 255)

    # Compute a rotated bounding rectangle for a stable quad to warp into
    rect = cv2.minAreaRect(pts.astype(np.float32))
    box = cv2.boxPoints(rect)  # 4x2 float

    # Prepare glove source corners
    gh, gw = glove_img.shape[:2]
    src = np.array([[0.0, 0.0], [gw - 1.0, 0.0], [gw - 1.0, gh - 1.0], [0.0, gh - 1.0]], dtype=np.float32)
    dst = box.astype(np.float32)

    # Compute perspective transform from glove -> hand rect and warp to full image
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(glove_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

    # If glove image has alpha channel, warp it too and use it as mask
    if glove_img.ndim == 3 and glove_img.shape[2] == 4:
        glove_bgr = warped[:, :, :3].astype(np.float32)
        # Warp the source alpha channel separately for better alpha precision
        warped_alpha = cv2.warpPerspective(glove_img[:, :, 3], M, (w, h), flags=cv2.INTER_LINEAR)
        glove_alpha = (warped_alpha.astype(np.float32) / 255.0) * (hull_mask.astype(np.float32) / 255.0)
    else:
        # If glove image is grayscale (2D) or BGR, use warped result and hull mask
        if warped.ndim == 2:
            glove_bgr = cv2.cvtColor(warped.astype(np.uint8), cv2.COLOR_GRAY2BGR).astype(np.float32)
        else:
            glove_bgr = warped.astype(np.float32)
        glove_alpha = (hull_mask.astype(np.float32) / 255.0)

    # Smooth and feather the alpha mask for nicer blending
    try:
        alpha_blur = cv2.GaussianBlur(glove_alpha, (31, 31), 0)
    except Exception:
        alpha_blur = glove_alpha

    alpha_final = np.clip(alpha_blur * float(blend_alpha), 0.0, 1.0)

    # Composite
    out = img.astype(np.float32).copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR).astype(np.float32)

    # Ensure glove_bgr is float32 and same channels
    if glove_bgr.ndim == 2:
        glove_bgr = cv2.cvtColor(glove_bgr.astype(np.uint8), cv2.COLOR_GRAY2BGR).astype(np.float32)
    else:
        glove_bgr = glove_bgr.astype(np.float32)

    alpha_stack = np.dstack([alpha_final] * 3)
    comp = (glove_bgr * alpha_stack) + (out * (1.0 - alpha_stack))
    comp = np.clip(comp, 0, 255).astype(img.dtype)
    return comp


def ar_glove_with_mediapipe(img: np.ndarray, glove_img: np.ndarray, blend_alpha: float = 0.85, max_num_hands: int = 1) -> np.ndarray:
    """Detect hand landmarks using MediaPipe and apply `ar_glove_overlay`.

    If MediaPipe isn't available a ValueError is raised.
    """
    try:
        import mediapipe as mp
    except Exception:
        raise ValueError("mediapipe is required for ar_glove_with_mediapipe but is not installed")

    mp_hands = mp.solutions.hands
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=max_num_hands) as hands:
        res = hands.process(rgb)
        if not res or not getattr(res, 'multi_hand_landmarks', None):
            return img.copy()
        # Use first detected hand
        hand_landmarks = res.multi_hand_landmarks[0]
        h, w = img.shape[:2]
        pts = []
        for lm in hand_landmarks.landmark:
            x = lm.x * float(w)
            y = lm.y * float(h)
            pts.append((x, y))
        return ar_glove_overlay(img, glove_img, pts, blend_alpha=blend_alpha)


def _rect_iou(a, b) -> float:
    """Compute IoU between two rectangles."""
    # a, b = (x,y,w,h)
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    area_a = a[2] * a[3]
    area_b = b[2] * b[3]
    union = area_a + area_b - inter
    return float(inter) / float(union) if union > 0 else 0.0


def _nms(rects, iou_thresh=0.3):
    """Non-Maximum Suppression for rectangles."""
    if not rects:
        return []
    rects = list(rects)
    picked = []
    areas = [r[2] * r[3] for r in rects]
    order = sorted(range(len(rects)), key=lambda i: areas[i], reverse=True)
    while order:
        i = order.pop(0)
        picked.append(rects[i])
        remaining = []
        for j in order:
            if _rect_iou(rects[i], rects[j]) <= iou_thresh:
                remaining.append(j)
        order = remaining
    return picked


def _adjust_gamma(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Adjust gamma of image."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    return cv2.LUT(img, table)




def save_image(output_dir: Path, original_path: Path, image: np.ndarray, filter_name: str) -> Path:
    """
    Save processed image directly into `output_dir` with suffix for the filter.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # only create the folder, no subfolders
    ext = original_path.suffix.lower() or ".jpg"
    out_name = f"{original_path.stem}_{filter_name}{ext}"
    out_path = output_dir / out_name

    success, encoded = cv2.imencode(ext, image)
    if not success:
        success, encoded = cv2.imencode(".jpg", image)
        out_path = out_path.with_suffix(".jpg")

    encoded.tofile(str(out_path))
    print(f"[save] Wrote: {out_path}")
    return out_path


def process_all(input_dir: Path, output_dir: Path) -> None:
    """
    Scan `input_dir`, apply filters to each image, and save results to `output_dir` directly.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = scan_images(input_dir)
    if not images:
        print(f"[process] No images found in {input_dir}. Exiting.")
        return

    for img_path in images:
        print(f"[process] Processing: {img_path.name}")
        img = load_image(img_path)
        if img is None:
            print(f"[process] Skipping unreadable image: {img_path.name}")
            continue

        # Example filters – all saved directly to output_dir
        try:
            grad = morphological_gradient(img, kernel_size=3)
            save_image(output_dir, img_path, grad, "morph_gradient_k3")
        except Exception as e:
            print(f"[process] Morphological gradient failed: {e}")

        try:
            mirrored = mirror_horizontal(img)
            save_image(output_dir, img_path, mirrored, "mirror_h")
        except Exception as e:
            print(f"[process] Mirror horizontal failed: {e}")

        try:
            half_lr = half_mirror(img, side="left")
            save_image(output_dir, img_path, half_lr, "mirror_half_lr")
        except Exception as e:
            print(f"[process] Half mirror LR failed: {e}")

        try:
            half_rl = half_mirror(img, side="right")
            save_image(output_dir, img_path, half_rl, "mirror_half_rl")
        except Exception as e:
            print(f"[process] Half mirror RL failed: {e}")

        try:
            us = unsharp_mask(img)
            save_image(output_dir, img_path, us, "unsharp")
        except Exception as e:
            print(f"[process] Unsharp mask failed: {e}")

        try:
            heat = thermal_map(img)
            save_image(output_dir, img_path, heat, "thermal")
        except Exception as e:
            print(f"[process] Thermal map failed: {e}")

        # Border framing feature removed

    print(f"[process] Completed processing {len(images)} image(s). Outputs in {output_dir}")
