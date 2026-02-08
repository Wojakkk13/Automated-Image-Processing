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

        # Face blur feature removed

    print(f"[process] Completed processing {len(images)} image(s). Outputs in {output_dir}")
