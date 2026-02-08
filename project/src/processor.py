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


def detect_faces(img: np.ndarray, min_size=(8, 8)) -> List[tuple]:
    """Detect faces robustly using multiple preprocessors and fallbacks.

    Returns a list of (x, y, w, h) rects.
    Strategy:
      - Try MediaPipe FaceDetection if available.
      - Otherwise run Haar cascade on original, CLAHE-enhanced, and gamma-corrected images.
      - Merge detections with simple NMS.
    """
    faces = []
    h, w = img.shape[:2]

    # Try MediaPipe first (if installed) and run it on upscaled variants
    try:
        import mediapipe as mp
        mp_fd = mp.solutions.face_detection
        rgb_orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        scales = [1.0, 1.5, 2.0]
        for scale in scales:
            try:
                if scale != 1.0:
                    sw, sh = int(w * scale), int(h * scale)
                    rgb = cv2.resize(rgb_orig, (sw, sh), interpolation=cv2.INTER_LINEAR)
                else:
                    rgb = rgb_orig
                with mp_fd.FaceDetection(model_selection=1) as fd:
                    res = fd.process(rgb)
                    if res and getattr(res, 'detections', None):
                        for d in res.detections:
                            bbox = d.location_data.relative_bounding_box
                            # relative coords map directly to original image size
                            x = int(bbox.xmin * w)
                            y = int(bbox.ymin * h)
                            ww = int(bbox.width * w)
                            hh = int(bbox.height * h)
                            faces.append((max(0, x), max(0, y), ww, hh))
                if faces:
                    return _nms(faces)
            except Exception:
                continue
    except Exception:
        # Mediapipe not available — proceed to other methods
        pass

    # Prepare Haar cascade
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
    except Exception:
        face_cascade = None

    gray = to_grayscale(img)

    # CLAHE to boost local contrast (helps underexposed/dark skin)
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_clahe = clahe.apply(gray)
    except Exception:
        gray_clahe = gray

    # Gamma correction to brighten shadows
    try:
        gamma_img = _adjust_gamma(img, 1.5)
        gamma_gray = to_grayscale(gamma_img)
    except Exception:
        gamma_gray = gray

    candidates = []

    if face_cascade is not None:
        try:
            # More permissive parameters for small/low-contrast faces
            params = dict(scaleFactor=1.05, minNeighbors=2, minSize=min_size)
            for g in (gray, gray_clahe, gamma_gray):
                dets = face_cascade.detectMultiScale(g, **params)
                for (x, y, ww, hh) in dets:
                    candidates.append((int(x), int(y), int(ww), int(hh)))
        except Exception:
            pass

    # Fallback: simple skin-color-ish segmentation to propose regions
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_channel = hsv[:, :, 0]
        s_channel = hsv[:, :, 1]
        v_channel = hsv[:, :, 2]
        # Heuristic: moderate saturation & value may indicate skin -- coarse
        skin_mask = (s_channel > 15) & (v_channel > 30)
        # find contours on mask
        skin_mask_u8 = (skin_mask.astype('uint8') * 255)
        contours, _ = cv2.findContours(skin_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, ww, hh = cv2.boundingRect(c)
            area = int(ww) * int(hh)
            area_frac = float(area) / float(w * h) if (w * h) > 0 else 0.0
            # Reject tiny regions or overly large regions (likely false positives).
            if ww < min_size[0] or hh < min_size[1]:
                continue
            # Skip very large skin-like regions (e.g. covering much of the image)
            if area_frac > 0.12:
                print(f"[detect] Skipping large skin candidate (area_frac={area_frac:.2f})")
                continue
            candidates.append((x, y, ww, hh))
    except Exception:
        pass

    # Always try a quick Haar on resized image as last resort
    if face_cascade is not None and not candidates:
        try:
            small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
            dets = face_cascade.detectMultiScale(small, scaleFactor=1.03, minNeighbors=2, minSize=(int(min_size[0] / 2), int(min_size[1] / 2)))
            for (x, y, ww, hh) in dets:
                candidates.append((int(x * 2), int(y * 2), int(ww * 2), int(hh * 2)))
        except Exception:
            pass

    # Merge candidates with simple NMS
    merged = _nms(candidates, iou_thresh=0.25)
    # Clip boxes to image
    out = []
    for (x, y, ww, hh) in merged:
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(w, x + ww)
        y1 = min(h, y + hh)
        if x1 > x0 and y1 > y0:
            out.append((x0, y0, x1 - x0, y1 - y0))

    return out


def estimate_face_distance(face_h_px: int, image_h_px: int, real_face_height_m: float = 0.16, focal_px: Optional[float] = None) -> float:
    """Estimate approximate distance (meters) to a face given its pixel height.

    Uses the pinhole camera model: distance = (real_height * focal_length_px) / pixel_height.
    If `focal_px` is not provided a heuristic focal length in pixels is used.
    """
    try:
        if face_h_px <= 0:
            return float('inf')
        if focal_px is None:
            # Heuristic focal length (in pixels) proportional to image height
            focal_px = float(image_h_px) * 1.2
        dist_m = (float(real_face_height_m) * float(focal_px)) / float(face_h_px)
        return float(dist_m)
    except Exception:
        return float('inf')


def _scale_blur_by_distance(base_k: int, face_h_px: int, image_h_px: int, min_scale: float = 0.6, max_scale: float = 2.5) -> int:
    """Return a kernel size scaled by an approximate distance factor.

    Larger face pixel heights (closer faces) increase kernel size; we clamp scale.
    """
    if face_h_px <= 0:
        return base_k
    try:
        ref = float(image_h_px) * 0.25
        scale = float(ref) / float(face_h_px)
        scale = max(min_scale, min(max_scale, scale))
        k = max(3, int(round(base_k * scale)))
        if k % 2 == 0:
            k += 1
        return k
    except Exception:
        return base_k


def mediapipe_face_blur(
    img: np.ndarray,
    blur_factor: float = 1.0,
    seg_threshold: float = 0.5,
    distance_scaling: bool = True,
    real_face_height_m: float = 0.16,
    focal_px: Optional[float] = None,
) -> np.ndarray:
    """Blur faces using MediaPipe selfie-segmentation + face-detection (no Face Mesh).

    This removes Face Mesh usage while keeping robust multi-variant preprocessing,
    aggressive mask dilation and minimum-kernel enforcement to improve recall on
    small/occluded/low-contrast faces.
    """
    try:
        import mediapipe as mp
    except Exception:
        print("[mp] mediapipe not available; falling back to Haar blur")
        faces = detect_faces(img)
        out = img.copy()
        for (x, y, w, h) in faces:
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(out.shape[1], x + w), min(out.shape[0], y + h)
            if x1 <= x0 or y1 <= y0:
                continue
            roi = out[y0:y1, x0:x1]
            base_k = max(3, int(max(w, h) * float(blur_factor)))
            if distance_scaling:
                k = _scale_blur_by_distance(base_k, h, img.shape[0])
            else:
                k = base_k
            if k % 2 == 0:
                k += 1
            # Create a soft elliptical mask inside the detected bbox so only the
            # central face area is blurred (prevents blurring hair/background).
            blurred = cv2.GaussianBlur(roi, (k, k), 0)
            rh, rw = roi.shape[:2]
            mask = np.zeros((rh, rw), dtype=np.uint8)
            # Ellipse centered in ROI; axes tuned to typical face proportions
            axes = (max(1, int(rw * 0.45)), max(1, int(rh * 0.55)))
            center = (int(rw / 2), int(rh / 2))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
            # Smooth mask edges for nicer blend
            try:
                mask = cv2.GaussianBlur(mask, (15, 15), 0)
            except Exception:
                pass

            if roi.ndim == 3:
                alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
                out[y0:y1, x0:x1] = (blurred.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)).astype(out.dtype)
            else:
                alpha = (mask.astype(np.float32) / 255.0)
                out[y0:y1, x0:x1] = (blurred.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)).astype(out.dtype)
        return out

    mp_seg = mp.solutions.selfie_segmentation
    mp_fd = mp.solutions.face_detection
    # Prepare RGB variants to improve detection under low-contrast / dark skin / shadow
    rgb_orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l2 = clahe.apply(l)
        lab2 = cv2.merge((l2, a, b))
        rgb_clahe = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        rgb_clahe = cv2.cvtColor(rgb_clahe, cv2.COLOR_BGR2RGB)
    except Exception:
        rgb_clahe = rgb_orig
    try:
        gamma_img = _adjust_gamma(img, 1.4)
        rgb_gamma = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2RGB)
    except Exception:
        rgb_gamma = rgb_orig

    rgb_variants = [rgb_orig, rgb_clahe, rgb_gamma]

    # FaceDetection params (less conservative for recall)
    fd_conf = 0.5
    fd_max_faces = 10

    out = img.copy()
    faces_all = []

    # Try variants to improve recall
    for rgb in rgb_variants:
        with mp_seg.SelfieSegmentation(model_selection=1) as seg, mp_fd.FaceDetection(model_selection=1) as fd:
            seg_res = seg.process(rgb)
            fd_res = fd.process(rgb)

            seg_mask = None
            if seg_res is not None and getattr(seg_res, 'segmentation_mask', None) is not None:
                seg_mask = seg_res.segmentation_mask
                seg_mask = cv2.resize(seg_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

            if fd_res is not None and getattr(fd_res, 'detections', None):
                for d in fd_res.detections:
                    conf = getattr(d, 'score', None)
                    # score may be list-like; accept if any above threshold
                    ok = False
                    try:
                        if isinstance(conf, (list, tuple)):
                            ok = any([s >= fd_conf for s in conf])
                        else:
                            ok = (conf is None) or (float(conf) >= fd_conf)
                    except Exception:
                        ok = True
                    if not ok:
                        continue
                    bbox = d.location_data.relative_bounding_box
                    x = int(bbox.xmin * img.shape[1])
                    y = int(bbox.ymin * img.shape[0])
                    w = int(bbox.width * img.shape[1])
                    h = int(bbox.height * img.shape[0])
                    faces_all.append((x, y, w, h))

    # If none found from MediaPipe detections, also try `detect_faces` on the
    # original and enhanced variants so small faces are captured by Haar/CLAHE/gamma.
    if not faces_all:
        try:
            # try on original
            faces_all = detect_faces(img, min_size=(8, 8))
            if not faces_all:
                # try CLAHE/gamma variants as well
                try:
                    clahe_bgr = cv2.cvtColor(rgb_clahe, cv2.COLOR_RGB2BGR)
                    faces_all = detect_faces(clahe_bgr, min_size=(8, 8))
                except Exception:
                    pass
            if not faces_all:
                try:
                    gamma_bgr = cv2.cvtColor(rgb_gamma, cv2.COLOR_RGB2BGR)
                    faces_all = detect_faces(gamma_bgr, min_size=(8, 8))
                except Exception:
                    pass
        except Exception:
            faces_all = []

    # Merge and NMS
    faces_merged = _nms(faces_all, iou_thresh=0.25)

    for (x, y, w, h) in faces_merged:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(out.shape[1], x + w), min(out.shape[0], y + h)
        if x1 <= x0 or y1 <= y0:
            continue

        roi = out[y0:y1, x0:x1]
        base_k = max(3, int(max(w, h) * float(blur_factor)))
        if distance_scaling:
            k = _scale_blur_by_distance(base_k, h, img.shape[0])
        else:
            k = base_k
        if k % 2 == 0:
            k += 1
        blurred = cv2.GaussianBlur(roi, (k, k), 0)
        # Build an elliptical face mask inside the bbox (safer default than
        # using the full box). If MediaPipe segmentation (`seg_mask`) exists,
        # intersect it with the ellipse to get a tighter face-only mask.
        rh, rw = roi.shape[:2]
        ellipse_mask = np.zeros((rh, rw), dtype=np.uint8)
        axes = (max(1, int(rw * 0.45)), max(1, int(rh * 0.55)))
        center = (int(rw / 2), int(rh / 2))
        cv2.ellipse(ellipse_mask, center, axes, 0, 0, 360, 255, -1)

        if 'seg_mask' in locals() and seg_mask is not None:
            roi_mask = (seg_mask[y0:y1, x0:x1] >= float(seg_threshold)).astype(np.uint8) * 255
            if roi_mask.size == 0:
                # fall back to ellipse-only
                combined = ellipse_mask
            else:
                combined = cv2.bitwise_and(ellipse_mask, roi_mask.astype(np.uint8))
                # If combined mask is too small, fall back to ellipse with slight dilation
                frac = float(combined.mean()) / 255.0
                if frac < 0.02:
                    dk = max(3, int(round(max(w, h) * 0.06)))
                    if dk % 2 == 0:
                        dk += 1
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dk, dk))
                    try:
                        combined = cv2.dilate(ellipse_mask, kernel, iterations=2)
                    except Exception:
                        combined = ellipse_mask
                # If segmentation mask covers almost the whole ROI, it's likely wrong; prefer ellipse
                elif frac > 0.60:
                    print(f"[mp_blur] Segmentation mask too large for ROI (frac={frac:.2f}), using ellipse fallback")
                    combined = ellipse_mask
        else:
            combined = ellipse_mask

        # Smooth edges for blending
        try:
            combined = cv2.GaussianBlur(combined, (15, 15), 0)
        except Exception:
            pass

        if roi.ndim == 3:
            alpha = (combined.astype(np.float32) / 255.0)[:, :, None]
            out[y0:y1, x0:x1] = (blurred.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)).astype(out.dtype)
        else:
            alpha = (combined.astype(np.float32) / 255.0)
            out[y0:y1, x0:x1] = (blurred.astype(np.float32) * alpha + roi.astype(np.float32) * (1.0 - alpha)).astype(out.dtype)

    return out



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

        try:
            mpb = mediapipe_face_blur(img)
            save_image(output_dir, img_path, mpb, "mp_faceblur")
        except Exception as e:
            print(f"[process] MediaPipe face blur failed: {e}")

    print(f"[process] Completed processing {len(images)} image(s). Outputs in {output_dir}")
