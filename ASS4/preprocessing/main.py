"""
=============================================================================
MODULE 1 — Data Collection & Preprocessing Pipeline
Computer Vision System for Scene Analysis
=============================================================================
Author  : AI Engineer — Computer Vision
Purpose : Load raw images / video frames, normalize them for downstream CV
          tasks (resize, color‑space conversion, noise reduction, lighting).
=============================================================================
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend — no display required
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import glob
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(
    source: Union[str, List[str]],
    max_frames: int = 30,
) -> List[Dict]:
    """
    Load images or extract frames from video files.

    Parameters
    ----------
    source      : path to a single image / video, a directory, or a list of
                  image paths.
    max_frames  : maximum number of frames to extract when source is a video.

    Returns
    -------
    List of dicts, each containing:
        - "name"  : filename / frame label
        - "image" : BGR numpy array (uint8)
    """
    samples: List[Dict] = []

    # ── normalise input to a flat list of path strings ──────────────────────
    if isinstance(source, list):
        paths = source
    elif os.path.isdir(source):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp")
        paths = []
        for ext in exts:
            paths.extend(glob.glob(os.path.join(source, ext)))
        paths.sort()
    else:
        paths = [source]

    for path in paths:
        ext = Path(path).suffix.lower()

        # ── video file → sample frames evenly ───────────────────────────────
        if ext in (".mp4", ".avi", ".mov", ".mkv", ".wmv"):
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f"  [WARNING] Cannot open video: {path}")
                continue
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            step  = max(1, total // max_frames)
            frame_idx = 0
            extracted = 0
            while extracted < max_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                samples.append({
                    "name":  f"{Path(path).stem}_frame{frame_idx:05d}",
                    "image": frame.copy(),
                })
                frame_idx += step
                extracted += 1
            cap.release()
            print(f"  [LOAD] Video '{Path(path).name}' → {extracted} frames extracted")

        # ── static image ────────────────────────────────────────────────────
        else:
            img = cv2.imread(path)
            if img is None:
                print(f"  [WARNING] Cannot read image: {path}")
                continue
            samples.append({"name": Path(path).stem, "image": img})
            print(f"  [LOAD] Image '{Path(path).name}' — shape {img.shape}")

    if not samples:
        raise FileNotFoundError(f"No valid images/videos found in: {source}")

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def resize_and_crop(
    image: np.ndarray,
    target_size: Tuple[int, int] = (512, 512),
    scale_factor: float = 1.0,
    crop: Optional[str] = None,          # None | "center" | "roi"
    roi: Optional[Tuple[int, int, int, int]] = None,  # x, y, w, h
) -> np.ndarray:
    """
    Resize an image to *target_size* after applying *scale_factor*, then
    optionally crop.

    Parameters
    ----------
    image       : input BGR image.
    target_size : (width, height) of the final output.
    scale_factor: intermediate scale applied before resize (>1 zooms in).
    crop        : "center" for square centre‑crop; "roi" uses *roi* tuple.
    roi         : (x, y, w, h) region‑of‑interest for "roi" crop mode.
    """
    h, w = image.shape[:2]
    print(f"    Resize  | original {w}×{h}", end="")

    # ── optional intermediate scaling ───────────────────────────────────────
    if scale_factor != 1.0:
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        h, w = image.shape[:2]

    # ── optional cropping before final resize ───────────────────────────────
    if crop == "center":
        side = min(h, w)
        y0   = (h - side) // 2
        x0   = (w - side) // 2
        image = image[y0 : y0 + side, x0 : x0 + side]
        print(f"  →  center‑crop {side}×{side}", end="")

    elif crop == "roi" and roi is not None:
        x, y, rw, rh = roi
        # clamp to valid bounds
        x, y = max(0, x), max(0, y)
        rw   = min(rw, w - x)
        rh   = min(rh, h - y)
        image = image[y : y + rh, x : x + rw]
        print(f"  →  ROI ({x},{y},{rw},{rh})", end="")

    # ── final resize ────────────────────────────────────────────────────────
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    print(f"  →  final {target_size[0]}×{target_size[1]}")
    return image


def convert_color_spaces(image: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Convert a BGR image to Grayscale and HSV.

    Returns a dict with keys "bgr", "gray", "hsv".
    """
    return {
        "bgr":  image,
        "gray": cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        "hsv":  cv2.cvtColor(image, cv2.COLOR_BGR2HSV),
    }


def reduce_noise(
    image: np.ndarray,
    method: str = "gaussian",
    ksize: int = 5,
) -> np.ndarray:
    """
    Apply noise‑reduction filter.

    Parameters
    ----------
    method : "gaussian"  — weighted average, good for Gaussian noise.
             "median"    — non‑linear, ideal for salt‑and‑pepper noise.
             "bilateral" — edge‑preserving smoothing.
    ksize  : kernel size (must be odd and positive).
    """
    ksize = ksize if ksize % 2 == 1 else ksize + 1   # enforce odd

    if method == "gaussian":
        # sigma = 0 → OpenCV derives it from ksize
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    elif method == "median":
        return cv2.medianBlur(image, ksize)

    elif method == "bilateral":
        # d=9, sigmaColor/Space=75 — good default for portrait/scene images
        return cv2.bilateralFilter(image, 9, 75, 75)

    else:
        raise ValueError(f"Unknown noise‑reduction method: '{method}'")


def adjust_lighting(
    image: np.ndarray,
    method: str = "clahe",
    clip_limit: float = 2.0,
    tile_grid: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Improve local contrast / brightness.

    Works on the L‑channel of LAB colour space so hue is preserved.

    Parameters
    ----------
    method     : "histeq"  — global Histogram Equalisation.
                 "clahe"   — Contrast‑Limited Adaptive HE (recommended).
    clip_limit : CLAHE clip limit; higher → more contrast, more noise.
    tile_grid  : CLAHE tile grid size.
    """
    # Convert BGR → LAB; enhance L channel; convert back
    is_color = (image.ndim == 3)

    if is_color:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
    else:
        l_ch = image

    if method == "histeq":
        l_enhanced = cv2.equalizeHist(l_ch)

    elif method == "clahe":
        clahe      = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        l_enhanced = clahe.apply(l_ch)

    else:
        raise ValueError(f"Unknown lighting method: '{method}'")

    if is_color:
        lab_enhanced = cv2.merge([l_enhanced, a_ch, b_ch])
        return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return l_enhanced


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FULL PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(
    image: np.ndarray,
    name: str = "image",
    target_size: Tuple[int, int] = (512, 512),
    scale_factor: float = 1.0,
    crop: Optional[str] = None,
    roi: Optional[Tuple[int, int, int, int]] = None,
    noise_method: str = "gaussian",
    noise_ksize: int = 5,
    lighting_method: str = "clahe",
    clip_limit: float = 2.0,
) -> Dict:
    """
    Full preprocessing pipeline for one image.

    Returns a dict with keys:
        original, resized, colors, denoised, lit_bgr, lit_gray,
        steps_log
    """
    steps_log: List[str] = []
    print(f"\n  ─── Processing: '{name}' ───")

    # ── Step 1 · Resize / Crop ───────────────────────────────────────────────
    resized = resize_and_crop(
        image, target_size=target_size,
        scale_factor=scale_factor, crop=crop, roi=roi,
    )
    steps_log.append(f"resize→{target_size[0]}×{target_size[1]}")

    # ── Step 2 · Colour‑space conversion ────────────────────────────────────
    colors = convert_color_spaces(resized)
    steps_log.append("color_convert(bgr→gray+hsv)")

    # ── Step 3 · Noise reduction (on BGR) ───────────────────────────────────
    denoised = reduce_noise(resized, method=noise_method, ksize=noise_ksize)
    steps_log.append(f"noise_reduce({noise_method},k={noise_ksize})")
    print(f"    Denoise | method={noise_method}, ksize={noise_ksize}")

    # ── Step 4 · Lighting adjustment ────────────────────────────────────────
    lit_bgr  = adjust_lighting(denoised,  method=lighting_method, clip_limit=clip_limit)
    # Also produce a grayscale version with enhanced lighting
    gray_den = reduce_noise(colors["gray"], method=noise_method, ksize=noise_ksize)
    lit_gray = adjust_lighting(gray_den, method=lighting_method, clip_limit=clip_limit)
    steps_log.append(f"lighting({lighting_method},clip={clip_limit})")
    print(f"    Lighting| method={lighting_method}, clip_limit={clip_limit}")

    print(f"    Steps   | {' → '.join(steps_log)}")

    return {
        "name":     name,
        "original": image,
        "resized":  resized,
        "colors":   colors,          # dict: bgr / gray / hsv
        "denoised": denoised,
        "lit_bgr":  lit_bgr,
        "lit_gray": lit_gray,
        "steps_log": steps_log,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def _bgr2rgb(img: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 → RGB uint8 for matplotlib."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def visualize_results(
    results: List[Dict],
    output_path: str = "preprocess.png",
    dpi: int = 150,
) -> None:
    """
    Produce a multi‑panel comparison figure for all processed images.

    Columns per image:
        Original | Gray | Denoised | Lighting (BGR) | Lighting (Gray)
    """
    n = len(results)
    cols = 5
    rows = n

    fig_w = cols * 3.0
    fig_h = rows * 3.2

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#1a1a2e")
    fig.suptitle(
        "Preprocessing Pipeline — Original vs Processed",
        fontsize=14, fontweight="bold", color="white", y=1.01,
    )

    col_titles = [
        "Original",
        "Grayscale",
        "Denoised\n(Gaussian)",
        "Lighting\n(CLAHE · Color)",
        "Lighting\n(CLAHE · Gray)",
    ]

    for row_idx, res in enumerate(results):
        panels = [
            ("Original",        _bgr2rgb(res["original"]),  "viridis", False),
            ("Gray",            res["colors"]["gray"],       "gray",    True),
            ("Denoised",        _bgr2rgb(res["denoised"]),  "viridis", False),
            ("Lit BGR",         _bgr2rgb(res["lit_bgr"]),   "viridis", False),
            ("Lit Gray",        res["lit_gray"],             "gray",    True),
        ]

        for col_idx, (_, panel_img, cmap, is_gray) in enumerate(panels):
            ax = fig.add_subplot(rows, cols, row_idx * cols + col_idx + 1)

            if is_gray:
                ax.imshow(panel_img, cmap=cmap, aspect="auto")
            else:
                ax.imshow(panel_img, aspect="auto")

            # Column header on first row
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=8,
                             color="white", pad=4, fontweight="bold")

            # Row label on first column
            if col_idx == 0:
                ax.set_ylabel(
                    res["name"], fontsize=7, color="#aad4f5",
                    labelpad=4, rotation=90, va="center",
                )

            ax.axis("off")

    plt.tight_layout(pad=0.8)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  [VIZ] Saved comparison figure → '{output_path}'")


def visualize_pipeline_steps(
    result: Dict,
    output_path: str = "pipeline_steps.png",
    dpi: int = 150,
) -> None:
    """
    Detailed step‑by‑step breakdown for a single image (including HSV channels).
    """
    name = result["name"]
    hsv  = result["colors"]["hsv"]
    h_ch, s_ch, v_ch = cv2.split(hsv)

    panels = [
        ("Original (BGR)",         _bgr2rgb(result["original"]), "viridis", False),
        ("After Resize",           _bgr2rgb(result["resized"]),  "viridis", False),
        ("Grayscale",              result["colors"]["gray"],      "gray",    True),
        ("HSV — H channel",        h_ch,                         "hsv",     True),
        ("HSV — S channel",        s_ch,                         "gray",    True),
        ("HSV — V channel",        v_ch,                         "gray",    True),
        ("After Denoising",        _bgr2rgb(result["denoised"]), "viridis", False),
        ("After CLAHE (Color)",    _bgr2rgb(result["lit_bgr"]),  "viridis", False),
        ("After CLAHE (Gray)",     result["lit_gray"],            "gray",    True),
    ]

    cols = 3
    rows = (len(panels) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.0),
                             facecolor="#1a1a2e")
    fig.suptitle(f"Step‑by‑Step Pipeline — '{name}'",
                 fontsize=12, fontweight="bold", color="white")
    axes = axes.flatten()

    for i, (title, img, cmap, is_gray) in enumerate(panels):
        ax = axes[i]
        if is_gray:
            ax.imshow(img, cmap=cmap, aspect="auto")
        else:
            ax.imshow(img, aspect="auto")
        ax.set_title(title, fontsize=8, color="white", pad=3)
        ax.axis("off")

    for j in range(len(panels), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(pad=0.6)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [VIZ] Detailed steps figure → '{output_path}'")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MAIN — Example usage with 3 images
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 68)
    print("  MODULE 1 — Data Collection & Preprocessing Pipeline")
    print("=" * 68)

    # ── Load images ──────────────────────────────────────────────────────────
    IMAGE_DIR = "../input"
    print(f"\n[STEP 1] Loading data from '{IMAGE_DIR}' …")
    samples = load_data(IMAGE_DIR)
    print(f"  Loaded {len(samples)} image(s).\n")

    # ── Preprocessing configuration ──────────────────────────────────────────
    CONFIG = dict(
        target_size    = (512, 512),
        scale_factor   = 1.0,
        crop           = "center",    # center‑crop before final resize
        noise_method   = "gaussian",
        noise_ksize    = 5,
        lighting_method= "clahe",
        clip_limit     = 2.5,
    )

    print("[STEP 2] Preprocessing images …")
    results = []
    for sample in samples:
        res = preprocess_image(sample["image"], name=sample["name"], **CONFIG)
        results.append(res)
    
    OUTPUT_DIR = "./output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Visualisation ─────────────────────────────────────────────────────────
    print("\n[STEP 3] Generating visualisations …")
    visualize_results(results, output_path=os.path.join(OUTPUT_DIR, "preprocess.png"), dpi=150)

    # Detailed step breakdown for each image
    for res in results:
        visualize_pipeline_steps(
            res,
            output_path=os.path.join(OUTPUT_DIR, f"pipeline_{res['name']}.png"),
            dpi=130,
        )

    # ── Debug summary ─────────────────────────────────────────────────────────
    print("\n[DEBUG SUMMARY]")
    print(f"  {'Name':<30}  {'Original':>16}  {'Processed':>16}  Steps")
    print("  " + "─" * 80)
    for res in results:
        oh, ow = res["original"].shape[:2]
        ph, pw = res["lit_bgr"].shape[:2]
        print(f"  {res['name']:<30}  {ow}×{oh:>6}     {pw}×{ph:>6}   "
              f"  {' → '.join(res['steps_log'])}")

    print("\n  [DONE] Module 1 preprocessing complete.\n")