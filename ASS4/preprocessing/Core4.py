import os
import glob
import json
import cv2
import numpy as np
import pandas as pd


# =========================
# PATH CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_DIR = os.path.join(BASE_DIR, "input")
ANALYSIS_DIR = os.path.join(BASE_DIR, "outputs_33")
MASK_DIR = os.path.join(ANALYSIS_DIR, "masks")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_34")
DETECTION_DIR = os.path.join(OUTPUT_DIR, "detections")
SEGMENT_DIR = os.path.join(OUTPUT_DIR, "segmentations")
FINAL_DIR = os.path.join(OUTPUT_DIR, "final")
COMPARE_DIR = os.path.join(OUTPUT_DIR, "comparisons")

os.makedirs(DETECTION_DIR, exist_ok=True)
os.makedirs(SEGMENT_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)
os.makedirs(COMPARE_DIR, exist_ok=True)


# =========================
# COLORS
# =========================
COLORS = {
    "sky": (255, 200, 0),      # BGR
    "water": (255, 0, 0),
    "land": (0, 180, 0),
    "box": (0, 255, 255),
    "horizon": (0, 0, 255),
    "text_bg": (30, 30, 30),
    "text_fg": (255, 255, 255),
}


# =========================
# UTILS
# =========================
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return (mask > 127).astype(np.uint8)


def draw_label(img, text, x, y, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1

    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)

    x = max(0, x)
    y = max(th + 4, y)

    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y), color, -1)
    cv2.putText(img, text, (x + 3, y - 3), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def overlay_single_mask(image, mask, color, alpha=0.30):
    color_layer = np.zeros_like(image)
    color_layer[:, :] = color
    mask_3 = np.stack([mask, mask, mask], axis=-1)

    blended = cv2.addWeighted(image, 1 - alpha, color_layer, alpha, 0)
    out = np.where(mask_3 == 1, blended, image)
    return out.astype(np.uint8)


def overlay_all_masks(image, masks):
    out = image.copy()
    for name in ["sky", "water", "land"]:
        if masks.get(name) is not None:
            out = overlay_single_mask(out, masks[name], COLORS[name], alpha=0.28)
    return out


def draw_detections(image, detections):
    out = image.copy()

    for det in detections:
        x1, y1 = det["x1"], det["y1"]
        x2, y2 = det["x2"], det["y2"]
        cls_name = det["class"]
        conf = det["confidence"]

        cv2.rectangle(out, (x1, y1), (x2, y2), COLORS["box"], 2)
        draw_label(out, f"{cls_name} {conf:.2f}", x1, y1 - 5, COLORS["box"])

    return out


def draw_horizon(image, horizon_y):
    out = image.copy()
    h, w = out.shape[:2]
    horizon_y = max(0, min(horizon_y, h - 1))
    cv2.line(out, (0, horizon_y), (w - 1, horizon_y), COLORS["horizon"], 2)
    draw_label(out, f"horizon = {horizon_y}", 10, max(25, horizon_y - 8), COLORS["horizon"])
    return out


def add_summary_text(image, reflection_score, detections, ratios):
    out = image.copy()

    panel_x1, panel_y1 = 15, 15
    panel_x2, panel_y2 = 330, 145
    cv2.rectangle(out, (panel_x1, panel_y1), (panel_x2, panel_y2), COLORS["text_bg"], -1)

    lines = [
        f"Reflection score: {reflection_score:.4f}",
        f"Detections: {len(detections)}",
        f"Sky ratio: {ratios.get('sky_ratio', 0):.3f}",
        f"Water ratio: {ratios.get('water_ratio', 0):.3f}",
        f"Land ratio: {ratios.get('land_ratio', 0):.3f}",
    ]

    y = 40
    for line in lines:
        cv2.putText(
            out, line, (25, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            COLORS["text_fg"], 1, cv2.LINE_AA
        )
        y += 22

    return out


def resize_same_height(images, target_height=350):
    resized = []
    for img in images:
        h, w = img.shape[:2]
        scale = target_height / h
        new_w = int(w * scale)
        resized.append(cv2.resize(img, (new_w, target_height)))
    return resized


def make_comparison_grid(img1, img2, img3, img4, titles):
    imgs = resize_same_height([img1, img2, img3, img4], target_height=350)

    def add_title(im, title):
        canvas = cv2.copyMakeBorder(
            im, 40, 0, 0, 0,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255)
        )
        cv2.putText(
            canvas, title, (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 0, 0), 2, cv2.LINE_AA
        )
        return canvas

    imgs = [add_title(im, t) for im, t in zip(imgs, titles)]

    top = cv2.hconcat([imgs[0], imgs[1]])
    bottom = cv2.hconcat([imgs[2], imgs[3]])

    max_w = max(top.shape[1], bottom.shape[1])

    if top.shape[1] < max_w:
        top = cv2.copyMakeBorder(
            top, 0, 0, 0, max_w - top.shape[1],
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )

    if bottom.shape[1] < max_w:
        bottom = cv2.copyMakeBorder(
            bottom, 0, 0, 0, max_w - bottom.shape[1],
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )

    grid = cv2.vconcat([top, bottom])
    return grid


def safe_get_row_value(row, col_name, default_value=0.0):
    if col_name in row.columns and len(row) > 0:
        return float(row[col_name].iloc[0])
    return float(default_value)


# =========================
# MAIN
# =========================
json_files = sorted(glob.glob(os.path.join(ANALYSIS_DIR, "*.json")))
scene_csv_path = os.path.join(ANALYSIS_DIR, "scene.csv")

if not os.path.exists(scene_csv_path):
    raise FileNotFoundError(f"Cannot find scene.csv at: {scene_csv_path}")

scene_df = pd.read_csv(scene_csv_path)
summary_rows = []

for json_path in json_files:
    data = load_json(json_path)

    # JSON đang có dạng boat1.jpg.json -> bỏ .json là ra boat1.jpg
    image_name = os.path.basename(json_path).replace(".json", "")
    image_path = os.path.join(INPUT_DIR, image_name)

    image = cv2.imread(image_path)
    if image is None:
        print(f"Skip {image_name}: cannot load input image")
        continue

    image_stem = os.path.splitext(image_name)[0]

    detections = data.get("detections", [])

    # =========================
    # FILTER DETECTIONS
    # =========================
    filtered = []
    h, w = image.shape[:2]
    img_area = h * w

    for det in detections:
        conf = det.get("confidence", 0)
        x1, y1 = det["x1"], det["y1"]
        x2, y2 = det["x2"], det["y2"]

        area = (x2 - x1) * (y2 - y1)
        area_ratio = area / img_area

        # giữ boat thôi
        if det["class"] != "boat":
            continue

        # bỏ confidence thấp
        if conf < 0.45:
            continue
        
        # bỏ bbox nhỏ
        if area_ratio < 0.01:
            continue

        # bỏ bbox quá lớn
        if area_ratio > 0.35:
            continue

        filtered.append(det)

    detections = filtered
    reflection_score = float(data.get("reflection", 0.0))

    row = scene_df[scene_df["image"] == image_name]

    if len(row) > 0:
        if "horizon" in row.columns:
            horizon = int(row["horizon"].iloc[0])
        else:
            horizon = image.shape[0] // 2

        sky_ratio = safe_get_row_value(row, "sky_ratio", 0.0)
        water_ratio = safe_get_row_value(row, "water_ratio", 0.0)

        if "land_ratio" in row.columns:
            land_ratio = safe_get_row_value(row, "land_ratio", 0.0)
        else:
            land_ratio = max(0.0, 1.0 - sky_ratio - water_ratio)
    else:
        horizon = image.shape[0] // 2
        sky_ratio = 0.0
        water_ratio = 0.0
        land_ratio = 0.0

    ratios = {
        "sky_ratio": sky_ratio,
        "water_ratio": water_ratio,
        "land_ratio": land_ratio,
    }
    h, w = image.shape[:2]
    min_horizon = int(h * 0.35)
    max_horizon = int(h * 0.70)
    horizon = max(min_horizon, min(horizon, max_horizon))

    masks = {
        "sky": load_mask(os.path.join(MASK_DIR, f"{image_stem}_sky.png")),
        "water": load_mask(os.path.join(MASK_DIR, f"{image_stem}_water.png")),
        "land": load_mask(os.path.join(MASK_DIR, f"{image_stem}_land.png")),
    }
    if masks["sky"] is not None:
        ys = []
        for x in range(w):
            col = np.where(masks["sky"][:, x] > 0)[0]
            if len(col) > 0:
                ys.append(col[int(len(col) * 0.9)])

        if len(ys) > 0:
            refined_horizon = int(np.median(ys))
            refined_horizon = max(min_horizon, min(refined_horizon, max_horizon))
            horizon = refined_horizon
    det_img = draw_detections(image, detections)

    seg_img = overlay_all_masks(image, masks)
    seg_img = draw_horizon(seg_img, horizon)

    final_img = draw_detections(seg_img, detections)
    final_img = add_summary_text(final_img, reflection_score, detections, ratios)

    compare_img = make_comparison_grid(
        image,
        det_img,
        seg_img,
        final_img,
        titles=[
            "Original",
            "Object Detection",
            "Segmentation + Horizon",
            "Final Visualization"
        ]
    )

    cv2.putText(final_img, f"Reflection: {reflection_score:.2f}",
                (10, 40),  
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2)

    cv2.imwrite(os.path.join(DETECTION_DIR, f"{image_stem}_det.jpg"), det_img)
    cv2.imwrite(os.path.join(SEGMENT_DIR, f"{image_stem}_seg.jpg"), seg_img)
    cv2.imwrite(os.path.join(FINAL_DIR, f"{image_stem}_final.jpg"), final_img)
    cv2.imwrite(os.path.join(COMPARE_DIR, f"{image_stem}_compare.jpg"), compare_img)

    summary_rows.append({
        "image": image_name,
        "num_detections": len(detections),
        "reflection_score": reflection_score,
        "horizon": horizon,
        "sky_ratio": sky_ratio,
        "water_ratio": water_ratio,
        "land_ratio": land_ratio,
    })

    print(f"Done: {image_name}")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUTPUT_DIR, "visualization_summary.csv"), index=False)

print("\nDONE!")
print("Saved to:", OUTPUT_DIR)