import glob
import json
import os

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import KMeans
from ultralytics import YOLO

# =========================
# PATH CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_DIR = os.path.join(BASE_DIR, "input")
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_33")

MASK_DIR = os.path.join(OUTPUT_DIR, "masks")
DET_DIR = os.path.join(OUTPUT_DIR, "detections")
SEG_DIR = os.path.join(OUTPUT_DIR, "segmentations")
HOR_DIR = os.path.join(OUTPUT_DIR, "horizons")
FINAL_DIR = os.path.join(OUTPUT_DIR, "final")
COMPARE_DIR = os.path.join(OUTPUT_DIR, "compare")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(DET_DIR, exist_ok=True)
os.makedirs(SEG_DIR, exist_ok=True)
os.makedirs(HOR_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)
os.makedirs(COMPARE_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)


# =========================
# BASIC FUNCTIONS
# =========================
def load_image(path):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_bgr, img_rgb


def save_rgb(path, img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)


# =========================
# OBJECT DETECTION
# =========================
def run_object_detection(img_bgr):
    results = model(img_bgr, conf=0.2, verbose=False)[0]

    detections = []
    if results.boxes is None:
        return detections

    for box in results.boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())

        detections.append({
            "class": model.names[cls_id],
            "confidence": conf,
            "x1": int(xyxy[0]),
            "y1": int(xyxy[1]),
            "x2": int(xyxy[2]),
            "y2": int(xyxy[3]),
        })

    return detections


def visualize_detections(img_bgr, detections):
    vis = img_bgr.copy()

    for det in detections:
        x1, y1 = det["x1"], det["y1"]
        x2, y2 = det["x2"], det["y2"]
        cls_name = det["class"]
        conf = det["confidence"]

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)

        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, max(0, y1 - th - 6)), (x1 + tw, y1), (0, 255, 255), -1)
        cv2.putText(
            vis,
            label,
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )

    return vis


# =========================
# SEGMENTATION
# =========================
def clean_mask(mask, kernel_size=7, min_area=2000):
    """
    Hậu xử lý mask:
    - open để bỏ nhiễu nhỏ
    - close để lấp lỗ
    - connected components để giữ vùng đủ lớn
    """
    mask = (mask > 0).astype(np.uint8)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 1

    return cleaned.astype(np.uint8)


def fill_largest_component(mask):
    """
    Giữ lại thành phần liên thông lớn nhất.
    Hợp với sky/water vì thường là vùng lớn.
    """
    mask = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return mask

    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(mask)
    out[labels == largest_idx] = 1
    return out


def segment_scene(img_rgb):
    """
    Segment scene thành 3 vùng chính: sky / water / land
    bằng KMeans + heuristic vị trí + hậu xử lý.
    """
    h, w, _ = img_rgb.shape

    # KMeans trong Lab
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    pixels = img_lab.reshape(-1, 3).astype(np.float32)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels).reshape(h, w)

    # Tìm cluster theo vị trí trung bình theo chiều dọc
    cluster_y = []
    for i in range(3):
        ys = np.where(labels == i)[0]
        mean_y = np.mean(ys) if len(ys) > 0 else h / 2
        cluster_y.append(mean_y)

    # sky ở cao nhất, water ở thấp nhất
    sky_cluster = int(np.argmin(cluster_y))
    water_cluster = int(np.argmax(cluster_y))

    raw_sky = (labels == sky_cluster).astype(np.uint8)
    raw_water = (labels == water_cluster).astype(np.uint8)

    # Hậu xử lý
    sky = clean_mask(raw_sky, kernel_size=9, min_area=3000)
    water = clean_mask(raw_water, kernel_size=5, min_area=1500)

    # Giữ vùng lớn nhất cho sky và water để giảm noise
    sky = fill_largest_component(sky)
    water = fill_largest_component(water)

    # Ưu tiên sky ở nửa trên, water ở nửa dưới
    upper_half = np.zeros((h, w), dtype=np.uint8)
    upper_half[: h // 2, :] = 1

    lower_half = np.zeros((h, w), dtype=np.uint8)
    lower_half[h // 2 :, :] = 1

    sky = (sky & upper_half).astype(np.uint8)

    # Giới hạn thêm cho sky để tránh tràn xuống dưới
    sky[int(h * 0.6):, :] = 0
    water = (water & lower_half).astype(np.uint8)

    # Morphology lại lần cuối
    sky = clean_mask(sky, kernel_size=11, min_area=5000)
    water = clean_mask(water, kernel_size=11, min_area=5000)

    # Land = phần còn lại
    land = np.ones((h, w), dtype=np.uint8)
    land[(sky == 1) | (water == 1)] = 0

    # clean land nhẹ hơn vì land có thể rời rạc
    land = cv2.morphologyEx(land, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    return {
        "sky": sky.astype(np.uint8),
        "water": water.astype(np.uint8),
        "land": land.astype(np.uint8),
    }


def make_segmentation_overlay(img_rgb, masks, alpha=0.35):
    """
    Tạo ảnh overlay phân vùng:
    - sky: xanh dương
    - water: cyan
    - land: xanh lá
    """
    overlay = img_rgb.copy()

    color_layer = np.zeros_like(img_rgb)
    color_layer[masks["sky"] == 1] = [0, 120, 255]      # sky
    color_layer[masks["water"] == 1] = [0, 255, 255]    # water
    color_layer[masks["land"] == 1] = [0, 200, 0]       # land

    seg = cv2.addWeighted(overlay, 1 - alpha, color_layer, alpha, 0)

    # chú thích
    cv2.putText(seg, "Sky", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 120, 255), 2, cv2.LINE_AA)
    cv2.putText(seg, "Water", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(seg, "Land", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2, cv2.LINE_AA)

    return seg


# =========================
# HORIZON + REFLECTION
# =========================
def estimate_horizon(masks):
    """
    Ước lượng horizon từ biên dưới của sky mask.
    """
    h, w = masks["sky"].shape
    ys = []

    for x in range(w):
        col = np.where(masks["sky"][:, x] > 0)[0]
        if len(col) > 0:
            ys.append(col.max())

    if len(ys) == 0:
        return h // 2

    return int(np.median(ys))


def reflection_score(img_rgb, horizon):
    upper = img_rgb[:horizon]
    lower = img_rgb[horizon:]

    if upper.shape[0] < 10 or lower.shape[0] < 10:
        return 0.0

    min_h = min(upper.shape[0], lower.shape[0])
    upper = upper[-min_h:]
    lower = lower[:min_h]

    lower = np.flipud(lower)

    u = cv2.cvtColor(upper, cv2.COLOR_RGB2GRAY)
    l = cv2.cvtColor(lower, cv2.COLOR_RGB2GRAY)

    return float(ssim(u, l, data_range=255))


def draw_horizon(img_rgb, horizon, label=True):
    vis = img_rgb.copy()
    h, w = vis.shape[:2]

    cv2.line(vis, (0, horizon), (w - 1, horizon), (255, 0, 0), 2)

    if label:
        text = f"horizon = {horizon}"
        cv2.rectangle(vis, (5, max(0, horizon - 24)), (140, max(18, horizon - 4)), (255, 0, 0), -1)
        cv2.putText(
            vis,
            text,
            (8, max(14, horizon - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return vis


def compute_region_ratios(masks):
    h, w = masks["sky"].shape
    total = float(h * w)

    return {
        "sky_ratio": float(masks["sky"].sum() / total),
        "water_ratio": float(masks["water"].sum() / total),
        "land_ratio": float(masks["land"].sum() / total),
    }


def make_final_visualization(img_rgb, detections, masks, horizon, ref_score):
    vis = img_rgb.copy()

    # Horizon
    vis = draw_horizon(vis, horizon, label=False)

    # Detection boxes
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    vis_bgr = visualize_detections(vis_bgr, detections)
    vis = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

    # Region ratios
    ratios = compute_region_ratios(masks)

    info_lines = [
        f"Reflection: {ref_score:.2f}",
        f"Detections: {len(detections)}",
        f"Sky ratio: {ratios['sky_ratio']:.3f}",
        f"Water ratio: {ratios['water_ratio']:.3f}",
        f"Land ratio: {ratios['land_ratio']:.3f}",
    ]

    # Box info góc trái trên
    box_w = 170
    box_h = 20 + 22 * len(info_lines)
    cv2.rectangle(vis, (8, 8), (8 + box_w, 8 + box_h), (30, 30, 30), -1)

    y = 28
    for i, line in enumerate(info_lines):
        color = (255, 255, 0) if i == 0 else (255, 255, 255)
        thickness = 2 if i == 0 else 1
        cv2.putText(
            vis,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            thickness,
            cv2.LINE_AA
        )
        y += 22

    return vis


def put_title(img_rgb, title):
    out = img_rgb.copy()
    cv2.putText(
        out,
        title,
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        4,
        cv2.LINE_AA
    )
    cv2.putText(
        out,
        title,
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    return out


def make_compare_panel(img_rgb, det_vis_bgr, seg_vis_rgb, final_vis_rgb):
    """
    Ghép 2x2:
    - Original
    - Object Detection
    - Segmentation + Horizon
    - Final Visualization
    """
    original = put_title(img_rgb.copy(), "Original")
    det_rgb = cv2.cvtColor(det_vis_bgr, cv2.COLOR_BGR2RGB)
    det_rgb = put_title(det_rgb, "Object Detection")
    seg_rgb = put_title(seg_vis_rgb.copy(), "Segmentation + Horizon")
    final_rgb = put_title(final_vis_rgb.copy(), "Final Visualization")

    h, w = img_rgb.shape[:2]

    top = np.hstack([original, det_rgb])
    bottom = np.hstack([seg_rgb, final_rgb])
    panel = np.vstack([top, bottom])

    return panel


# =========================
# MAIN
# =========================
image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))

print("BASE_DIR =", BASE_DIR)
print("IMAGE_DIR =", IMAGE_DIR)
print("Found images:", len(image_paths))

scene_data = []
det_data = []

for path in image_paths:
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    print("Processing:", name)

    img_bgr, img_rgb = load_image(path)

    # 1. Object detection
    detections = run_object_detection(img_bgr)
    det_vis_bgr = visualize_detections(img_bgr, detections)
    cv2.imwrite(os.path.join(DET_DIR, f"{stem}_det.jpg"), det_vis_bgr)

    # 2. Scene segmentation
    masks = segment_scene(img_rgb)

    # save masks
    for k, m in masks.items():
        cv2.imwrite(
            os.path.join(MASK_DIR, f"{name}_{k}.png"),
            (m * 255).astype(np.uint8)
        )

    # 3. Horizon
    horizon = estimate_horizon(masks)

    # 4. Reflection
    ref_score = reflection_score(img_rgb, horizon)

    # 5. Scene stats
    ratios = compute_region_ratios(masks)
    sky_ratio = ratios["sky_ratio"]
    water_ratio = ratios["water_ratio"]
    land_ratio = ratios["land_ratio"]

    # 6. Segmentation visualization
    seg_vis = make_segmentation_overlay(img_rgb, masks, alpha=0.35)
    seg_vis = draw_horizon(seg_vis, horizon, label=True)
    save_rgb(os.path.join(SEG_DIR, f"{stem}_seg.jpg"), seg_vis)

    # 7. Horizon-only visualization
    hor_vis = draw_horizon(img_rgb, horizon, label=True)
    save_rgb(os.path.join(HOR_DIR, f"{stem}_horizon.jpg"), hor_vis)

    # 8. Final visualization
    final_vis = make_final_visualization(img_rgb, detections, masks, horizon, ref_score)
    save_rgb(os.path.join(FINAL_DIR, f"{stem}_final.jpg"), final_vis)

    # 9. Compare visualization
    compare_vis = make_compare_panel(img_rgb, det_vis_bgr, seg_vis, final_vis)
    save_rgb(os.path.join(COMPARE_DIR, f"{stem}_compare.jpg"), compare_vis)

    # 10. Scene stats table
    scene_data.append({
        "image": name,
        "horizon": horizon,
        "reflection_score": ref_score,
        "sky_ratio": sky_ratio,
        "water_ratio": water_ratio,
        "land_ratio": land_ratio,
    })

    # 11. Detection stats table
    for d in detections:
        d["image"] = name
        det_data.append(d)

    # 12. Save JSON per image
    with open(os.path.join(OUTPUT_DIR, name + ".json"), "w", encoding="utf-8") as f:
        json.dump({
            "image": name,
            "detections": detections,
            "reflection": ref_score,
            "reflection_score": ref_score,
            "horizon": horizon,
            "sky_ratio": sky_ratio,
            "water_ratio": water_ratio,
            "land_ratio": land_ratio,
        }, f, indent=2, ensure_ascii=False)

# Save CSV
pd.DataFrame(scene_data).to_csv(os.path.join(OUTPUT_DIR, "scene.csv"), index=False)
pd.DataFrame(det_data).to_csv(os.path.join(OUTPUT_DIR, "detections.csv"), index=False)

print("\nDONE!")
print("Saved results to:", OUTPUT_DIR)
print(" - masks/")
print(" - detections/")
print(" - segmentations/")
print(" - horizons/")
print(" - final/")
print(" - compare/")
print(" - scene.csv")
print(" - detections.csv")