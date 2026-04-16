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
# Core3.py nằm trong preprocessing/
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGE_DIR = os.path.join(BASE_DIR, "input")
MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_33")
MASK_DIR = os.path.join(OUTPUT_DIR, "masks")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)


# =========================
# FUNCTIONS
# =========================
def load_image(path):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_bgr, img_rgb


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


    sky[: int(h * 0.6), :] = sky[: int(h * 0.6), :]
    sky[int(h * 0.6):, :] = 0
    water = (water & lower_half).astype(np.uint8)

    # Morphology lại lần cuối
    sky = clean_mask(sky, kernel_size=11, min_area=5000)
    water = clean_mask(water, kernel_size=11, min_area=5000)

    # Land = phần còn lại
    land = np.ones((h, w), dtype=np.uint8)
    land[(sky == 1) | (water == 1)] = 0

    # clean land nhẹ hơn vì land có thể rời rạc
    land = cv2.morphologyEx(land, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

    return {
        "sky": sky.astype(np.uint8),
        "water": water.astype(np.uint8),
        "land": land.astype(np.uint8),
    }


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
    print("Processing:", name)

    img_bgr, img_rgb = load_image(path)

    # 1. Object detection
    detections = run_object_detection(img_bgr)

    # 2. Scene segmentation
    masks = segment_scene(img_rgb)

    # save masks
    for k, m in masks.items():
        cv2.imwrite(
            os.path.join(MASK_DIR, f"{name}_{k}.png"),
            (m * 255).astype(np.uint8)
        )

    # 3. Reflection
    horizon = estimate_horizon(masks)
    ref_score = reflection_score(img_rgb, horizon)

    # 4. Scene stats
    h, w = img_rgb.shape[:2]
    sky_ratio = float(masks["sky"].sum() / (h * w))
    water_ratio = float(masks["water"].sum() / (h * w))
    land_ratio = float(masks["land"].sum() / (h * w))

    scene_data.append({
        "image": name,
        "horizon": horizon,
        "reflection_score": ref_score,
        "sky_ratio": sky_ratio,
        "water_ratio": water_ratio,
        "land_ratio": land_ratio,
    })

    # 5. Detection stats
    for d in detections:
        d["image"] = name
        det_data.append(d)

    # 6. Save JSON per image
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
print("Saved scene.csv, detections.csv and masks to:", OUTPUT_DIR)