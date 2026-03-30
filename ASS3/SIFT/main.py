import cv2
import numpy as np
import os


def compute_homography(img1, img2):
    sift = cv2.SIFT_create(2000)
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < 8:
        raise ValueError("Not enough matches")

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    H = H / H[2, 2]
    return H


def make_weight_map(img):
    """
    Tạo weight map hình chóp (pyramid):
    - Pixel ở giữa ảnh → weight cao
    - Pixel ở rìa       → weight = 0
    Dùng để feather blend tại vùng overlap.
    """
    h, w = img.shape[:2]

    # Khoảng cách từ mỗi pixel tới rìa gần nhất (4 cạnh)
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)

    dist_left   = xs                    # khoảng cách tới cạnh trái
    dist_right  = (w - 1) - xs         # khoảng cách tới cạnh phải
    dist_top    = ys                    # khoảng cách tới cạnh trên
    dist_bottom = (h - 1) - ys         # khoảng cách tới cạnh dưới

    # Weight theo chiều ngang và dọc
    weight_x = np.minimum(dist_left, dist_right)    # shape (w,)
    weight_y = np.minimum(dist_top,  dist_bottom)   # shape (h,)

    # Outer product → weight map 2D, shape (h, w)
    weight_map = np.outer(weight_y, weight_x).astype(np.float32)

    # Normalize về [0, 1]
    max_val = weight_map.max()
    if max_val > 0:
        weight_map /= max_val

    return weight_map


def warp_to_canvas(img, H, canvas, canvas_weight):
    """
    Warp ảnh lên canvas với feather blending:
      - canvas        : bộ tích lũy màu  (float32, shape H×W×3)
      - canvas_weight : bộ tích lũy weight (float32, shape H×W)

    Mỗi pixel = tổng (màu × weight) / tổng weight
    """
    cH, cW = canvas.shape[:2]

    # ── Warp ảnh màu ──────────────────────────────────────────────────────────
    warped = cv2.warpPerspective(img, H, (cW, cH)).astype(np.float32)

    # ── Warp weight map của ảnh gốc ───────────────────────────────────────────
    weight_src = make_weight_map(img)                          # (h, w)
    warped_weight = cv2.warpPerspective(weight_src, H, (cW, cH))  # (cH, cW)

    # ── Tích lũy (chỉ pixel hợp lệ của warped) ────────────────────────────────
    valid = warped_weight > 0
    canvas[valid]        += warped[valid] * warped_weight[valid, np.newaxis]
    canvas_weight[valid] += warped_weight[valid]

    return canvas, canvas_weight


def stitch_center(images):
    n = len(images)
    center_idx = n // 2
    center_img = images[center_idx]

    h, w = center_img.shape[:2]
    cH, cW = h * 3, w * 6

    # Dùng float32 để tích lũy màu và weight
    canvas        = np.zeros((cH, cW, 3), dtype=np.float32)
    canvas_weight = np.zeros((cH, cW),    dtype=np.float32)

    offset_x = (cW - w) // 2
    offset_y = (cH - h) // 2

    H_center = np.array([[1, 0, offset_x],
                         [0, 1, offset_y],
                         [0, 0, 1]], dtype=np.float64)

    # Đặt center image vào canvas (cũng dùng feather blend)
    canvas, canvas_weight = warp_to_canvas(center_img, H_center, canvas, canvas_weight)

    # ── LEFT ──────────────────────────────────────────────────────────────────
    H_acc = H_center.copy()
    for i in range(center_idx - 1, -1, -1):
        print(f"Stitch LEFT [{i}]")
        H = compute_homography(images[i], images[i + 1])
        H_acc = H_acc @ H
        canvas, canvas_weight = warp_to_canvas(images[i], H_acc, canvas, canvas_weight)

    # ── RIGHT ─────────────────────────────────────────────────────────────────
    H_acc = H_center.copy()
    for i in range(center_idx + 1, n):
        print(f"Stitch RIGHT [{i}]")
        H = compute_homography(images[i], images[i - 1])
        H_acc = H_acc @ H
        canvas, canvas_weight = warp_to_canvas(images[i], H_acc, canvas, canvas_weight)

    # ── Normalize: màu cuối = tổng(màu×w) / tổng(w) ──────────────────────────
    safe_weight = np.where(canvas_weight > 0, canvas_weight, 1.0)
    result = canvas / safe_weight[:, :, np.newaxis]
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Vùng không có ảnh nào → đen
    result[canvas_weight == 0] = 0

    return crop_black(result)


def crop_black(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return img[y:y+h, x:x+w]


# =========================
# MAIN
# =========================
imgs = [
    cv2.imread("../input/img1.jpg"),
    cv2.imread("../input/img2.jpg"),
    cv2.imread("../input/img3.jpg"),
    cv2.imread("../input/img4.jpg"),
]

imgs = [cv2.resize(img, (0, 0), fx=0.3, fy=0.3) for img in imgs]

panorama = stitch_center(imgs)

os.makedirs("./output", exist_ok=True)
cv2.imwrite("./output/panorama_center.jpg", panorama)