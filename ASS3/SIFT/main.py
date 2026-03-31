import cv2
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================================================
# BƯỚC 0 — LOAD & TIỀN XỬ LÝ ẢNH
# =========================================================

def load_and_preprocess(paths, scale=0.3):
    """
    Bước 1: Load ảnh từ đường dẫn và tiền xử lý:
      - Kiểm tra ảnh hợp lệ
      - Resize về tỉ lệ scale (giảm kích thước để xử lý nhanh)
      - Chuyển sang ảnh xám (grayscale) để phục vụ feature detection

    Trả về: (imgs_color, imgs_gray)
      - imgs_color: danh sách ảnh BGR gốc (dùng để warp và blending)
      - imgs_gray : danh sách ảnh xám tương ứng (dùng để detect feature)
    """
    imgs_color = []
    imgs_gray  = []

    for path in paths:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {path}")

        # ── Resize (chuẩn hóa kích thước) ────────────────────
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)

        # ── Chuyển sang ảnh xám ───────────────────────────────
        # SIFT chỉ cần thông tin cường độ sáng (intensity), không cần màu.
        # Chuyển sang grayscale trước khi detect giúp giảm nhiễu màu
        # và đảm bảo pipeline tường minh theo yêu cầu.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        imgs_color.append(img)
        imgs_gray.append(gray)

        h, w = img.shape[:2]
        print(f"  [✓] {os.path.basename(path):20s}  {w}×{h}px")

    return imgs_color, imgs_gray


# =========================================================
# BƯỚC 1 — TRÍCH XUẤT ĐẶC TRƯNG (SIFT)
# =========================================================

def extract_features_sift(gray_img, n_features=2000):
    """
    Bước 2: Phát hiện keypoints và tính descriptor bằng SIFT.

    SIFT (Scale-Invariant Feature Transform):
      - Phát hiện keypoints trên nhiều tỉ lệ (scale-space / DoG)
      - Tính descriptor 128 chiều bất biến với scale, rotation, illumination
      - n_features: giới hạn số keypoints tốt nhất được giữ lại

    Trả về: (keypoints, descriptors)
    """
    sift = cv2.SIFT_create(nfeatures=n_features)
    keypoints, descriptors = sift.detectAndCompute(gray_img, None)
    return keypoints, descriptors


# =========================================================
# BƯỚC 2 — SO KHỚP ĐẶC TRƯNG
# =========================================================

def match_features(des1, des2, ratio_thresh=0.5):
    """
    Bước 3: So khớp descriptor giữa hai ảnh bằng BFMatcher + KNN + ratio test.

    - BFMatcher với NORM_L2 (phù hợp với SIFT float descriptor)
    - knnMatch(k=2): tìm 2 ứng viên gần nhất
    - Lowe's ratio test: giữ match m nếu m.distance < ratio × n.distance
      → loại bỏ match mơ hồ (ambiguous), chỉ giữ match rõ ràng

    Trả về: danh sách good_matches
    """
    bf      = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    good    = [m for m, n in matches if m.distance < ratio_thresh * n.distance]

    if len(good) < 6:
        raise ValueError(
            f"Không đủ good matches ({len(good)} < 6). "
            "Ảnh cần có vùng overlap lớn hơn hoặc tăng ratio_thresh.")

    return good


# =========================================================
# BƯỚC 3 — ƯỚC LƯỢNG HOMOGRAPHY (RANSAC)
# =========================================================

def compute_homography(kp1, kp2, good_matches):
    """
    Bước 4: Ước lượng ma trận Homography H từ các cặp điểm tương ứng.

    - H ánh xạ tọa độ từ img1 → img2 (src → dst)
    - RANSAC loại bỏ outliers (false matches)
    - Chuẩn hóa H[2,2] = 1 để đảm bảo tính ổn định số học

    Trả về: H (3×3), mask (inlier mask)
    """
    src = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    if H is None:
        raise RuntimeError("findHomography thất bại — không đủ inliers.")

    # Chuẩn hóa để H[2,2] = 1 → ổn định khi chaining nhiều H
    H = H / H[2, 2]

    n_inliers = int(mask.sum())
    print(f"    Inliers: {n_inliers}/{len(good_matches)} "
          f"({n_inliers/len(good_matches)*100:.0f}%)")

    return H, mask


# =========================================================
# BƯỚC 4 — BLENDING: WEIGHT MAP HÌNH CHÓP
# =========================================================

def make_weight_map(img):
    """
    Tạo weight map hình chóp (pyramid / feather mask):
      - Pixel ở tâm ảnh → weight cao
      - Pixel ở rìa ảnh → weight = 0

    Mục đích: khi hai ảnh chồng lấn, pixel có weight cao hơn
    đóng góp nhiều hơn vào kết quả → transition mượt, không seam.

    Kỹ thuật này gọi là "feather blending" hay "distance-to-edge blending".
    """
    h, w = img.shape[:2]

    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)

    weight_x = np.minimum(xs, (w - 1) - xs)        # (w,)  khoảng cách đến cạnh ngang
    weight_y = np.minimum(ys, (h - 1) - ys)        # (h,)  khoảng cách đến cạnh dọc

    weight_map = np.outer(weight_y, weight_x).astype(np.float32)

    max_val = weight_map.max()
    if max_val > 0:
        weight_map /= max_val                       # Normalize về [0, 1]

    return weight_map


# =========================================================
# BƯỚC 5 — WARP + TÍCH LŨY TRÊN CANVAS
# =========================================================

def warp_to_canvas(img, H, canvas, canvas_weight):
    """
    Bước 5: Warp ảnh lên canvas theo H và tích lũy với feather blending.

    Công thức blending:
      canvas_final[x,y] = Σ(color_i × weight_i) / Σ(weight_i)

    - Vùng chỉ có 1 ảnh → pixel đó chiếm toàn bộ trọng số
    - Vùng overlap    → trộn theo weight map của từng ảnh (mượt dần từ tâm ra biên)
    """
    cH, cW = canvas.shape[:2]

    # Warp ảnh màu lên canvas
    warped        = cv2.warpPerspective(img, H, (cW, cH)).astype(np.float32)

    # Warp weight map của ảnh gốc theo cùng H
    weight_src    = make_weight_map(img)
    warped_weight = cv2.warpPerspective(weight_src, H, (cW, cH))

    # Tích lũy: chỉ xử lý pixel có weight > 0 (pixel hợp lệ sau warp)
    valid = warped_weight > 0
    canvas[valid]        += warped[valid] * warped_weight[valid, np.newaxis]
    canvas_weight[valid] += warped_weight[valid]

    return canvas, canvas_weight


# =========================================================
# BƯỚC 6 — TÍNH CANVAS
# =========================================================

def compute_canvas_bounds(images, H_list):
    """
    Hardcode canvas:
      - Chiều cao = h * 3
      - Chiều rộng = w * 6
      - Offset đặt ảnh trung tâm vào giữa canvas
    """
    h, w = images[0].shape[:2]

    cH = h * 3
    cW = w * 6

    # Đặt ảnh trung tâm gần giữa canvas
    offset_x = w * 2
    offset_y = h * 1

    return cW, cH, offset_x, offset_y

# =========================================================
# PIPELINE CHÍNH — CENTER-BASED STITCHING
# =========================================================

def stitch_center(imgs_color, imgs_gray, output_dir):
    """
    Pipeline ghép ảnh toàn cảnh theo chiến lược Center-based Stitching.

    Chiến lược:
      - Chọn ảnh trung tâm C = N//2 làm hệ tọa độ tham chiếu
      - Tính H từng ảnh bên TRÁI → chain về C
      - Tính H từng ảnh bên PHẢI → chain về C
      - Warp tất cả ảnh gốc lên canvas chung + feather blend

    Ưu điểm so với chained-left:
      - Lỗi tích lũy (drift) phân bổ đều 2 phía → giảm ~50%
      - Canvas cân bằng trái-phải → nhỏ hơn, ít méo hơn
    """
    n          = len(imgs_color)
    center_idx = n // 2

    print(f"\n[*] Ảnh trung tâm: #{center_idx + 1}")

    H_to_center = [None] * n
    H_to_center[center_idx] = np.eye(3, dtype=np.float64)

    # ================= LEFT =================
    H_acc = np.eye(3, dtype=np.float64)
    for i in range(center_idx - 1, -1, -1):
        print(f"\n  Tính H: ảnh {i+1} → ảnh {i+2}")

        kp1, des1 = extract_features_sift(imgs_gray[i])
        kp2, des2 = extract_features_sift(imgs_gray[i + 1])

        print(f"    Keypoints: {len(kp1)} / {len(kp2)}", end="  ")

        # 🔥 SAVE KEYPOINTS
        save_keypoints_vis(
            imgs_color[i], imgs_gray[i],
            os.path.join(output_dir, f"sift_kp_img_{i+1}.jpg")
        )
        save_keypoints_vis(
            imgs_color[i+1], imgs_gray[i+1],
            os.path.join(output_dir, f"sift_kp_img_{i+2}.jpg")
        )

        good = match_features(des1, des2)
        print(f"    Good matches: {len(good)}", end="  ")

        # 🔥 SAVE MATCHES
        save_matches_vis(
            imgs_color[i], imgs_color[i+1],
            imgs_gray[i], imgs_gray[i+1],
            os.path.join(output_dir, f"sift_match_{i+1}_{i+2}.jpg")
        )

        H, _ = compute_homography(kp1, kp2, good)

        H_acc = H_acc @ H
        H_to_center[i] = H_acc.copy()

    # ================= RIGHT =================
    H_acc = np.eye(3, dtype=np.float64)
    for i in range(center_idx + 1, n):
        print(f"\n  Tính H: ảnh {i+1} → ảnh {i}")

        kp1, des1 = extract_features_sift(imgs_gray[i])
        kp2, des2 = extract_features_sift(imgs_gray[i - 1])

        print(f"    Keypoints: {len(kp1)} / {len(kp2)}", end="  ")

        # 🔥 SAVE KEYPOINTS
        save_keypoints_vis(
            imgs_color[i], imgs_gray[i],
            os.path.join(output_dir, f"sift_kp_img_{i+1}.jpg")
        )
        save_keypoints_vis(
            imgs_color[i-1], imgs_gray[i-1],
            os.path.join(output_dir, f"sift_kp_img_{i}.jpg")
        )

        good = match_features(des1, des2)
        print(f"    Good matches: {len(good)}", end="  ")

        # 🔥 SAVE MATCHES
        save_matches_vis(
            imgs_color[i], imgs_color[i-1],
            imgs_gray[i], imgs_gray[i-1],
            os.path.join(output_dir, f"sift_match_{i+1}_{i}.jpg")
        )

        H, _ = compute_homography(kp1, kp2, good)

        H_acc = H_acc @ H
        H_to_center[i] = H_acc.copy()

    # ================= HARDCODE CANVAS =================
    h, w = imgs_color[0].shape[:2]
    cH, cW = h * 3, w * 6
    off_x, off_y = w * 2, h * 1

    print(f"\n[*] Canvas hardcode: {cW}×{cH}")

    T_offset = np.array([
        [1, 0, off_x],
        [0, 1, off_y],
        [0, 0, 1]
    ], dtype=np.float64)

    canvas        = np.zeros((cH, cW, 3), dtype=np.float32)
    canvas_weight = np.zeros((cH, cW),    dtype=np.float32)

    for i, (img, H_rel) in enumerate(zip(imgs_color, H_to_center)):
        H_final = T_offset @ H_rel
        canvas, canvas_weight = warp_to_canvas(img, H_final, canvas, canvas_weight)

        print(f"  Warped ảnh {i+1}/{n}")

    safe_weight = np.where(canvas_weight > 0, canvas_weight, 1.0)
    result = canvas / safe_weight[:, :, np.newaxis]
    result = np.clip(result, 0, 255).astype(np.uint8)
    result[canvas_weight == 0] = 0

    return crop_black(result)
# =========================================================
# BƯỚC 7 — CROP VÙNG ĐEN
# =========================================================

def crop_black(img):
    """
    Cắt bỏ vùng đen (pixel = 0) xung quanh panorama sau khi warp.
    Dùng boundingRect của contour lớn nhất để tìm vùng hợp lệ.
    """
    gray      = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    cnt      = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return img[y:y+h, x:x+w]


# =========================================================
# VISUALIZATION — lưu ảnh minh họa cho báo cáo
# =========================================================

def save_keypoints_vis(img_color, gray, path):
    """Vẽ keypoints SIFT lên ảnh và lưu file."""
    sift = cv2.SIFT_create(nfeatures=2000)
    kps, _ = sift.detectAndCompute(gray, None)
    vis    = cv2.drawKeypoints(img_color, kps, None,
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.set_title(f"SIFT Keypoints — {len(kps)} điểm", fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [✓] {os.path.basename(path)}")


def save_matches_vis(img1, img2, gray1, gray2, path, ratio_thresh=0.5):
    """Vẽ inlier matches giữa hai ảnh liền kề và lưu file."""
    sift      = cv2.SIFT_create(nfeatures=2000)
    kp1, d1   = sift.detectAndCompute(gray1, None)
    kp2, d2   = sift.detectAndCompute(gray2, None)
    good      = match_features(d1, d2, ratio_thresh)
    src       = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst       = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    _, mask   = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers   = [m for m, k in zip(good, mask.ravel()) if k]
    vis = cv2.drawMatches(img1, kp1, img2, kp2, inliers[:80], None,
                           matchColor=(0,180,255),
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.set_title(f"SIFT Inlier Matches — {len(inliers)} inliers / {len(good)} good matches",
                 fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [✓] {os.path.basename(path)}")


def save_panorama_vis(panorama, path):
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Panorama — SIFT Center-based  "
                 f"[{panorama.shape[1]}×{panorama.shape[0]}px]",
                 fontsize=14, fontweight="bold", color="navy")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [✓] {os.path.basename(path)}")


# =========================================================
# MAIN
# =========================================================

IMAGE_PATHS = [
    "../input/img1.jpg",
    "../input/img2.jpg",
    "../input/img3.jpg",
    "../input/img4.jpg",
]
SCALE      = 0.3
OUTPUT_DIR = "./output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 54)
print("  Panoramic Image Stitching — SIFT Center-based")
print("=" * 54)

# ── Bước 0: Load & tiền xử lý ──────────────────────────
print(f"\n[*] Bước 0: Load và tiền xử lý (scale={SCALE})...")
imgs_color, imgs_gray = load_and_preprocess(IMAGE_PATHS, scale=SCALE)

# ── Bước 1–7: Pipeline ghép ảnh ─────────────────────────
print("\n[*] Chạy pipeline Center-based Stitching...")
panorama = stitch_center(imgs_color, imgs_gray, OUTPUT_DIR)

# ── Lưu kết quả ─────────────────────────────────────────
out_jpg = os.path.join(OUTPUT_DIR, "panorama_sift.jpg")
cv2.imwrite(out_jpg, panorama, [cv2.IMWRITE_JPEG_QUALITY, 95])
print(f"\n[✓] Panorama: {panorama.shape[1]}×{panorama.shape[0]}px → {out_jpg}")

save_panorama_vis(panorama, os.path.join(OUTPUT_DIR, "sift_panorama_vis.jpg"))
print(f"\n[✓] Xong! Tất cả kết quả tại: {OUTPUT_DIR}/")