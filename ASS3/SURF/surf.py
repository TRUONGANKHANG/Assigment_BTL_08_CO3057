import cv2
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

# =========================================================
# SURF
#
# Pipeline:
# 1. Load ảnh  
# 2. Chuyển grayscale  
# 3. SURF feature detection  
# 4. Matching (KNN + ratio test) cho TẤT CẢ các cặp 
# 5. Visualization Matches (Lưu ảnh minh họa)
# 6. Homography (Center-based) 
# 7. Warp + Stitch + Crop
# =========================================================


# =========================================================
# HÀM KHỞI TẠO SURF 
# =========================================================
def get_surf(hessianThreshold=400):
    try:
        return cv2.xfeatures2d.SURF_create(hessianThreshold=hessianThreshold)
    except AttributeError:
        try:
            return cv2.SURF_create(hessianThreshold=hessianThreshold)
        except AttributeError:
            raise RuntimeError("Lỗi: Không tìm thấy SURF. Hãy chắc chắn bạn đã cài opencv-contrib-python.")

# =========================================================
# B0
# =========================================================
def load_and_preprocess(paths, scale=0.3):
    imgs_color = []
    imgs_gray  = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {path}")
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgs_color.append(img)
        imgs_gray.append(gray)
        h, w = img.shape[:2]
        print(f"  [✓] {os.path.basename(path):20s}  {w}×{h}px")
    return imgs_color, imgs_gray

# =========================================================
# B1
# =========================================================
def extract_features_surf(gray_img, hessianThreshold=400):
    surf = get_surf(hessianThreshold)
    keypoints, descriptors = surf.detectAndCompute(gray_img, None)
    return keypoints, descriptors

# =========================================================
# B2
# =========================================================
def match_features(des1, des2, ratio_thresh=0.75):
    bf = cv2.BFMatcher(cv2.NORM_L2) # SURF dùng NORM_L2 giống SIFT
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < ratio_thresh * n.distance]
    if len(good) < 6:
        raise ValueError(f"Không đủ good matches ({len(good)} < 6).")
    return good

# =========================================================
# B3
# =========================================================
def compute_homography(kp1, kp2, good_matches):
    src = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("findHomography thất bại — không đủ inliers.")
    H = H / H[2, 2]
    n_inliers = int(mask.sum())
    print(f"    Inliers: {n_inliers}/{len(good_matches)} ({n_inliers/len(good_matches)*100:.0f}%)")
    return H, mask

# =========================================================
# B4 & B5
# =========================================================
def make_weight_map(img):
    h, w = img.shape[:2]
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    weight_x = np.minimum(xs, (w - 1) - xs)        
    weight_y = np.minimum(ys, (h - 1) - ys)        
    weight_map = np.outer(weight_y, weight_x).astype(np.float32)
    max_val = weight_map.max()
    if max_val > 0:
        weight_map /= max_val                       
    return weight_map

def warp_to_canvas(img, H, canvas, canvas_weight):
    cH, cW = canvas.shape[:2]
    warped = cv2.warpPerspective(img, H, (cW, cH)).astype(np.float32)
    weight_src = make_weight_map(img)
    warped_weight = cv2.warpPerspective(weight_src, H, (cW, cH))
    valid = warped_weight > 0
    canvas[valid] += warped[valid] * warped_weight[valid, np.newaxis]
    canvas_weight[valid] += warped_weight[valid]
    return canvas, canvas_weight

# =========================================================
# VISUALIZATION 
# =========================================================
def save_keypoints_vis(img_color, kps, path):
    vis = cv2.drawKeypoints(img_color, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.set_title(f"SURF Keypoints — {len(kps)} điểm", fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close() 
    print(f"  Done! {os.path.basename(path)}")

def save_matches_vis(img1, img2, kp1, kp2, good, path):
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inliers = [m for m, k in zip(good, mask.ravel()) if k]
    vis = cv2.drawMatches(img1, kp1, img2, kp2, inliers[:80], None,
                          matchColor=(0,180,255), flags=2)
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.set_title(f"SURF Inlier Matches — {len(inliers)} inliers / {len(good)} good matches",
                 fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close() 
    print(f"  [✓] {os.path.basename(path)}")

def save_panorama_vis(panorama, path):
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.imshow(cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB))
    ax.set_title(f"Panorama — SURF Center-based  [{panorama.shape[1]}×{panorama.shape[0]}px]",
                 fontsize=14, fontweight="bold", color="darkred")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close() 
    print(f"  [✓] {os.path.basename(path)}")

# =========================================================
# B7: SỬA LỖI TƯƠNG THÍCH CHO OpenCV 3.4.2.16
# =========================================================
def crop_black(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    # Ở cv2 bản 3.4, findContours trả về 3 biến. Bản 4 trả về 2 biến.
    # Dòng này giúp lấy mảng contours chính xác trên cả 2 phiên bản.
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]
    
    if not contours: return img
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return img[y:y+h, x:x+w]

# =========================================================
# PIPELINE 
# =========================================================
def stitch_center(imgs_color, imgs_gray, output_dir):
    n = len(imgs_color)
    center_idx = n // 2
    print(f"\n[*] Ảnh trung tâm: #{center_idx + 1}")

    H_to_center = [None] * n
    H_to_center[center_idx] = np.eye(3, dtype=np.float64)

    H_acc = np.eye(3, dtype=np.float64)
    for i in range(center_idx - 1, -1, -1):
        print(f"\n  Tính H: ảnh {i+1} → ảnh {i+2}")
        kp1, des1 = extract_features_surf(imgs_gray[i])
        kp2, des2 = extract_features_surf(imgs_gray[i + 1])
        print(f"    Keypoints: {len(kp1)} / {len(kp2)}", end="  ")

        save_keypoints_vis(imgs_color[i], kp1, os.path.join(output_dir, f"surf_kp_img_{i+1}.jpg"))
        save_keypoints_vis(imgs_color[i+1], kp2, os.path.join(output_dir, f"surf_kp_img_{i+2}.jpg"))

        good = match_features(des1, des2)
        print(f"    Good matches: {len(good)}", end="  ")

        save_matches_vis(imgs_color[i], imgs_color[i+1], kp1, kp2, good, os.path.join(output_dir, f"surf_match_{i+1}_{i+2}.jpg"))

        H, _ = compute_homography(kp1, kp2, good)
        H_acc = H_acc @ H
        H_to_center[i] = H_acc.copy()

    H_acc = np.eye(3, dtype=np.float64)
    for i in range(center_idx + 1, n):
        print(f"\n  Tính H: ảnh {i+1} → ảnh {i}")
        kp1, des1 = extract_features_surf(imgs_gray[i])
        kp2, des2 = extract_features_surf(imgs_gray[i - 1])
        print(f"    Keypoints: {len(kp1)} / {len(kp2)}", end="  ")

        save_keypoints_vis(imgs_color[i], kp1, os.path.join(output_dir, f"surf_kp_img_{i+1}.jpg"))
        save_keypoints_vis(imgs_color[i-1], kp2, os.path.join(output_dir, f"surf_kp_img_{i}.jpg"))

        good = match_features(des1, des2)
        print(f"    Good matches: {len(good)}", end="  ")

        save_matches_vis(imgs_color[i], imgs_color[i-1], kp1, kp2, good, os.path.join(output_dir, f"surf_match_{i+1}_{i}.jpg"))

        H, _ = compute_homography(kp1, kp2, good)
        H_acc = H_acc @ H
        H_to_center[i] = H_acc.copy()

    h, w = imgs_color[0].shape[:2]
    cH, cW = h * 3, w * 6
    off_x, off_y = w * 2, h * 1

    T_offset = np.array([[1, 0, off_x], [0, 1, off_y], [0, 0, 1]], dtype=np.float64)
    canvas = np.zeros((cH, cW, 3), dtype=np.float32)
    canvas_weight = np.zeros((cH, cW), dtype=np.float32)

    for i, (img, H_rel) in enumerate(zip(imgs_color, H_to_center)):
        H_final = T_offset @ H_rel
        canvas, canvas_weight = warp_to_canvas(img, H_final, canvas, canvas_weight)

    safe_weight = np.where(canvas_weight > 0, canvas_weight, 1.0)
    result = canvas / safe_weight[:, :, np.newaxis]
    result = np.clip(result, 0, 255).astype(np.uint8)
    result[canvas_weight == 0] = 0

    return crop_black(result)

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
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
    print("Panoramic Image Stitching — SURF Center-based")
    print("=" * 54)

    print(f"\nBước 0: Load và tiền xử lý (scale={SCALE})...")
    imgs_color, imgs_gray = load_and_preprocess(IMAGE_PATHS, scale=SCALE)

    print("\nPipeline Center-based Stitching với SURF...")
    panorama = stitch_center(imgs_color, imgs_gray, OUTPUT_DIR)

    out_jpg = os.path.join(OUTPUT_DIR, "panorama_surf.jpg")
    cv2.imwrite(out_jpg, panorama, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"\nẢnh Panorama thô: {panorama.shape[1]}×{panorama.shape[0]}px → {out_jpg}")

    save_panorama_vis(panorama, os.path.join(OUTPUT_DIR, "surf_panorama_vis.jpg"))
    print(f"\nDone, KQ (Keypoints, Matches, Panorama) đã lưu tại: {OUTPUT_DIR}/")