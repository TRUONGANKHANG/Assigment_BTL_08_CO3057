# BÀI TẬP LỚN MÔN XỬ LÝ ẢNH SỐ VÀ THỊ GIÁC MÁY TÍNH (CO3057)

## Lớp: LO1 --- Nhóm BTL_08 --- GVHD: ThS Võ Thanh Hùng

<div align="center">

| STT |  MSSV   |    Họ và tên     |
| :-: | :-----: | :--------------: |
|  1  | 2213848 |  Nguyễn Minh Tú  |
|  2  | 2211476 | Trương An Khang  |
|  3  | 2211873 | Dương Hoàng Long |
|  4  | 2211756 |   Lê Tuấn Kiệt   |

</div>

---

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-orange)
![NumPy](https://img.shields.io/badge/Numpy-Numerical-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](LICENSE)

## TỔNG QUAN

Môn học **Xử lý ảnh số và Thị giác máy tính (Digital Image Processing and Computer Vision)** tập trung vào các phương pháp và thuật toán cho phép máy tính thu nhận, phân tích và hiểu thông tin từ hình ảnh. Nội dung môn học bao gồm các kỹ thuật cơ bản trong xử lý ảnh như lọc ảnh, biến đổi ảnh, phát hiện biên, cũng như các phương pháp nâng cao trong thị giác máy tính như phát hiện đặc trưng, ghép ảnh và phân tích cấu trúc hình ảnh.

Mục tiêu của các bài tập là giúp sinh viên hiểu và vận dụng các kỹ thuật cơ bản và nâng cao trong xử lý ảnh và thị giác máy tính thông qua việc hiện thực các thuật toán, trực quan hóa kết quả và phân tích hiệu quả của các phương pháp.

Thông qua bài tập lớn, sinh viên được tiếp cận với nhiều chủ đề quan trọng trong Computer Vision, bao gồm biểu diễn ảnh, lọc ảnh, chỉnh sửa ảnh trong miền gradient, biến đổi hình học, ghép ảnh toàn cảnh và xây dựng pipeline phân tích cảnh. Các bài tập này giúp củng cố kiến thức lý thuyết đồng thời rèn luyện kỹ năng lập trình và đánh giá kết quả thực nghiệm.

📋 **Kiến thức và kỹ năng đạt được:**

Sau khi hoàn thành các bài tập lớn, sinh viên có thể nắm được các nội dung chính sau:

- Biểu diễn ảnh số, mối quan hệ giữa ảnh màu và ảnh xám, cũng như vai trò của các kênh màu RGB.

- Các kỹ thuật lọc ảnh trong miền không gian như low-pass filtering và high-pass filtering nhằm làm trơn ảnh, giảm nhiễu và làm nổi bật chi tiết.

- Các phương pháp Gradient Domain Editing, đặc biệt là Poisson Image Editing, cho phép ghép ảnh và chỉnh sửa ảnh một cách tự nhiên dựa trên trường gradient.

- Các phép biến đổi hình học trong Computer Vision như translation, rotation, scaling, affine transformation và projective transformation.

- Bài toán ghép ảnh toàn cảnh (image stitching) và quy trình feature-based stitching, bao gồm trích xuất đặc trưng, so khớp và ước lượng homography.

- So sánh và đánh giá các phương pháp trích xuất đặc trưng (SIFT, SURF, ORB) dựa trên chất lượng ghép ảnh.

- Thiết kế và hiện thực một pipeline Computer Vision hoàn chỉnh, kết hợp nhiều mô-đun từ tiền xử lý, phân tích hình học đến phân tích nội dung và trực quan hóa kết quả.

- Áp dụng các kỹ thuật phân tích hình học nâng cao như phát hiện biên (Canny), đường thẳng (Hough Transform), trích xuất đặc trưng (ORB, Harris) và ước lượng phép biến đổi bằng RANSAC.

- Triển khai các phương pháp phân tích nội dung ảnh, bao gồm phát hiện đối tượng bằng mô hình học sâu (YOLOv8) và phân vùng ảnh cơ bản bằng K-Means.

- Gán nhãn ngữ nghĩa cho các vùng trong ảnh (sky, water, land) và phân tích cấu trúc cảnh dựa trên bố cục không gian.

- Ước lượng các đặc trưng quan trọng của cảnh như đường chân trời (horizon detection) và đánh giá mức độ phản chiếu bằng chỉ số SSIM.

- Phát triển kỹ năng trực quan hóa kết quả xử lý ảnh, bao gồm hiển thị bounding boxes, segmentation masks, đường chân trời và tạo ảnh so sánh giữa các bước trong pipeline.

- Rèn luyện khả năng phân tích và đánh giá hệ thống Computer Vision thông qua cả định tính và định lượng, đồng thời nhận diện các hạn chế và đề xuất hướng cải tiến.

---

## Assignment 1 - Biểu diễn ảnh màu và lọc tín hiệu

### Các nhiệm vụ cần hoàn thành

1. **Biểu diễn ảnh màu và ảnh xám**
   - Load một hoặc nhiều ảnh màu (RGB).
   - Chuyển đổi ảnh màu sang ảnh xám.
   - Tách riêng từng kênh màu **Red, Green, Blue** từ ảnh gốc.
   - Hiển thị mỗi kênh màu dưới dạng ảnh xám.
   - Kết hợp lại các kênh màu để:
     - Tái tạo ảnh màu ban đầu.
     - Tạo các ảnh mới bằng cách hoán đổi hoặc thay thế các kênh màu.

2. **Áp dụng các kỹ thuật lọc ảnh**
   - Thực hiện **low-pass filtering** để làm trơn ảnh và giảm nhiễu.
   - Thực hiện **high-pass filtering** để làm nổi bật biên và chi tiết ảnh.
   - Thử nghiệm với các kernel phổ biến như:
     - Mean filter
     - Gaussian filter
     - Laplacian filter
     - Sobel filter

3. **Phân tích và đánh giá kết quả**
   - So sánh trực quan giữa ảnh gốc và ảnh sau khi xử lý.
   - Phân tích ảnh hưởng của các phép lọc đến chất lượng ảnh.
   - Nhận xét sự khác biệt khi áp dụng các bộ lọc trên các loại ảnh khác nhau (ảnh nhiều chi tiết, ảnh ít chi tiết, ảnh có nhiễu).

Thông qua bài tập này, sinh viên hiểu rõ cách biểu diễn ảnh số, vai trò của các kênh màu và tác động của các phép lọc ảnh trong xử lý ảnh số.

---

## Assignment 2 - Gradient Domain Editing và biến đổi hình học

### Các nhiệm vụ cần hoàn thành

1. **Gradient Domain Editing (Poisson Image Editing)**
   - Chọn hai ảnh đầu vào:
     - Ảnh nền (background image).
     - Ảnh chứa đối tượng cần ghép (source image).
   - Xác định vùng đối tượng cần ghép từ ảnh nguồn (có thể sử dụng mask thủ công).
   - Hiện thực kỹ thuật **Gradient Domain Editing** để ghép đối tượng từ ảnh nguồn vào ảnh nền.
   - Đảm bảo kết quả ghép ảnh tự nhiên về biên và ánh sáng.
   - So sánh kết quả với phương pháp **ghép ảnh trực tiếp (direct paste)**.

2. **Geometric Transformations**
   - Xây dựng các chương trình minh họa cho các phép biến đổi hình học:
     - Translation
     - Rotation
     - Scaling
   - Hiện thực các phép biến đổi nâng cao:
     - **Affine Transformation**
     - **Projective Transformation (Homography)**
   - Trực quan hóa kết quả của từng phép biến đổi trên cùng một ảnh đầu vào.

3. **Thử nghiệm mở rộng (khuyến khích)**
   - Dán một hình chữ nhật hoặc hình vuông lên một mặt phẳng trong ảnh nền.
   - Xác định vùng mặt phẳng đích (ví dụ: biển quảng cáo, màn hình hoặc bức tường).
   - Sử dụng **projective transformation** để ánh xạ nội dung vào mặt phẳng đó.
   - So sánh giữa cách xác định điểm tương ứng **thủ công** và **tự động** (nếu có).

4. **Phân tích và đánh giá**
   - Phân tích chất lượng ghép ảnh bằng Gradient Domain Editing.
   - So sánh với phương pháp ghép ảnh trực tiếp.
   - Đánh giá sự khác nhau giữa **affine transformation** và **projective transformation** thông qua các ví dụ trực quan.

## Assignment 3 - Ghép ảnh toàn cảnh (Image Stitching)

### Các nhiệm vụ cần hoàn thành

1. **Xây dựng pipeline ghép ảnh**
   - Load một tập gồm 3–4 ảnh chụp cùng một cảnh với các góc nhìn khác nhau.
   - Thực hiện tiền xử lý ảnh:
     - Chuyển ảnh sang ảnh xám.
     - Chuẩn hóa kích thước ảnh (nếu cần).
   - Trích xuất đặc trưng từ ảnh:
     - Sử dụng các phương pháp như **SIFT, SURF, ORB**.
   - Thực hiện so khớp đặc trưng giữa các cặp ảnh.
   - Ước lượng phép biến đổi hình học:
     - Tính toán ma trận **Homography**.
   - Căn chỉnh và ghép ảnh:
     - Warp ảnh về cùng hệ tọa độ.
     - Ghép các ảnh để tạo panorama.
   - Xử lý vùng chồng lấn:
     - Áp dụng các kỹ thuật blending cơ bản.

2. **So sánh các phương pháp trích xuất đặc trưng**
   - Áp dụng cùng một pipeline cho ít nhất hai phương pháp khác nhau:
     - SIFT
     - SURF
     - ORB
   - Đảm bảo điều kiện thực nghiệm giống nhau:
     - Cùng tập ảnh đầu vào
     - Cùng tham số matching và RANSAC
   - So sánh kết quả dựa trên:
     - Độ chính xác căn chỉnh (alignment)
     - Tính liên tục của ảnh (seamlessness)
     - Chất lượng trực quan của panorama

3. **Đánh giá chất lượng ghép ảnh**
   - So sánh trực quan giữa ảnh panorama và ảnh gốc.
   - Phân tích các lỗi thường gặp:
     - Ghosting (nhân đôi chi tiết)
     - Misalignment (lệch hình)
     - Distortion (méo hình)
   - Nhận xét ảnh hưởng của:
     - Chất lượng feature matching
     - Phân bố keypoints
     - Phương pháp blending

4. **Phân tích và nhận xét**
   - Phân tích vai trò của từng bước trong pipeline ghép ảnh.
   - So sánh ưu và nhược điểm của các phương pháp:
     - SIFT (độ chính xác cao)
     - SURF (cân bằng giữa tốc độ và chất lượng)
     - ORB (tốc độ nhanh, phù hợp thời gian thực)
   - Nhận xét các trường hợp:
     - Ghép ảnh thành công
     - Ghép ảnh gặp khó khăn (ít texture, thay đổi ánh sáng, góc nhìn lớn)

## Assignment 4 - Phân tích và trực quan hóa không gian cảnh

### Các nhiệm vụ cần hoàn thành

1. **Thu thập và tiền xử lý dữ liệu**
   - Lựa chọn tập dữ liệu gồm các ảnh hoặc video cảnh thực (indoor/outdoor).
   - Thực hiện các bước tiền xử lý:
     - Resize và crop ảnh về kích thước chuẩn.
     - Chuyển đổi không gian màu (BGR → RGB/Gray/HSV).
     - Giảm nhiễu bằng Gaussian Blur.
     - Cân bằng ánh sáng bằng CLAHE.
   - Đảm bảo dữ liệu đầu vào ổn định để phục vụ các bước phân tích tiếp theo.

2. **Phân tích hình học và cấu trúc cảnh**
   - Áp dụng các kỹ thuật xử lý ảnh để trích xuất đặc trưng hình học:
     - Phát hiện biên (Canny Edge Detection).
     - Phát hiện đường thẳng (Hough Transform).
     - Phát hiện điểm đặc trưng (ORB).
     - Phát hiện góc (Harris Corner).
   - Thực hiện so khớp đặc trưng giữa các ảnh.
   - Ước lượng phép biến đổi hình học (Homography) bằng RANSAC.
   - (Tuỳ chọn) Thực hiện ghép ảnh panorama từ nhiều ảnh cùng cảnh.

3. **Phân tích nội dung hoặc đối tượng**
   - Áp dụng ít nhất một kỹ thuật phân tích nội dung:
     - Phát hiện đối tượng bằng mô hình học sâu (YOLOv8).
     - Phân vùng ảnh mức cơ bản bằng K-Means.
   - Gán nhãn ngữ nghĩa cho các vùng chính:
     - Sky (bầu trời)
     - Water (mặt nước)
     - Land (đất liền/công trình)
   - Ước lượng các đặc trưng của cảnh:
     - Đường chân trời (Horizon Detection).
     - Tính toán mức độ phản chiếu (Reflection Score) bằng SSIM.
     - Tỉ lệ diện tích các vùng ngữ nghĩa.

4. **Trực quan hóa kết quả**
   - Hiển thị rõ ràng các kết quả xử lý:
     - Bounding boxes từ object detection.
     - Mask từ segmentation.
     - Đường chân trời trên ảnh.
   - Xây dựng ảnh trực quan hóa tổng hợp:
     - Overlay các lớp thông tin lên ảnh gốc.
   - Tạo ảnh so sánh giữa các bước xử lý:
     - Original
     - Detection
     - Segmentation + Horizon
     - Final Visualization

5. **Thử nghiệm mở rộng (khuyến khích)**
   - So sánh nhiều phương pháp segmentation khác nhau.
   - Thử nghiệm với nhiều loại cảnh khác nhau (ánh sáng, thời tiết).
   - Tối ưu pipeline để tăng tốc độ xử lý.
   - Áp dụng pipeline cho video thay vì ảnh tĩnh.

6. **Phân tích và đánh giá**
   - Đánh giá kết quả phát hiện đối tượng (số lượng, độ tin cậy).
   - Phân tích chất lượng segmentation (nhiễu, sai lệch biên).
   - Đánh giá độ chính xác của đường chân trời.
   - So sánh trực quan giữa các bước xử lý.
   - Phân tích các trường hợp thành công và thất bại của hệ thống.
---

## Project Structure

```
ASSIGMENT_BTL_08_CO3057
├── ASS1/
│   ├── output/
│   ├── sample/
│   ├── High-pass.ipynb
│   ├── Low-pass.ipynb
│   └── RGB.ipynb
├── ASS2/
│   ├── input/
│   ├── output/
│   ├── GradientDomain.ipynb
│   ├── GraDomainCopy-Paste.ipynb
│   ├── homography.ipynb
│   └── tranform.ipynb
├── ASS3/
│   ├── input/
│   ├── ORB/
│   ├── SIFT/
│   └── SURF/
├── ASS4/
│   ├── input/
│   ├── preprocessing
│   ├── pipeline.ipynb
│   └── ouput/
└── README.md

```

## Hướng dẫn chạy

### Điều kiện

Trước khi chạy project, cần cài đặt các thư viện sau:

- Python >= 3.12
- NumPy
- OpenCV
- Matplotlib
- SciPy
- scikit-image
- Jupyter Notebook

### Cài đặt

Clone repository:

```bash
git clone https://github.com/TRUONGANKHANG/Assigment_BTL_08_CO3057.git
cd Assigment_BTL_08_CO3057
```

## License

Dự án này được phát triển cho mục đích học thuật trong môn học **Xử lý ảnh số và Thị giác máy tính (CO3057)**.

**Authors**: Nguyễn Minh Tú, Trương An Khang, Dương Hoàng Long, Lê Tuấn Kiệt

**Institution**: Faculty of Computer Science and Engineering, Ho Chi Minh City University of Technology, VNU-HCM
