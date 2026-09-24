# Báo Cáo Đóng Góp Phần Hành Demo & Tích Hợp Hệ Thống (Report Contribution)

**Thành viên:** Phan Công Thành  
**Vai trò đảm nhận:** 
- Xây dựng 04_app_demo.ipynb & app.py
- Biên soạn README.md & Tài liệu kỹ thuật
- Tích hợp mô hình học sâu vào ứng dụng Web Demo Gradio
- Chuẩn bị slide thuyết trình & Kịch bản Demo  
**Môn học:** Công nghệ phần mềm nâng cao  
**Tên dự án:** FER-Video-Emotion-Recognition  

---

## I. Tóm Tắt Đóng Góp (Executive Summary)

Trong đồ án nghiên cứu và phát triển hệ thống **Nhận diện biểu cảm khuôn mặt từ video ngắn (Facial Emotion Recognition - FER)** trên bộ dữ liệu RAVDESS, phân hành công việc của tôi tập trung vào giai đoạn đóng gói mô hình, chuẩn hóa pipeline suy luận (Inference Pipeline), xây dựng giao diện ứng dụng web tương tác trực quan (Gradio Web Demo) và hoàn thiện bộ tài liệu báo cáo dự án.

Mã nguồn được tổ chức theo kiến trúc mô-đun hóa độc lập (`src/video_preprocessing.py`, `src/inference.py`, `app.py`, `04_app_demo.ipynb`), đảm bảo tính tương thích tuyệt đối với các phân đoạn tiền xử lý (`data_preprocessing.ipynb`) và huấn luyện mô hình (`model_training.ipynb`) của các thành viên trong nhóm.

---

## II. Chi Tiết Các Hạng Mục Thực Hiện

### 1. Mô-đun Tiền Xử Lý Video Trực Tiếp (`src/video_preprocessing.py`)
- **Trích xuất khung hình đồng đều (Uniform Sampling):** Thực hiện giải thuật lấy đều 16 khung hình từ video ngắn bất kỳ bằng phương pháp tính chỉ số `np.linspace(0, total_frames - 1, 16, dtype=int)`.
- **Cắt mặt & Xử lý lề (Face Cropping with 10% Padding):** Tích hợp OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`) trích xuất vùng khuôn mặt chính có diện tích lớn nhất. Tự động thêm lề 10% chiều rộng và chiều cao nhằm giữ lại thông tin đường nét khuôn mặt, đồng thời bổ sung cơ chế fallback lấy toàn bộ khung hình khi không tìm thấy khuôn mặt.
- **Chuẩn hóa Tensor:** Chuyển đổi không gian màu BGR sang RGB, resize kích thước về `(224, 224)` và áp dụng chuẩn hóa ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

### 2. Mô-đun Suy Luận & Tích Hợp Mô Hình (`src/inference.py`)
- **Tái tạo kiến trúc SpatialTemporalFERModel:** Tương thích chính xác với mô hình ResNet-18 Backbone kết hợp Bidirectional LSTM 2 lớp (`hidden_dim=256`, `dropout=0.5`) và Temporal Global Average Pooling.
- **Nạp Checkpoint An Toàn:** Độc file trọng số `best_model.pth` và cấu hình `model_config.json`. Bổ sung cơ chế bắt lỗi nghiêm ngặt (`strict=True`) khi thiếu trọng số hoặc sai định dạng dictionary checkpoint.
- **Chế độ suy luận tối ưu:** Sử dụng `torch.inference_mode()` và tự động chuyển đổi thiết bị tính toán (CUDA GPU / CPU), giảm thiểu tối đa bộ nhớ VRAM tiêu thụ.

### 3. Giao Diện Ứng Dụng Web Gradio (`app.py` & `04_app_demo.ipynb`)
- Thiết kế giao diện khoa học, chuẩn hóa cho buổi báo cáo với giảng viên:
  - Khung tải video và xem lại trực tiếp (Video Player).
  - Khung thông tin giới thiệu 8 nhãn cảm xúc RAVDESS.
  - Hiển thị nhãn biểu cảm dự đoán có độ tin cậy cao nhất và phần trăm độ tin cậy (Confidence %).
  - Biểu đồ phân bố xác suất chi tiết (Probability Bar Plot) cho toàn bộ 8 cảm xúc.
  - Bảng log hệ thống và thông báo hướng dẫn xử lý khi thiếu file checkpoint `best_model.pth`.

---

## III. Sơ Đồ Kiến Trúc Luồng Xử Lý (System Architecture Diagram)

```
[Người dùng Upload Video] 
         │
         ▼
[04_app_demo / app.py]
         │
         ▼
[src/video_preprocessing.py] ──► (Lấy 16 frames -> Haar Cascade Cropping -> RGB 224x224 -> ImageNet Normalization)
         │
         ▼
[src/inference.py]           ──► (Nạp best_model.pth -> SpatialTemporalFERModel -> Softmax)
         │
         ▼
[Gradio Block Interface]      ──► (Hiển thị Nhãn Cảm Xúc, Confidence % & Biểu đồ Xác suất 8 lớp)
```

---

## IV. Hạn Chế & Hướng Phát Triển (Limitations & Future Work)

### 1. Hạn chế hiện tại:
- Tốc độ suy luận phụ thuộc vào phần cứng khi chạy trên CPU không có GPU hỗ trợ.
- Thuật toán Haar Cascade phát hiện khuôn mặt nhanh nhưng có thể bị hạn chế khi góc mặt nghiêng lớn hoặc điều kiện ánh sáng quá tối.

### 2. Hướng phát triển:
- Nâng cấp bộ phát hiện khuôn mặt sang MediaPipe Face Detection hoặc RetinaFace để tăng độ chính xác trong điều kiện thực tế.
- Tối ưu hóa mô hình bằng TensorRT hoặc ONNX Runtime để phục vụ thời gian thực (Real-time Video Stream).
