# Presentation Outline & Demo Script

**Project Title:** RAVDESS Video Emotion Recognition (FER)  
**Course:** Công nghệ phần mềm nâng cao  
**Presenter for Demo & Application Section:** Phan Công Thành  

---

## I. Slide Deck Outline (Cấu Trúc Slide Thuyết Trình)

### Slide 1: Title & Team Introduction (Trang Tiêu Đề & Thành Viên)
- **Title:** Nhận diện Cảm xúc Khuôn mặt từ Video Ngắn (Facial Emotion Recognition from Short Videos)
- **Course:** Công nghệ phần mềm nâng cao
- **Team Members & Responsibilities:**
  - Cao Huỳnh Minh Quân (Group Leader - Model Architecture & Training)
  - Phạm Nhật Huy (Data Preprocessing & Feature Engineering)
  - Phan Công Thành (Gradio Web App, Integration, README & Technical Documentation)
  - Nguyễn Thành Trung (Model Evaluation, Metrics & Error Analysis)

### Slide 2: Problem Statement & Objectives (Đặt Vấn Đề & Mục Tiêu)
- **Context:** Nhận diện cảm xúc qua chuỗi video đóng vai trò quan trọng trong giao tiếp người - máy (HCI), hỗ trợ y tế, giáo dục trực tuyến và phân tích tâm lý.
- **Challenge:** Video chứa yếu tố không gian (khuôn mặt, ánh mắt, biểu cảm) và yếu tố thời gian (sự biến đổi cảm xúc theo chuỗi khung hình).
- **Objective:** Xây dựng hệ thống học sâu kết hợp 2D CNN (ResNet-18) và Recurrent Neural Network (BiLSTM) để phân loại 8 cảm xúc từ video ngắn RAVDESS.

### Slide 3: Dataset Overview & 8 Emotion Classes (Bộ Dữ Liệu RAVDESS)
- **Dataset:** RAVDESS (Ryerson Audio-Visual Emotional Speech and Song).
- **Format:** 2,880 video ngắn chuẩn hóa phát âm và hát của 24 diễn viên (12 nam, 12 nữ).
- **8 Emotion Classes:**
  1. Neutral (Bình thường) | 2. Calm (Bình tĩnh) | 3. Happy (Vui vẻ) | 4. Sad (Buồn rầu)
  5. Angry (Tức giận) | 6. Fearful (E sợ) | 7. Disgust (Chán ghét) | 8. Surprised (Bất ngờ)

### Slide 4: Data Preprocessing Pipeline (Quy Trình Tiền Xử Lý Dữ Liệu)
- **Step 1 — Frame Sampling:** Lấy 16 khung hình đại diện theo phân bố đều (`np.linspace`).
- **Step 2 — Face Detection:** Sử dụng OpenCV Haar Cascade để phát hiện khuôn mặt.
- **Step 3 — Padding & Bounding Box:** Bổ sung lề 10% (10% padding) để không mất góc mặt.
- **Step 4 — Normalization:** Chuyển RGB, resize `(224, 224)` và chuẩn hóa ImageNet (`mean`, `std`).
- *[Diagram Placeholder: Sơ đồ luồng tiền xử lý từ Video -> 16 Cropped Face Frames]*

### Slide 5: Model Architecture — SpatialTemporalFERModel (Kiến Trúc Mô Hình)
- **Spatial Feature Extractor:** ResNet-18 Backbone (Pretrained / Finetuned) trích xuất đặc trưng 512 chiều mỗi frame.
- **Temporal Sequence Modeling:** 2-layer Bidirectional LSTM (`hidden_dim=256`), mã hóa sự thay đổi biểu cảm qua 16 bước thời gian.
- **Temporal Global Average Pooling (GAP):** Tổng hợp đặc trưng chuỗi theo trục thời gian.
- **Classifier Head:** MLP (`Linear(512, 128) -> ReLU -> Dropout(0.5) -> Linear(128, 8)`).

### Slide 6: Gradio Web Application & Integration (Ứng Dụng Web Demo)
- **Architecture:** `app.py` & `04_app_demo.ipynb` tích hợp trực tiếp `src/video_preprocessing.py` và `src/inference.py`.
- **Features:**
  - Tải video mp4/avi trực tiếp.
  - Tự động nạp checkpoint `best_model.pth`.
  - Hiển thị nhãn dự đoán, phần trăm độ tin cậy và biểu đồ xác suất 8 lớp.
  - Xử lý ngoại lệ an toàn khi thiếu weights hoặc video lỗi.
- *[Screenshot Placeholder: Giao diện Gradio Web App khi thực hiện dự đoán]*

### Slide 7: Experimental Results & Performance (Kết Quả Thực Nghiệm)
- **Training Epochs:** 30 Epochs (Optimizer: AdamW, Scheduler: ReduceLROnPlateau).
- **Validation Accuracy:** `[To be filled after evaluation - e.g. 85.79%]`
- **Test Accuracy:** `[To be filled after evaluation]`
- **Metrics Table Placeholder:**
  | Emotion Class | Precision | Recall | F1-Score |
  | ------------- | --------- | ------ | -------- |
  | Neutral       | [Metrics] | [Metrics] | [Metrics] |
  | Calm          | [Metrics] | [Metrics] | [Metrics] |
  | Happy         | [Metrics] | [Metrics] | [Metrics] |
  | Sad           | [Metrics] | [Metrics] | [Metrics] |
  | Angry         | [Metrics] | [Metrics] | [Metrics] |
  | Fearful       | [Metrics] | [Metrics] | [Metrics] |
  | Disgust       | [Metrics] | [Metrics] | [Metrics] |
  | Surprised     | [Metrics] | [Metrics] | [Metrics] |

### Slide 8: Live Demo & Conclusion (Trình Chiếu Demo & Kết Luận)
- **Live Demo:** Thực hiện chạy trực tiếp ứng dụng Gradio trên video mẫu.
- **Conclusion:** Hệ thống hoàn thiện từ tiền xử lý, huấn luyện đến đóng gói ứng dụng web.
- **Q&A Session:** Sẵn sàng trả lời câu hỏi từ Giảng viên hội đồng.

---

## II. Live Demo Script (Kịch Bản Trình Chiếu Demo)

1. **Mở đầu (30s):**
   > *"Kính thưa Thầy/Cô, sau đây em xin đại diện nhóm trình bày phân đoạn Demo ứng dụng Web Nhận diện Biểu cảm Khuôn mặt từ Video."*

2. **Thao tác 1 — Khởi chạy ứng dụng (30s):**
   > *"Ứng dụng được đóng gói trong file `app.py` hoặc chạy trực tiếp trong `04_app_demo.ipynb`. Khi khởi chạy, hệ thống tự động kiểm tra thiết bị tính toán (GPU T4 / CPU) và nạp trọng số mô hình `best_model.pth`."*

3. **Thao tác 2 — Upload Video & Dự đoán (60s):**
   > *"Em tiến hành upload 1 video mẫu từ bộ dữ liệu RAVDESS đại diện cho cảm xúc 'Happy'. Sau khi nhấn 'Predict Emotion', mô hình trích xuất 16 khung hình mặt, đưa qua ResNet-18 + BiLSTM và trả về kết quả ngay lập tức: Cảm xúc HAPPY với độ tin cậy 92.45%."*

4. **Thao tác 3 — Phân tích Biểu đồ Xác suất (30s):**
   > *"Bên cạnh kết quả chính, ứng dụng trực quan hóa biểu đồ xác suất trên toàn bộ 8 lớp cảm xúc, giúp giảng viên dễ dàng đánh giá mức độ phân biệt của mô hình."*
