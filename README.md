## 🎭 Phân Tích Biểu Cảm Con Người Từ Video Ngắn (Facial Emotion Recognition)

Đồ án môn học: **Công nghệ phần mềm nâng cao**
Đề tài nghiên cứu và phát triển hệ thống nhận diện, phân tích chuỗi biểu cảm khuôn mặt từ chuỗi khung hình video ngắn.

---

## 📌 1. Thành viên thực hiện

* **Cao Huỳnh Minh Quân** (Trưởng nhóm)
* **Phạm Nhật Huy**
* **Phan Công Thành**
* **Nguyễn Thành Trung**

---

## 🛠️ 2. Công nghệ & Thư viện sử dụng

* **Môi trường:** Google Colab (GPU T4), Python 3.x
* **Xử lý khuôn mặt & Video:** OpenCV, MediaPipe
* **Deep Learning:** PyTorch, timm (ResNet + LSTM / Transformer)
* **Dataset:** RAVDESS (Emotional Speech Video)
* **Giao diện Demo:** Gradio / Streamlit

---

## 📂 3. Cấu trúc Mã nguồn (Repository Structure)

```plaintext
FER-Video-Emotion-Recognition/
├── data_preprocessing.ipynb  # Tiền xử lý: Cắt mặt từ video bằng MediaPipe
├── model_training.ipynb      # Xây dựng & Huấn luyện mô hình Deep Learning
├── evaluation_demo.ipynb     # Đánh giá độ chính xác & Dựng Web Demo
└── README.md                 # Tài liệu hướng dẫn dự án
```
