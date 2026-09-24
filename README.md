# 🎭 FER-Video-Emotion-Recognition

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-FF5500.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end deep learning project for **Facial Emotion Recognition (FER) from Short Videos** using the RAVDESS dataset. Built for the course *Công nghệ phần mềm nâng cao (Advanced Software Engineering)*.

---

## 📌 1. Project Overview & Objectives

Facial Emotion Recognition in video streams requires capturing both **spatial information** (facial keypoints, expressions, geometry) and **temporal dynamics** (expression transitions over time).

This project implements a complete pipeline:
- **Preprocessing:** Uniform 16-frame sampling, OpenCV Haar Cascade face detection with 10% bounding box padding, RGB conversion, and ImageNet tensor normalization.
- **Deep Learning Model:** A spatial-temporal architecture (`SpatialTemporalFERModel`) combining a **ResNet-18 2D CNN backbone** for frame feature extraction with a **2-layer Bidirectional LSTM** and **Temporal Global Average Pooling** for sequence classification.
- **Web Application:** An interactive **Gradio** web interface (`app.py` & `04_app_demo.ipynb`) enabling short video uploads, real-time emotion prediction, confidence scoring, and full 8-class probability visualization.

---

## 👥 2. Team Members & Roles

* **Cao Huỳnh Minh Quân** (Team Leader) — Model Architecture & Deep Learning Training (`model_training.ipynb`)
* **Phạm Nhật Huy** — Data Preprocessing & Pipeline Engineering (`data_preprocessing.ipynb`)
* **Phan Công Thành** — Gradio Web App, Application Integration, Technical Documentation & Presentation Lead (`04_app_demo.ipynb`, `app.py`, `src/`)
* **Nguyễn Thành Trung** — Model Evaluation, Metrics & Error Analysis (`model_evaluation.ipynb`)

---

## 📊 3. Dataset & 8 Emotion Classes

The project uses the **RAVDESS** (Ryerson Audio-Visual Emotional Speech and Song) dataset containing 2,880 short video files across 24 professional actors (12 female, 12 male).

The model classifies 8 standardized facial emotion categories:
1. **Neutral** (0)
2. **Calm** (1)
3. **Happy** (2)
4. **Sad** (3)
5. **Angry** (4)
6. **Fearful** (5)
7. **Disgust** (6)
8. **Surprised** (7)

---

## 🔄 4. Preprocessing & Model Architecture

```
Raw MP4 Video ---> Uniform 16-Frame Sampling ---> Haar Cascade Face Crop (10% pad) 
              ---> RGB Conversion & 224x224 Resize ---> ImageNet Tensor Normalization
              ---> ResNet-18 Backbone (512-dim) ---> 2-layer BiLSTM (256 hidden)
              ---> Temporal GAP ---> MLP Classifier ---> 8-Class Softmax Probabilities
```

- **Frame Sampling:** `sequence_length = 16` uniformly sampled across video duration using `np.linspace`.
- **Face Crop:** OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`) selecting largest face bounding box with 10% bounding box expansion. Fallback to full frame if face is not detected.
- **Backbone:** ResNet-18 (`weights=None` when loading checkpoint, pre-trained on ImageNet during initial training).
- **Recurrent Head:** 2-layer Bidirectional LSTM (`hidden_dim=256`, `dropout=0.5`).
- **Temporal Pooling:** Mean reduction across the 16 time steps.
- **Classifier MLP:** `Linear(512, 128) -> ReLU -> Dropout(0.5) -> Linear(128, 8)`.

---

## 📂 5. Repository Structure

```plaintext
FER-Video-Emotion-Recognition/
├── 04_app_demo.ipynb            # Interactive Gradio demo notebook (Colab & Local)
├── data_preprocessing.ipynb     # Authoritative data preprocessing pipeline notebook
├── model_training.ipynb         # Deep learning model definition & training loop
├── model_evaluation.ipynb       # Evaluation metrics, confusion matrix & validation
├── demo.ipynb                   # Entry point notebook linking to 04_app_demo.ipynb
├── app.py                       # Standalone Gradio Web Application entry point
├── requirements.txt             # Project Python dependencies
├── .gitignore                   # Git exclusion rules
├── README.md                    # Project documentation
├── src/                         # Modular Python package
│   ├── __init__.py
│   ├── video_preprocessing.py   # Frame sampling & face detection logic
│   └── inference.py             # PyTorch model definition & EmotionPredictor class
├── docs/                        # Technical documentation & presentation assets
│   ├── APP_DEMO_AND_INTEGRATION.md
│   ├── report_contribution.md
│   └── presentation_outline.md
└── checkpoints/                 # Model checkpoints (place best_model.pth here)
    ├── best_model.pth (gitignored)
    └── model_config.json
```

---

## ⚙️ 6. Google Drive & Dataset Layout

When running in **Google Colab**, place dataset files in Google Drive under:

```plaintext
/content/drive/MyDrive/Công nghệ phần mềm nâng cao/Project_FER_Video/
├── ravdess.zip (23.86 GB raw dataset archive)
├── processed_data/
│   └── ravdess_batches/
│       ├── batch_001.npz
│       └── ... (batch_001 to batch_029)
└── checkpoints/
    ├── best_model.pth
    ├── model_config.json
    └── training_history.json
```

---

## 🚀 7. Installation & Quick Start

### Step 1: Clone Repository
```bash
git clone https://github.com/Wuanzz/FER-Video-Emotion-Recognition.git
cd FER-Video-Emotion-Recognition
```

### Step 2: Set Up Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### Step 3: Checkpoint Placement
Ensure `best_model.pth` and `model_config.json` exist in the `checkpoints/` folder:
- Expected checkpoint location: `checkpoints/best_model.pth`
- Config JSON location: `checkpoints/model_config.json`

---

## 🌐 8. Running the Gradio Web Application

### Option A: Standalone Python Script
```bash
python app.py
```
Open the printed local URL (e.g. `http://127.0.0.1:7860`) in your web browser.

### Option B: Jupyter Notebook / Google Colab
Open `04_app_demo.ipynb` in Google Colab or Jupyter Lab, execute cells sequentially, and run `launch_gradio_app()`.

---

## 💡 9. Using the Web Demo

1. **Upload Video:** Drag and drop or browse for a short video file (`.mp4`, `.avi`, `.mov`).
2. **Review Settings:** Verify that the checkpoint path is set to `checkpoints/best_model.pth`.
3. **Click "Predict Emotion":** The system will sample 16 frames, detect faces, run PyTorch inference, and display:
   - **Predicted Emotion:** Top class name (e.g. `HAPPY`).
   - **Confidence Score:** Percentage confidence (e.g. `92.45%`).
   - **Probability Distribution:** Interactive bar chart showing scores for all 8 emotions.

---

## 📈 10. Experimental Results Summary

- **Training Epochs:** 30 Epochs
- **Best Validation Accuracy:** 85.79% *(obtained at Epoch 5)*
- **Test Accuracy:** `[To be filled after final evaluation run]`
- **Training History Curves:** Saved in `checkpoints/training_curves.png`.

---

## ⚠️ 11. Known Limitations & Troubleshooting

- **Missing Checkpoint Message:** If `best_model.pth` is missing, the web app will display a setup message guiding you to place `best_model.pth` into `checkpoints/`.
- **Face Detection Fallback:** If a video contains no clear frontal face, the preprocessing pipeline safely falls back to processing the full frame.
- **Device Support:** Inference automatically utilizes CUDA GPU if available, falling back gracefully to CPU.

---

## 📄 12. License & Acknowledgments

This project is licensed under the MIT License. RAVDESS dataset is publicly available for academic research.
