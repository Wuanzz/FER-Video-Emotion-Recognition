# Application Demo & Pipeline Integration Documentation

**Project:** FER-Video-Emotion-Recognition  
**Author:** Cong Thanh (Cong Thanh - App Demo, Documentation & Presentation Lead)  
**Course:** Công nghệ phần mềm nâng cao  

---

## 1. Overview & Data Flow

This document details the architectural design and pipeline integration of the Gradio web application for 8-class facial emotion recognition on the RAVDESS video dataset.

```
+------------------+     +-------------------------------+     +----------------------------------+
|  Uploaded Video  | --> | 16-Frame Uniform Sampling     | --> | Haar Cascade Face Detection      |
|  (.mp4, .avi)    |     | (np.linspace index selection) |     | (10% padding + bounding box crop)|
+------------------+     +-------------------------------+     +----------------------------------+
                                                                                 |
                                                                                 v
+------------------+     +-------------------------------+     +----------------------------------+
| Prediction Output| <-- | SpatialTemporalFERModel       | <-- | RGB Conversion & 224x224 Resize |
| Label & Chart    |     | (ResNet-18 + BiLSTM + GAP)    |     | ImageNet Normalization           |
+------------------+     +-------------------------------+     +----------------------------------+
```

---

## 2. Integration with Notebooks

### A. Preprocessing Alignment (`data_preprocessing.ipynb`)
- **Emotion Classes (8):** `['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']`
- **Zero-indexed Mapping:** `label = emotion_code - 1` (range 0 to 7).
- **Sampling Strategy:** Extracts 16 uniformly spaced frames per video using `np.linspace(0, total_frames - 1, 16, dtype=int)`.
- **Face Detection:** OpenCV Haar Cascade (`haarcascade_frontalface_default.xml`) operating on grayscale frames with parameters `scaleFactor=1.1`, `minNeighbors=5`, `minSize=(30, 30)`.
- **Bounding Box Padding:** 10% width (`pad_x = 0.1 * w`) and 10% height (`pad_y = 0.1 * h`) padding added around detected faces with border clipping `[0, frame_width]`, `[0, frame_height]`.
- **No-Face Fallback:** Full original frame is preserved when face detection returns no candidates.
- **Decoding Fallback:** Duplicate previous valid frame if OpenCV fails to read a specific frame index.
- **Color Space & Resolution:** BGR to RGB conversion via `cv2.cvtColor`, resized to `(224, 224)`.

### B. Model & Training Alignment (`model_training.ipynb`)
- **Model Architecture Class:** `SpatialTemporalFERModel`
- **Backbone:** ResNet-18 (512-dim output feature map with `resnet.fc = nn.Identity()`).
- **Recurrent Network:** 2-layer Bidirectional LSTM (`hidden_dim=256`, `dropout=0.5`, `batch_first=True`). Output dimension per timestep: 512.
- **Temporal Pooling:** Global Average Pooling across the 16 time steps: `out_feature = torch.mean(rnn_out, dim=1)`.
- **Classifier Head:** MLP `Sequential(Linear(512, 128), ReLU(), Dropout(0.5), Linear(128, 8))`.
- **Input Tensor Normalization:** Standard ImageNet channel mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]` applied on float tensors scaled to `[0.0, 1.0]`.

---

## 3. Checkpoint & Configuration Format

The application expects checkpoint files formatted according to `model_training.ipynb`:

1. **`best_model.pth` Location:** `checkpoints/best_model.pth`
   - Checkpoint PyTorch `dict` keys:
     - `'epoch'`: Epoch number where highest validation accuracy was obtained.
     - `'model_state_dict'`: Complete model weights dictionary.
     - `'optimizer_state_dict'`: Optimizer state.
     - `'val_acc'`: Best validation accuracy score (float).
     - `'config'`: Architecture & hyperparameter dictionary.
2. **`model_config.json` Location:** `checkpoints/model_config.json`
   - JSON structure:
     ```json
     {
         "image_size": [224, 224],
         "sequence_length": 16,
         "num_classes": 8,
         "backbone": "resnet18",
         "rnn_type": "LSTM",
         "hidden_dim": 256,
         "num_rnn_layers": 2,
         "dropout": 0.5,
         "batch_size": 16,
         "learning_rate": 0.0001,
         "weight_decay": 0.0001,
         "num_epochs": 30,
         "seed": 42
     }
     ```

---

## 4. Gradio Demonstration Usage

### Launching Standalone Script:
```bash
python app.py
```

### Running in Colab Notebook:
Open `04_app_demo.ipynb`, execute Cells 1–7, and launch the Gradio server.

### Features Included:
1. Video upload box with instant preview.
2. Status & Log window displaying system notifications.
3. Top emotion prediction label and percentage confidence score.
4. Interactive bar distribution plot displaying probability percentages across all 8 RAVDESS emotions.
5. Graceful handling when model weights or checkpoints are missing (presents clear setup instructions instead of breaking).

---

## 5. Manual Setup & Known Artifact Dependencies

- **Missing Checkpoint Artifact:** The trained model weights `best_model.pth` (~170MB) must be generated by running `model_training.ipynb` on GPU (e.g. Google Colab T4) or manually placed in the `checkpoints/` directory.
- **Google Drive Mount:** When running in Google Colab, mount Google Drive to `/content/drive` so that `/content/drive/MyDrive/Công nghệ phần mềm nâng cao/Project_FER_Video` is accessible.
