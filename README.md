# 🎭 Video Facial Emotion Recognition

A deep learning project for recognizing human facial emotions from short video clips using a spatial-temporal deep learning architecture.

The system processes a video by extracting representative frames, detecting and cropping the face, extracting spatial features using ResNet18, modeling temporal information using a Bidirectional LSTM, and predicting one of eight emotional classes.

The project also provides a **Gradio Web Demo** that allows users to upload a short video and receive the predicted emotion, confidence score, and probability distribution.

---

## 1. Project Overview

Facial expressions contain important information about a person's emotional state. Unlike static image-based emotion recognition, video-based emotion recognition needs to consider both:

* **Spatial information:** facial features in each individual frame.
* **Temporal information:** how facial expressions change across frames.

Therefore, this project combines a CNN-based feature extractor with a recurrent neural network to process the spatial and temporal characteristics of facial expressions.

### Main workflow

```text
Input Video
     ↓
Read Video
     ↓
Sample 16 Frames
     ↓
Face Detection
     ↓
Face Cropping
     ↓
Resize to 224 × 224
     ↓
ResNet18 Feature Extraction
     ↓
Bidirectional LSTM
     ↓
Temporal Feature Aggregation
     ↓
Fully Connected Classifier
     ↓
8 Emotion Classes
```

---

## 2. Dataset

### RAVDESS

The project uses the **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)**.

The dataset contains video recordings representing eight different emotional states.

### Dataset statistics

| Property                 | Value     |
| ------------------------ | --------- |
| Dataset                  | RAVDESS   |
| Number of videos         | 2,880     |
| Number of classes        | 8         |
| Input type               | Video     |
| Number of sampled frames | 16        |
| Image size               | 224 × 224 |
| Color format             | RGB       |

### Emotion classes

The system recognizes the following eight emotions:

1. Neutral
2. Calm
3. Happy
4. Sad
5. Angry
6. Fearful
7. Disgust
8. Surprised

---

## 3. Data Preprocessing

The preprocessing pipeline is designed to convert raw video files into a fixed-length sequence of facial images.

### Step 1 — Read the video

OpenCV is used to read the input video frame by frame.

### Step 2 — Sample frames

Because videos can contain different numbers of frames, the system samples a fixed number of frames.

The project uses:

```text
Sequence Length = 16 frames
```

The frames are sampled evenly throughout the video using `numpy.linspace`.

This allows videos with different durations to be converted into sequences with the same temporal length.

### Step 3 — Face Detection

The project uses the OpenCV Haar Cascade face detector:

```text
haarcascade_frontalface_default.xml
```

For each sampled frame:

1. Convert the frame to grayscale.
2. Detect faces.
3. Select the largest detected face.
4. Add approximately 10% padding around the detected face.
5. Crop the face region.

If no face is detected, the original full frame is used.

### Step 4 — Convert to RGB

OpenCV reads images in BGR format.

Therefore, the cropped image is converted from:

```text
BGR → RGB
```

### Step 5 — Resize

Each processed frame is resized to:

```text
224 × 224 pixels
```

The final preprocessing output has the shape:

```text
(16, 224, 224, 3)
```

---

## 4. Data Transformation

Before being passed to the neural network, the images are converted into PyTorch tensors and normalized using ImageNet normalization.

```text
Mean = [0.485, 0.456, 0.406]

Std = [0.229, 0.224, 0.225]
```

The final tensor format used by the model is:

```text
(Batch, Sequence, Channels, Height, Width)
```

For one video:

```text
(1, 16, 3, 224, 224)
```

---

## 5. Model Architecture

The proposed model is a spatial-temporal architecture consisting of:

```text
ResNet18
     ↓
Frame-level Feature Extraction
     ↓
Bidirectional LSTM
     ↓
Temporal Mean Pooling
     ↓
Fully Connected Classifier
     ↓
8 Emotion Classes
```

### 5.1 ResNet18

ResNet18 is used as the spatial feature extractor.

For every frame:

```text
224 × 224 × 3
        ↓
     ResNet18
        ↓
512-dimensional feature vector
```

The final classification layer of ResNet18 is removed so that the network can be used as a feature extractor.

---

### 5.2 Bidirectional LSTM

The 16 frame-level feature vectors are passed sequentially into a two-layer Bidirectional LSTM.

Configuration:

| Parameter        |         Value |
| ---------------- | ------------: |
| Number of layers |             2 |
| Hidden size      |           256 |
| Direction        | Bidirectional |
| Dropout          |           0.5 |

Because the LSTM is bidirectional, it can capture temporal information from both directions of the sequence.

The output feature dimension is:

```text
256 × 2 = 512
```

---

### 5.3 Temporal Aggregation

After the Bidirectional LSTM, temporal mean pooling is applied to aggregate information from all 16 frames.

```text
16 × 512
     ↓
Mean Pooling
     ↓
512
```

---

### 5.4 Classification Head

The aggregated feature is passed through a fully connected classification head:

```text
512
 ↓
Linear(512 → 128)
 ↓
ReLU
 ↓
Dropout(0.5)
 ↓
Linear(128 → 8)
 ↓
8 Emotion Classes
```

---

## 6. Complete Model Architecture

```text
Input Video
    │
    ├── 16 sampled frames
    │
    ▼
Face Detection & Cropping
    │
    ▼
224 × 224 RGB Frames
    │
    ▼
ResNet18
    │
    ▼
512-D Feature per Frame
    │
    ▼
2-Layer Bidirectional LSTM
    │
    ▼
512-D Temporal Features
    │
    ▼
Temporal Mean Pooling
    │
    ▼
Linear 512 → 128
    │
    ▼
ReLU
    │
    ▼
Dropout 0.5
    │
    ▼
Linear 128 → 8
    │
    ▼
8 Emotion Classes
```

---

## 7. Training Configuration

The model was trained using the following configuration:

| Parameter               |             Value |
| ----------------------- | ----------------: |
| Batch size              |                16 |
| Learning rate           |            0.0001 |
| Weight decay            |            0.0001 |
| Number of epochs        |                30 |
| Optimizer               |             AdamW |
| Loss function           |  CrossEntropyLoss |
| Learning-rate scheduler | ReduceLROnPlateau |
| Dropout                 |               0.5 |
| Sequence length         |                16 |
| Image size              |         224 × 224 |

### Optimizer

The project uses AdamW:

```text
AdamW
Learning Rate = 1e-4
Weight Decay = 1e-4
```

### Loss Function

Since this is a multi-class classification problem, Cross Entropy Loss is used:

```text
CrossEntropyLoss
```

### Learning Rate Scheduler

The training process uses:

```text
ReduceLROnPlateau
```

to adjust the learning rate when validation performance stops improving.

---

## 8. Model Checkpoint

The trained model is saved as:

```text
best_model.pth
```

The checkpoint contains:

* Model state dictionary
* Optimizer state dictionary
* Training epoch
* Validation accuracy
* Model configuration

The demo loads the trained model from:

```text
checkpoints/best_model.pth
```

---

## 9. Evaluation

The trained model was evaluated on a test set containing:

```text
500 samples
```

The direct evaluation results from the evaluation notebook are:

| Metric            |     Result |
| ----------------- | ---------: |
| Test Accuracy     | **72.60%** |
| Macro F1-score    | **72.45%** |
| Weighted F1-score | **72.29%** |

### Evaluation Metrics

#### Accuracy

Accuracy measures the percentage of correctly classified samples.

```text
Accuracy = Correct Predictions / Total Predictions
```

The obtained test accuracy was:

```text
72.60%
```

#### Macro F1-score

Macro F1 calculates the F1-score independently for each emotion and then takes the average.

This provides a general view of performance across the eight classes.

```text
Macro F1 = 72.45%
```

#### Weighted F1-score

Weighted F1 takes class support into account when calculating the average.

```text
Weighted F1 = 72.29%
```

---

## 10. Web Demo

The project provides a Web Demo using **Gradio**.

Users can upload a short video directly through the web interface.

### Demo workflow

```text
Upload Video
     ↓
Click "Predict Emotion"
     ↓
Video Preprocessing
     ↓
Face Detection
     ↓
Frame Sampling
     ↓
Model Prediction
     ↓
Softmax Probability
     ↓
Display Result
```

### Demo outputs

The Web Demo displays:

* Predicted emotion
* Prediction confidence
* Probability of all 8 emotions
* Probability distribution chart

Example:

```text
Predicted Emotion: HAPPY

Confidence: 82.35%
```

The probability chart displays the predicted probabilities for:

```text
Neutral
Calm
Happy
Sad
Angry
Fearful
Disgust
Surprised
```

---

## 11. Technologies

The project uses:

* Python
* PyTorch
* Torchvision
* OpenCV
* NumPy
* Matplotlib
* Gradio
* Google Colab
* Google Drive

### Deep Learning

```text
PyTorch
ResNet18
Bidirectional LSTM
```

### Computer Vision

```text
OpenCV
Haar Cascade
```

### Web Demo

```text
Gradio
```

---

## 12. Project Structure

A simplified project structure is:

```text
Project_FER_Video/
│
├── checkpoints/
│   └── best_model.pth
│
├── data/
│
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   ├── model_evaluation.ipynb
│   └── demo.ipynb
│
├── README.md
└── ...
```

> `04_app_demo.ipynb` is not used as the final demo notebook. The final implementation is maintained in `demo.ipynb`.

---

## 13. Demo Notebook

The final demo notebook performs the following steps:

```text
1. Install dependencies
2. Import libraries
3. Mount Google Drive
4. Configure model path
5. Initialize Haar Cascade
6. Define video preprocessing
7. Test video preprocessing
8. Define model architecture
9. Load best_model.pth
10. Define image transformation
11. Convert frames to tensors
12. Define prediction function
13. Test model prediction
14. Generate probability chart
15. Define Gradio prediction function
16. Create Gradio interface
17. Launch Web Demo
```

---

## 14. How to Run the Demo

### Step 1 — Open Google Colab

Open the final:

```text
demo.ipynb
```

### Step 2 — Mount Google Drive

Make sure the model checkpoint is available at:

```text
/content/drive/MyDrive/Công nghệ phần mềm nâng cao/Project_FER_Video/checkpoints/best_model.pth
```

### Step 3 — Run the notebook

Run the cells in order from top to bottom.

### Step 4 — Launch Gradio

Run the final cell:

```python
demo.launch(share=True, debug=True)
```

### Step 5 — Upload a video

Upload a short video containing a visible human face.

### Step 6 — Predict

Click:

```text
Predict Emotion
```

The system will return the predicted emotion, confidence score, and probability distribution.

---

## 15. Limitations

Although the model achieves promising results, several limitations remain.

### 15.1 Face Detection

The system uses Haar Cascade for face detection.

Performance may decrease when:

* The face is partially occluded.
* The face is too small.
* The face is rotated significantly.
* Lighting conditions are poor.

### 15.2 Dataset Characteristics

The RAVDESS dataset contains relatively controlled recording conditions.

Real-world videos can contain:

* Different lighting conditions
* Different camera angles
* Background objects
* Multiple people
* Motion blur
* Different facial expressions

Therefore, performance on real-world videos may differ from the test-set performance.

### 15.3 Temporal Information

The system uses 16 sampled frames per video.

Very fast or subtle emotional changes may not be fully represented by these sampled frames.

---

## 16. Future Improvements

Potential improvements include:

* Using a stronger face detector.
* Increasing the diversity of training data.
* Applying data augmentation.
* Testing different CNN backbones.
* Comparing LSTM with GRU or Transformer-based temporal models.
* Improving handling of multiple faces.
* Supporting longer videos.
* Optimizing inference speed.
* Deploying the demo as a standalone web application.

---

## 17. Conclusion

This project implements a video-based facial emotion recognition system using a spatial-temporal deep learning architecture.

The system combines:

```text
ResNet18
+
Bidirectional LSTM
```

to extract facial features from individual frames and model temporal changes across the video.

The final system recognizes eight emotions and provides a Web Demo through Gradio.

The direct evaluation on the test set achieved:

```text
Accuracy:       72.60%
Macro F1-score: 72.45%
Weighted F1:    72.29%
```

The project demonstrates how computer vision and deep learning can be combined to recognize emotional expressions from video sequences.

---

## 18. Team

### Project Team

* Cao Huỳnh Minh Quân
* Phạm Nhật Huy
* Phan Công Thành
* Nguyễn Thành Trung

---

## 19. Keywords

```text
Facial Emotion Recognition
Video Emotion Recognition
RAVDESS
Deep Learning
Computer Vision
ResNet18
BiLSTM
LSTM
PyTorch
OpenCV
Haar Cascade
Gradio
```
