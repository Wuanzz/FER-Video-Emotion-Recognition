"""
Video Preprocessing Module for FER (Facial Emotion Recognition).

This module strictly mirrors the preprocessing pipeline specified in `data_preprocessing.ipynb`:
1. Uniform sampling of 16 representative frames from an input video using np.linspace.
2. Face detection and cropping using OpenCV Haar Cascade Classifier (haarcascade_frontalface_default.xml).
3. 10% bounding box padding with boundary clipping.
4. Fallback to full frame if no face is detected.
5. Conversion from BGR to RGB color space.
6. Resizing to (224, 224).
7. Frame duplication fallback for unreadable/corrupted frames.
8. Conversion to PyTorch Tensor with ImageNet normalization.
"""

import cv2
import numpy as np
import torch
from torchvision import transforms


def get_face_cascade():
    """
    Loads and returns the Haar Cascade classifier for face detection.
    """
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load Haar Cascade classifier from {cascade_path}")
    return face_cascade


def process_video(video_path, sequence_length=16, image_size=(224, 224)):
    """
    Extracts and preprocesses frames from a video file following the authoritative
    specification from `data_preprocessing.ipynb`.

    Args:
        video_path (str): Path to input video file (.mp4, .avi, etc.)
        sequence_length (int): Number of frames to sample per video (default: 16)
        image_size (tuple): Target (width, height) resolution (default: (224, 224))

    Returns:
        np.ndarray or None: Processed array of shape (sequence_length, height, width, 3)
                            in uint8 RGB format, or None if video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[Error] Could not open video: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        print(f"[Error] Invalid total frames count ({total_frames}) for video: {video_path}")
        return None

    # Exact uniform frame sampling indices matching data_preprocessing.ipynb
    frame_indices = np.linspace(
        0,
        total_frames - 1,
        sequence_length,
        dtype=int
    )

    face_cascade = get_face_cascade()
    frames = []

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        success, frame = cap.read()

        if not success or frame is None:
            # Fallback for failed frame decoding: duplicate previous frame if available
            if len(frames) > 0:
                frames.append(frames[-1].copy())
            continue

        # Face detection on grayscale frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) > 0:
            # Select bounding box with maximum area
            x, y, w, h = max(
                faces,
                key=lambda rect: rect[2] * rect[3]
            )

            # 10% bounding box padding logic
            pad_x = int(0.1 * w)
            pad_y = int(0.1 * h)

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(frame.shape[1], x + w + pad_x)
            y2 = min(frame.shape[0], y + h + pad_y)

            face = frame[y1:y2, x1:x2]
        else:
            # Fallback when no face is detected: use whole frame
            face = frame

        # Convert color space from BGR to RGB and resize
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, image_size)

        frames.append(face)

    cap.release()

    if len(frames) == 0:
        print(f"[Error] No valid frames were extracted from {video_path}")
        return None

    # Pad with copies of the last frame if sequence length is under required length
    while len(frames) < sequence_length:
        frames.append(frames[-1].copy())

    frames = frames[:sequence_length]

    return np.array(frames, dtype=np.uint8)


def transform_frames_to_tensor(frames):
    """
    Transforms a preprocessed frame sequence (uint8 array) into a PyTorch Tensor
    with ImageNet normalization as expected by SpatialTemporalFERModel.

    Args:
        frames (np.ndarray): Array of shape (sequence_length, 224, 224, 3) in uint8 [0, 255].

    Returns:
        torch.Tensor: Tensor of shape (1, sequence_length, 3, 224, 224) with float32 values
                      normalized using ImageNet mean & std.
    """
    if isinstance(frames, np.ndarray):
        video_seq = torch.from_numpy(frames)  # (T, H, W, C)
    else:
        video_seq = frames

    # Permute to (T, C, H, W) and scale to float range [0.0, 1.0]
    if video_seq.dim() == 4 and video_seq.shape[-1] == 3:
        video_seq = video_seq.permute(0, 3, 1, 2).float() / 255.0

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    processed_frames = []
    for t in range(video_seq.size(0)):
        frame_tensor = normalize(video_seq[t])
        processed_frames.append(frame_tensor)

    # Stack to (sequence_length, 3, 224, 224)
    video_tensor = torch.stack(processed_frames, dim=0)

    # Add batch dimension -> (1, sequence_length, 3, 224, 224)
    return video_tensor.unsqueeze(0)
