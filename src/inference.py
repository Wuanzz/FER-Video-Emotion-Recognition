"""
Inference Engine and Model Definition for FER Video Emotion Recognition.

This module implements:
1. `SpatialTemporalFERModel`: Exact PyTorch model combining a 2D CNN backbone (ResNet-18/MobileNetV2)
   with a Bidirectional Recurrent Neural Network (LSTM/GRU) and Temporal Global Average Pooling.
2. `EmotionPredictor`: Production wrapper for loading checkpoints, handling inference mode,
   device placement, and formatting probability distributions.
"""

import os
import json
import torch
import torch.nn as nn
from torchvision import models

from .video_preprocessing import process_video, transform_frames_to_tensor


EMOTIONS = [
    'neutral',
    'calm',
    'happy',
    'sad',
    'angry',
    'fearful',
    'disgust',
    'surprised'
]

EMOTION_MAP = {
    1: 'neutral',
    2: 'calm',
    3: 'happy',
    4: 'sad',
    5: 'angry',
    6: 'fearful',
    7: 'disgust',
    8: 'surprised'
}


class SpatialTemporalFERModel(nn.Module):
    """
    Spatial-Temporal Facial Emotion Recognition Model.
    Matches the architecture defined in model_training.ipynb and model_evaluation.ipynb.
    """
    def __init__(
        self,
        backbone_name='resnet18',
        rnn_type='LSTM',
        hidden_dim=256,
        num_rnn_layers=2,
        dropout=0.5,
        num_classes=8
    ):
        super(SpatialTemporalFERModel, self).__init__()

        self.backbone_name = backbone_name
        self.rnn_type = rnn_type

        # 1. 2D CNN Feature Extractor Backbone
        if backbone_name == 'resnet18':
            resnet = models.resnet18(weights=None)
            feature_dim = resnet.fc.in_features
            resnet.fc = nn.Identity()
            self.backbone = resnet
        elif backbone_name == 'mobilenet_v2':
            mobilenet = models.mobilenet_v2(weights=None)
            feature_dim = mobilenet.classifier[1].in_features
            mobilenet.classifier = nn.Identity()
            self.backbone = mobilenet
        else:
            raise ValueError(
                f"Unsupported backbone: '{backbone_name}'. "
                "Must be 'resnet18' or 'mobilenet_v2'."
            )

        # 2. Recurrent Network (LSTM or GRU)
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=feature_dim,
                hidden_size=hidden_dim,
                num_layers=num_rnn_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_rnn_layers > 1 else 0
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                input_size=feature_dim,
                hidden_size=hidden_dim,
                num_layers=num_rnn_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_rnn_layers > 1 else 0
            )
        else:
            raise ValueError(
                f"Unsupported rnn_type: '{rnn_type}'. "
                "Must be 'LSTM' or 'GRU'."
            )

        # 3. Fully Connected Classifier Head
        rnn_out_dim = hidden_dim * 2  # Bidirectional doubles output feature size
        self.classifier = nn.Sequential(
            nn.Linear(rnn_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Forward pass.
        Accepts tensor layout: (Batch, T, H, W, C) uint8/float OR (Batch, T, C, H, W) float.
        """
        if x.dim() == 5 and x.shape[-1] == 3:
            # (Batch, T, H, W, C) -> (Batch, T, C, H, W)
            x = x.permute(0, 1, 4, 2, 3).float() / 255.0

        batch_size, seq_len, C, H, W = x.shape

        # Merge Batch and Sequence dimensions for 2D CNN feature extraction
        c_in = x.reshape(batch_size * seq_len, C, H, W)
        cnn_features = self.backbone(c_in)  # (Batch * T, feature_dim)

        # Reshape back to sequence representation
        r_in = cnn_features.view(batch_size, seq_len, -1)  # (Batch, T, feature_dim)

        # Process through Recurrent Network
        rnn_out, _ = self.rnn(r_in)  # (Batch, T, hidden_dim * 2)

        # Temporal Global Average Pooling
        out_feature = torch.mean(rnn_out, dim=1)  # (Batch, hidden_dim * 2)

        # Classification logits
        logits = self.classifier(out_feature)  # (Batch, num_classes)
        return logits


class EmotionPredictor:
    """
    High-level inference predictor for FER video evaluation and Gradio demonstration.
    """
    def __init__(self, checkpoint_path=None, config_path=None, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.model = None
        self.config = None
        self.is_loaded = False
        self.load_error = None

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path, config_path)

    def load_checkpoint(self, checkpoint_path, config_path=None):
        """
        Loads trained weights and configuration from a model checkpoint file.
        Strictly validates checkpoint structure and model weights.
        """
        if not os.path.exists(checkpoint_path):
            self.is_loaded = False
            self.load_error = f"Checkpoint file not found at: '{checkpoint_path}'"
            print(f"[Warning] {self.load_error}")
            return False

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint:
                self.is_loaded = False
                self.load_error = f"Invalid checkpoint dictionary format in '{checkpoint_path}'. Missing 'model_state_dict'."
                print(f"[Error] {self.load_error}")
                return False

            # Retrieve config from checkpoint or external config_path JSON
            if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
                self.config = checkpoint['config']
            elif config_path and os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            else:
                # Default fallback config matching project standards
                self.config = {
                    "backbone": "resnet18",
                    "rnn_type": "LSTM",
                    "hidden_dim": 256,
                    "num_rnn_layers": 2,
                    "dropout": 0.5,
                    "num_classes": 8
                }

            # Build model instance with exact hyperparameters
            self.model = SpatialTemporalFERModel(
                backbone_name=self.config.get('backbone', 'resnet18'),
                rnn_type=self.config.get('rnn_type', 'LSTM'),
                hidden_dim=self.config.get('hidden_dim', 256),
                num_rnn_layers=self.config.get('num_rnn_layers', 2),
                dropout=self.config.get('dropout', 0.5),
                num_classes=self.config.get('num_classes', 8)
            )

            # Load model state dict strictly
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            self.model.to(self.device)
            self.model.eval()

            self.is_loaded = True
            self.load_error = None
            print(f"[Success] Loaded model checkpoint from '{checkpoint_path}' onto device '{self.device}'.")
            return True

        except Exception as e:
            self.is_loaded = False
            self.load_error = f"Failed to load checkpoint: {str(e)}"
            print(f"[Error] {self.load_error}")
            return False

    def predict_video(self, video_path):
        """
        Runs complete emotion prediction pipeline on a video file.

        Args:
            video_path (str): Path to video file.

        Returns:
            dict: Results containing top emotion, confidence score, and full class probabilities.
        """
        if not self.is_loaded or self.model is None:
            return {
                "success": False,
                "error": self.load_error or "Model checkpoint has not been loaded. Please provide a valid 'best_model.pth'."
            }

        if not os.path.exists(video_path):
            return {
                "success": False,
                "error": f"Uploaded video file not found at: '{video_path}'"
            }

        # Step 1: Preprocess video frames
        frames = process_video(
            video_path,
            sequence_length=self.config.get('sequence_length', 16),
            image_size=self.config.get('image_size', (224, 224))
        )

        if frames is None or len(frames) == 0:
            return {
                "success": False,
                "error": f"Failed to decode or sample frames from video: '{video_path}'"
            }

        # Step 2: Transform to model input tensor
        input_tensor = transform_frames_to_tensor(frames).to(self.device)

        # Step 3: Run model inference
        with torch.inference_mode():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        top_class_idx = int(np.argmax(probs))
        top_emotion = EMOTIONS[top_class_idx]
        confidence = float(probs[top_class_idx] * 100.0)

        probabilities_dict = {
            emotion: float(probs[i] * 100.0)
            for i, emotion in enumerate(EMOTIONS)
        }

        return {
            "success": True,
            "top_emotion": top_emotion,
            "confidence": confidence,
            "probabilities": probabilities_dict,
            "raw_probabilities": [float(p) for p in probs],
            "frames_count": len(frames)
        }
