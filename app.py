"""
Gradio Web Application for RAVDESS Video Emotion Recognition.

Author: Cong Thanh (Project FER-Video-Emotion-Recognition)
Course: Công nghệ phần mềm nâng cao

Features:
- Video Upload and Playback Preview
- Face Detection & 16-Frame Uniform Sampling Pipeline
- Inference using SpatialTemporalFERModel (ResNet-18 + LSTM)
- Display of Top Emotion, Confidence Score %, and Full Probability Distribution
- Setup validation and graceful error handling for missing checkpoints/videos
"""

import os
import gradio as gr
from src.inference import EmotionPredictor, EMOTIONS

# Default file paths
DEFAULT_CHECKPOINT = os.path.join('checkpoints', 'best_model.pth')
DEFAULT_CONFIG = os.path.join('checkpoints', 'model_config.json')

# Global predictor instance
predictor = EmotionPredictor()


def handle_checkpoint_load(checkpoint_path, config_path):
    """
    Handles manual or automatic checkpoint reloading in Gradio UI.
    """
    if not checkpoint_path:
        checkpoint_path = DEFAULT_CHECKPOINT
    if not config_path:
        config_path = DEFAULT_CONFIG

    success = predictor.load_checkpoint(checkpoint_path, config_path)
    if success:
        return f"✓ Model loaded successfully from '{checkpoint_path}' (Device: {predictor.device})"
    else:
        return f"⚠️ Warning: {predictor.load_error}"


def predict_video_emotion(video_path, checkpoint_path, config_path):
    """
    Gradio event handler for video emotion recognition.
    """
    if video_path is None:
        return (
            "Please upload a valid video file.",
            "0.0%",
            {},
            "⚠️ No video file uploaded."
        )

    # Reload checkpoint if not already loaded or if path changed
    if not predictor.is_loaded or predictor.checkpoint_path != checkpoint_path:
        load_msg = handle_checkpoint_load(checkpoint_path, config_path)
        if not predictor.is_loaded:
            return (
                "Checkpoint Missing",
                "0.0%",
                {},
                f"❌ Model Checkpoint Error: {predictor.load_error}\n\n"
                f"Please place your trained 'best_model.pth' inside the 'checkpoints/' folder or specify its valid path."
            )

    result = predictor.predict_video(video_path)

    if not result.get("success", False):
        error_msg = result.get("error", "An unknown error occurred during inference.")
        return (
            "Inference Error",
            "0.0%",
            {},
            f"❌ Prediction Failed: {error_msg}"
        )

    top_emotion = result["top_emotion"].upper()
    confidence_str = f"{result['confidence']:.2f}%"
    probabilities = {emotion.capitalize(): score for emotion, score in result["probabilities"].items()}
    status_text = (
        f"✓ Successfully processed video across 16 sampled frames.\n"
        f"Detected Emotion: {top_emotion} ({confidence_str} confidence)."
    )

    return top_emotion, confidence_str, probabilities, status_text


def build_app():
    """
    Constructs the Gradio Blocks UI interface.
    """
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
    )

    with gr.Blocks(theme=theme, title="RAVDESS Video Emotion Recognition") as demo:
        gr.Markdown(
            """
            # 🎭 RAVDESS Video Emotion Recognition (FER) Demo
            **Môn học:** Công nghệ phần mềm nâng cao | **Thực hiện:** Phan Công Thành
            
            Demonstration of Spatial-Temporal Facial Emotion Recognition using a deep learning pipeline:
            Uniform 16-Frame Sampling → Haar Cascade Face Detection → ResNet-18 Feature Extractor → Bidirectional LSTM → 8-Class Emotion Classification.
            """
        )

        with gr.Accordion("📌 RAVDESS 8 Emotion Classes Overview", open=False):
            gr.Markdown(
                """
                The RAVDESS (Ryerson Audio-Visual Emotional Speech and Song) dataset defines 8 distinct emotional categories:
                1. **Neutral** (Bình thường / Trung tính)
                2. **Calm** (Bình tĩnh / Thư thái)
                3. **Happy** (Vui vẻ / Hạnh phúc)
                4. **Sad** (Buồn rầu / Thất vọng)
                5. **Angry** (Tức giận / Phẫn nộ)
                6. **Fearful** (E sợ / Lo âu)
                7. **Disgust** (Chán ghét / Gê tởm)
                8. **Surprised** (Bất ngờ / Ngạc nhiên)
                """
            )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Upload Video & Settings")
                video_input = gr.Video(
                    label="Upload Short Video (.mp4, .avi, .mov)",
                    sources=["upload"],
                    interactive=True
                )

                with gr.Accordion("⚙️ Model Checkpoint Settings", open=False):
                    ckpt_input = gr.Textbox(
                        label="Checkpoint Path (.pth)",
                        value=DEFAULT_CHECKPOINT,
                        placeholder="checkpoints/best_model.pth"
                    )
                    cfg_input = gr.Textbox(
                        label="Config JSON Path (.json)",
                        value=DEFAULT_CONFIG,
                        placeholder="checkpoints/model_config.json"
                    )
                    load_btn = gr.Button("Reload Checkpoint", variant="secondary", size="sm")

                submit_btn = gr.Button("🚀 Predict Emotion", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### 2. Emotion Recognition Results")
                status_output = gr.Textbox(
                    label="System Status & Log",
                    value=f"Initial Status: Checkpoint default path set to '{DEFAULT_CHECKPOINT}'",
                    interactive=False
                )

                with gr.Row():
                    top_emotion_output = gr.Textbox(
                        label="Predicted Emotion",
                        placeholder="Waiting for video...",
                        interactive=False
                    )
                    confidence_output = gr.Textbox(
                        label="Confidence",
                        placeholder="0.0%",
                        interactive=False
                    )

                probabilities_output = gr.Label(
                    label="Probability Distribution across 8 Emotion Classes",
                    num_top_classes=8
                )

        # Event Bindings
        load_btn.click(
            fn=handle_checkpoint_load,
            inputs=[ckpt_input, cfg_input],
            outputs=[status_output]
        )

        submit_btn.click(
            fn=predict_video_emotion,
            inputs=[video_input, ckpt_input, cfg_input],
            outputs=[top_emotion_output, confidence_output, probabilities_output, status_output]
        )

        gr.Markdown(
            """
            ---
            *Note: If no model checkpoint `best_model.pth` is present in `checkpoints/`, the interface will present a setup message explaining missing weights.*
            """
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(share=False)
