"""Live camera appliance detection using the trained MobileNetV3-Small backbone.

Improvements for robust real-world detection:
  1. Multi-crop inference — center + 4 corners → averaged predictions
  2. Temporal smoothing — exponential moving average over frames
  3. Temperature scaling — sharper softmax for more decisive predictions

Opens the webcam and classifies each frame in real-time, displaying:
  - Predicted class label
  - Confidence score
  - FPS counter

Controls:
    q / ESC  — quit
    s        — save a screenshot

Usage:
    python src/live_detect.py
"""

import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
from collections import deque

# ── Config ───────────────────────────────────────────────────────
MODEL_PATH = "models/backbone.pt"
CLASSES = ["air_conditioner", "background", "refrigerator", "tv"]
CONFIDENCE_THRESHOLD = 0.4   # lowered — smoothing raises effective confidence
TEMPERATURE = 0.5            # <1 = sharper predictions (more confident)
SMOOTHING_ALPHA = 0.6        # EMA weight: higher = more weight on current frame
HISTORY_SIZE = 8             # frames for majority-vote stabilization

# Colors for each class (BGR)
CLASS_COLORS = {
    "air_conditioner": (255, 140, 50),   # blue-ish
    "background":      (100, 100, 100),  # dark gray
    "refrigerator":    (50, 220, 100),    # green
    "tv":              (80, 100, 255),    # red-ish
    "unknown":         (128, 128, 128),   # gray
}

# Pretty display names
CLASS_DISPLAY = {
    "air_conditioner": "Air Conditioner",
    "background":      "Background (None)",
    "refrigerator":    "Refrigerator",
    "tv":              "TV / Monitor",
}

# ── Preprocessing ────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(MEAN, STD)


def load_model():
    """Load the trained MobileNetV3-Small backbone."""
    model = torchvision.models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(1024, len(CLASSES))
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()
    return model


def get_crops(frame, crop_size=320):
    """Extract center crop + 4 corner crops from a frame for multi-crop inference.

    This helps because the object could be anywhere in the webcam view.
    Returns a batch tensor [5, 3, crop_size, crop_size].
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # Resize so shortest side = crop_size * 1.2 (gives room for crops)
    scale = max(crop_size * 1.2 / h, crop_size * 1.2 / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(rgb, (new_w, new_h))
    rh, rw = resized.shape[:2]

    crops = []
    positions = [
        # center
        ((rh - crop_size) // 2, (rw - crop_size) // 2),
        # top-left
        (0, 0),
        # top-right
        (0, rw - crop_size),
        # bottom-left
        (rh - crop_size, 0),
        # bottom-right
        (rh - crop_size, rw - crop_size),
    ]

    for y, x in positions:
        y = max(0, min(y, rh - crop_size))
        x = max(0, min(x, rw - crop_size))
        crop = resized[y:y+crop_size, x:x+crop_size]
        tensor = torch.from_numpy(crop).permute(2, 0, 1).float() / 255.0
        tensor = normalize(tensor)
        crops.append(tensor)

    return torch.stack(crops)  # [5, 3, 320, 320]


def predict_frame(model, frame, smoothed_probs=None):
    """Run multi-crop inference with temperature scaling and temporal smoothing.

    Returns:
        class_name (str): predicted class or 'unknown'
        confidence (float): smoothed confidence [0, 1]
        all_probs (dict): class → probability mapping
        smoothed_probs (np.array): updated EMA probability vector
    """
    # Multi-crop inference
    batch = get_crops(frame)  # [5, 3, 320, 320]

    with torch.no_grad():
        logits = model(batch)  # [5, num_classes]
        # Temperature scaling — divide logits by T before softmax
        scaled_logits = logits / TEMPERATURE
        probs = F.softmax(scaled_logits, dim=1)  # [5, num_classes]
        # Average across crops
        avg_probs = probs.mean(dim=0).numpy()  # [num_classes]

    # Temporal smoothing (exponential moving average)
    if smoothed_probs is None:
        smoothed_probs = avg_probs
    else:
        smoothed_probs = SMOOTHING_ALPHA * avg_probs + (1 - SMOOTHING_ALPHA) * smoothed_probs

    pred_idx = smoothed_probs.argmax()
    confidence = smoothed_probs[pred_idx]

    all_probs = {CLASSES[i]: float(smoothed_probs[i]) for i in range(len(CLASSES))}

    if confidence < CONFIDENCE_THRESHOLD:
        return "unknown", confidence, all_probs, smoothed_probs

    return CLASSES[pred_idx], confidence, all_probs, smoothed_probs


def draw_overlay(frame, class_name, confidence, all_probs, fps, stable_label):
    """Draw a sleek HUD overlay on the frame."""
    h, w = frame.shape[:2]

    # ── Top bar with prediction ──────────────────────────────────
    display_name = CLASS_DISPLAY.get(stable_label, "Unknown Object")
    color = CLASS_COLORS.get(stable_label, CLASS_COLORS["unknown"])

    # Semi-transparent top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Class label
    cv2.putText(frame, display_name, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

    # Confidence
    conf_text = f"Confidence: {confidence:.1%}"
    cv2.putText(frame, conf_text, (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

    # Confidence bar
    bar_x, bar_y, bar_w, bar_h = w - 220, 20, 200, 25
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (80, 80, 80), -1)
    fill_w = int(bar_w * confidence)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                  color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (200, 200, 200), 1)

    # FPS counter
    fps_text = f"FPS: {fps:.0f}"
    cv2.putText(frame, fps_text, (w - 130, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2, cv2.LINE_AA)

    # ── Bottom bar with all class probabilities ──────────────────
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 100), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)

    bar_total_w = w - 40
    x_offset = 20
    y_base = h - 70

    for i, cls in enumerate(CLASSES):
        prob = all_probs.get(cls, 0.0)
        cls_color = CLASS_COLORS.get(cls, (128, 128, 128))
        label = CLASS_DISPLAY.get(cls, cls)

        # Label
        cv2.putText(frame, f"{label}: {prob:.1%}", (x_offset, y_base - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # Probability bar
        seg_w = bar_total_w // len(CLASSES) - 10
        cv2.rectangle(frame, (x_offset, y_base + 5),
                      (x_offset + seg_w, y_base + 20), (60, 60, 60), -1)
        fill = int(seg_w * prob)
        cv2.rectangle(frame, (x_offset, y_base + 5),
                      (x_offset + fill, y_base + 20), cls_color, -1)
        cv2.rectangle(frame, (x_offset, y_base + 5),
                      (x_offset + seg_w, y_base + 20), (150, 150, 150), 1)

        x_offset += seg_w + 10

    # Controls hint
    cv2.putText(frame, "Q/ESC: Quit | S: Screenshot", (20, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)

    return frame


def main():
    print("Loading model...")
    model = load_model()
    print(f"Model loaded. Classes: {CLASSES}")
    print(f"Multi-crop: 5 crops | Temperature: {TEMPERATURE} | Smoothing: {SMOOTHING_ALPHA}")

    print("Opening camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open camera. Check permissions:")
        print("  macOS: System Preferences → Privacy & Security → Camera")
        return

    # Try to set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_w}x{actual_h}")
    print("Press 'q' or ESC to quit, 's' to save screenshot")
    print()

    fps = 0.0
    frame_count = 0
    fps_start = time.time()
    screenshot_count = 0
    smoothed_probs = None
    label_history = deque(maxlen=HISTORY_SIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        # Run inference with smoothing
        class_name, confidence, all_probs, smoothed_probs = predict_frame(
            model, frame, smoothed_probs
        )

        # Majority vote stabilization — prevents flickering
        label_history.append(class_name)
        if len(label_history) >= 3:
            # Use the most common label in recent history
            from collections import Counter
            vote_counts = Counter(label_history)
            stable_label = vote_counts.most_common(1)[0][0]
        else:
            stable_label = class_name

        # Use the smoothed confidence for the stable label
        stable_confidence = all_probs.get(stable_label, confidence)

        # Calculate FPS
        frame_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()

        # Draw overlay
        display = draw_overlay(frame, class_name, stable_confidence,
                               all_probs, fps, stable_label)

        # Show
        cv2.imshow("MoE Edge Detector - Live", display)

        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # q or ESC
            break
        elif key == ord('s'):
            screenshot_count += 1
            path = f"screenshot_{screenshot_count}.jpg"
            cv2.imwrite(path, frame)
            print(f"Screenshot saved: {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("Camera released. Bye!")


if __name__ == "__main__":
    main()
