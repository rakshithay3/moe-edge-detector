"""Live MoE router detection from webcam frames.

Improvements for robust real-world detection:
  1. Multi-crop inference — center + 4 corners → averaged predictions
  2. Temporal smoothing — exponential moving average over frames
  3. Temperature scaling — sharper softmax for more decisive predictions

Opens the webcam and shows:
  - Selected MoE expert (Kitchen/Display/Climate/Utility)
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

# MoE components
from src.backbone import load_backbone, extract_gap
from src.router import load_router

# ── Config ───────────────────────────────────────────────────────
BACKBONE_PATH = "models/backbone.pt"
ROUTER_PATH = "models/router.pt"

EXPERT_IDS = [0, 1, 2, 3]
CONFIDENCE_THRESHOLD = 0.7   # below this, treat as "unknown"
# Match `src/router.predict_expert()` (no temperature scaling) for consistency
TEMPERATURE = 1.0
SMOOTHING_ALPHA = 0.3        # less stickiness helps when the scene changes
HISTORY_SIZE = 4            # fewer frames reduces lag/mis-sticking

# Colors for each expert (BGR)
EXPERT_COLORS = {
    0: (50, 220, 100),    # kitchen green
    1: (80, 100, 255),    # display red-ish
    2: (255, 140, 50),    # climate blue-ish
    3: (200, 200, 200),   # utility/background gray
}

# Pretty display names for experts
EXPERT_DISPLAY = {
    0: "Kitchen (Refrigerator / Microwave / Dishwasher)",
    1: "Display (TV)",
    2: "Climate (Air Conditioner / Air Purifier)",
    3: "Utility (Washer / Robot Vacuum) / Background",
}

# ── Preprocessing ────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(MEAN, STD)


def load_model():
    """Load backbone + router for live MoE expert routing."""
    backbone = load_backbone(BACKBONE_PATH, num_classes=8)
    router = load_router(ROUTER_PATH)
    return backbone, router


USE_MULTICROP = True  # webcam objects may be off-center; use robust pooling


def preprocess_frame(frame, size=320):
    """Train-aligned preprocessing: resize full frame to (320,320) once.

    This matches how GAP vectors were generated in `train/generate_gap.py`.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0  # [3,H,W]
    tensor = normalize(tensor).unsqueeze(0)  # [1,3,H,W]
    return tensor


def get_crops(frame, crop_size=320):
    """Extract center crop + 4 corner crops from a frame for multi-crop inference.

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


def predict_frame(backbone, router, frame, smoothed_probs=None):
    """Run multi-crop inference + MoE routing with temporal smoothing."""
    with torch.no_grad():
        if USE_MULTICROP:
            batch = get_crops(frame)  # [5, 3, 320, 320]
            gap = extract_gap(backbone, batch)   # [5, 576]
            logits = router(gap)                # [5, 4]
            # Aggregation: average logits across crops, then softmax once.
            # This keeps probabilities calibrated (unlike max-over-crops).
            scaled_logits = logits / TEMPERATURE  # [5, 4]
            mean_logits = scaled_logits.mean(dim=0)  # [4]
            avg_probs = F.softmax(mean_logits, dim=0).cpu().numpy()  # [4]
        else:
            x = preprocess_frame(frame, size=320)  # [1,3,320,320]
            gap = extract_gap(backbone, x)          # [1,576]
            logits = router(gap)                  # [1,4]
            scaled_logits = logits / TEMPERATURE
            probs = F.softmax(scaled_logits, dim=1)  # [1,4]
            avg_probs = probs[0].cpu().numpy()       # [4]

    # Temporal smoothing (exponential moving average)
    if smoothed_probs is None:
        smoothed_probs = avg_probs
    else:
        smoothed_probs = SMOOTHING_ALPHA * avg_probs + (1 - SMOOTHING_ALPHA) * smoothed_probs

    pred_expert_id = int(smoothed_probs.argmax())
    confidence = float(smoothed_probs[pred_expert_id])

    all_probs = {expert_id: float(smoothed_probs[expert_id]) for expert_id in EXPERT_IDS}

    if confidence < CONFIDENCE_THRESHOLD:
        return "unknown", None, confidence, all_probs, smoothed_probs

    return EXPERT_DISPLAY[pred_expert_id], pred_expert_id, confidence, all_probs, smoothed_probs


def draw_overlay(frame, expert_label, expert_id, confidence, all_probs, fps, stable_expert_id):
    """Draw a sleek HUD overlay on the frame."""
    h, w = frame.shape[:2]

    # ── Top bar with prediction ──────────────────────────────────
    if stable_expert_id is None:
        display_name = "Unknown Expert"
        color = (128, 128, 128)
    else:
        display_name = EXPERT_DISPLAY.get(stable_expert_id, "Unknown Expert")
        color = EXPERT_COLORS.get(stable_expert_id, (128, 128, 128))

    # Semi-transparent top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

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

    # ── Bottom bar with all expert probabilities ────────────────
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 100), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)

    bar_total_w = w - 40
    x_offset = 20
    y_base = h - 70

    for expert_id in EXPERT_IDS:
        prob = all_probs.get(expert_id, 0.0)
        cls_color = EXPERT_COLORS.get(expert_id, (128, 128, 128))
        label = EXPERT_DISPLAY.get(expert_id, str(expert_id))

        # Label
        short_label = label.split(" (")[0]
        cv2.putText(frame, f"{short_label}: {prob:.1%}", (x_offset, y_base - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # Probability bar
        seg_w = bar_total_w // len(EXPERT_IDS) - 10
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
    backbone, router = load_model()
    print("Models loaded. Showing MoE router experts.")
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
    expert_id_history = deque(maxlen=HISTORY_SIZE)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        # Run inference with smoothing
        expert_label, expert_id, confidence, all_probs, smoothed_probs = predict_frame(
            backbone, router, frame, smoothed_probs
        )

        # Majority vote stabilization — prevents flickering
        if expert_id is not None:
            expert_id_history.append(expert_id)
        if len(expert_id_history) >= 3:
            # Use the most common label in recent history
            from collections import Counter
            vote_counts = Counter(expert_id_history)
            stable_expert_id = vote_counts.most_common(1)[0][0]
        else:
            # If we don't have enough confident history, don't default to an expert.
            stable_expert_id = expert_id if expert_id is not None else None

        # Use the smoothed confidence for the stable label
        stable_confidence = confidence if stable_expert_id is None else all_probs.get(stable_expert_id, confidence)

        # Calculate FPS
        frame_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()

        # Draw overlay
        display = draw_overlay(frame, expert_label, expert_id, stable_confidence,
                               all_probs, fps, stable_expert_id)

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
