"""End-to-end MoE inference demo.

Usage:
    python src/inference_demo.py <image_path>
"""

import sys
import torch

from src.preprocess import load_frame
from src.backbone import load_backbone, extract_gap
from src.router import load_router, predict_expert

# Expert names for display
EXPERT_NAMES = {
    0: "Display (TV)",
    1: "Kitchen (Refrigerator / Microwave)",
    2: "Climate (Air Conditioner)",
}


def run_inference(image_path,
                  backbone_path="models/backbone.pt",
                  router_path="models/router.pt"):
    """Run the full MoE inference pipeline on a single image.

    Pipeline:
        Image → Preprocess (320×320) → MobileNetV3 backbone → GAP (960-d)
              → Router MLP → Expert selection → Output
    """
    # 1. Preprocess
    print(f"[1/4] Loading & preprocessing: {image_path}")
    x = load_frame(image_path)                          # [1, 3, 320, 320]

    # 2. Backbone → GAP
    print("[2/4] Extracting GAP features via MobileNetV3-Small...")
    backbone = load_backbone(backbone_path)
    gap = extract_gap(backbone, x)                      # [1, 960]
    print(f"       GAP vector shape: {gap.shape}")

    # 3. Router → expert selection
    print("[3/4] Routing through MLP...")
    router = load_router(router_path)
    expert_id, confidence = predict_expert(router, gap)

    expert_name = EXPERT_NAMES.get(expert_id, f"Unknown ({expert_id})")
    print(f"       Selected Expert: {expert_id} — {expert_name}")
    print(f"       Confidence:      {confidence:.4f}")

    # 4. Expert model (placeholder — load and run expert-specific model)
    print("[4/4] Running expert model...")
    expert_weights = {
        0: "models/expert_0_display.pt",
        1: "models/expert_1_kitchen.pt",
        2: "models/expert_2_climate.pt",
    }

    expert_path = expert_weights[expert_id]
    print(f"       Would load expert weights from: {expert_path}")
    # TODO: Load and run the expert-specific detection head here
    # expert = load_expert(expert_path)
    # detections = expert(gap)  # or expert(features)
    # boxes, scores, labels = filter_detections(...)

    print()
    print("=" * 50)
    print(f"  RESULT: {expert_name}")
    print(f"  Confidence: {confidence:.2%}")
    print("=" * 50)

    return expert_id, confidence


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/inference_demo.py <image_path>")
        sys.exit(1)

    run_inference(sys.argv[1])
