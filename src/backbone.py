import torch
import torchvision


def load_backbone(weights_path=None, num_classes=8):
    """Load MobileNetV3-Small backbone, optionally with custom weights."""
    model = torchvision.models.mobilenet_v3_small(weights="IMAGENET1K_V1")
    # Match the classifier head used during training
    model.classifier[3] = torch.nn.Linear(1024, num_classes)

    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))

    model.eval()
    return model


def extract_gap(model, x):
    """Extract Global Average Pooled features from the backbone.

    Args:
        model: MobileNetV3-Small model
        x: Input tensor [B, 3, 320, 320]

    Returns:
        GAP vector [B, 576]
    """
    with torch.no_grad():
        features = model.features(x)                    # [B, 576, H, W]
        gap = torch.nn.functional.adaptive_avg_pool2d(features, 1)
        return gap.view(x.size(0), -1)                  # [B, 576]

