import torch
import torchvision


def load_backbone(weights_path=None):
    """Load MobileNetV3-Small backbone, optionally with custom weights."""
    model = torchvision.models.mobilenet_v3_small(pretrained=True)

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
        GAP vector [B, 960]
    """
    with torch.no_grad():
        features = model.features(x)                    # [B, 960, H, W]
        gap = torch.nn.functional.adaptive_avg_pool2d(features, 1)
        return gap.view(x.size(0), -1)                  # [B, 960]
