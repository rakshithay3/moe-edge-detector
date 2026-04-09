from PIL import Image
import torch
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_frame(path):
    """Load and preprocess an image for the backbone.

    Args:
        path: Path to the image file

    Returns:
        Preprocessed tensor [1, 3, 320, 320]
    """
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)  # [1,3,320,320]
