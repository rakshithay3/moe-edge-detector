"""Router MLP — routes GAP vectors to the correct expert."""

import torch
import torch.nn as nn


class RouterMLP(nn.Module):
    """Lightweight MLP that maps 576-d GAP vectors to expert indices.

    Architecture:
        576 → 256 → ReLU → Dropout → 3 (expert logits)
    """

    def __init__(self, input_dim=576, hidden_dim=256, num_experts=4, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x: GAP vector [B, 576]

        Returns:
            Expert logits [B, num_experts]
        """
        return self.net(x)


def load_router(weights_path, input_dim=576, hidden_dim=256, num_experts=3):
    """Load a trained router from disk."""
    router = RouterMLP(input_dim, hidden_dim, num_experts)
    router.load_state_dict(torch.load(weights_path, map_location="cpu"))
    router.eval()
    return router


def predict_expert(router, gap_vector):
    """Predict which expert to route to.

    Args:
        router: Trained RouterMLP
        gap_vector: [1, 576] tensor

    Returns:
        expert_id (int), confidence (float)
    """
    with torch.no_grad():
        logits = router(gap_vector)                     # [1, num_experts]
        probs = torch.softmax(logits, dim=-1)
        confidence, expert_id = probs.max(dim=-1)
        return expert_id.item(), confidence.item()
