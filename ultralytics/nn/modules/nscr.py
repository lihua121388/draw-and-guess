"""P2/P3 fusion and spacing-conditioned receptive field routing for NSCR experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


class NSCRFusion(nn.Module):
    """Fuse P2/P3, optionally adding a single branch or a learned two-branch router.

    Gate logits are retained only during training and consumed by NSCRDetectionLoss.
    RGB-only inference does not compute or require any box-derived quantities.
    """

    def __init__(self, channels, c2, mode="gated", gamma=0.1):
        super().__init__()
        if mode not in {"none", "single", "gated"}:
            raise ValueError(f"Unknown NSCR mode: {mode}")
        self.mode = mode
        self.fusion = Conv(sum(channels), c2, 1)
        self.gate_logits = None
        if mode == "single":
            # One 3x3 spatial path; select bottleneck width to approximate gated-module parameters.
            gate_hidden = max(c2 // 4, 8)
            gated_budget = 2 * (9 * c2 + 2 * c2) + c2 * gate_hidden + 3 * gate_hidden + 1
            gated_budget += c2 * c2 + 2 * c2
            hidden = max(1, round((gated_budget - 2 * c2) / (2 * c2 + 13)))
            self.local = nn.Sequential(Conv(c2, hidden, 1), Conv(hidden, hidden, 3, g=hidden))
            self.projection = Conv(hidden, c2, 1)
            self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        elif mode == "gated":
            self.local = Conv(c2, c2, 3, g=c2)
            self.projection = Conv(c2, c2, 1)
            self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        if mode == "gated":
            self.wide = Conv(c2, c2, 3, g=c2, d=3)
            self.gate = nn.Sequential(Conv(c2, max(c2 // 4, 8), 1), nn.Conv2d(max(c2 // 4, 8), 1, 1))
            nn.init.zeros_(self.gate[-1].weight)
            nn.init.zeros_(self.gate[-1].bias)

    def forward(self, features):
        """Return a P2-resolution feature tensor from [P2, P3]."""
        p2, p3 = features
        fused = self.fusion(torch.cat((p2, F.interpolate(p3, size=p2.shape[-2:], mode="nearest")), 1))
        if self.mode == "none":
            return fused
        local = self.local(fused)
        if self.mode == "single":
            return fused + self.gamma * self.projection(local)
        logits = self.gate(fused)
        self.gate_logits = logits if self.training else None
        gate = logits.sigmoid()
        return fused + self.gamma * self.projection((1 - gate) * local + gate * self.wide(fused))
