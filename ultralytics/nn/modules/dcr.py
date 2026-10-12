"""Directional clearance routing: a candidate whose geometry supervision is training-only."""
import torch
from torch import nn
from torch.nn import functional as F
from .conv import Conv

DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))  # right, left, down, up


def spatial_sample(x, dx, dy):
    """At (x,y), read (x+dx,y+dy); zero padding prevents circular wraparound."""
    h, w = x.shape[-2:]
    pad = max(abs(dx), abs(dy))
    return F.pad(x, (pad, pad, pad, pad))[..., pad + dy:pad + dy + h, pad + dx:pad + dx + w]


class DCRFusion(nn.Module):
    """P2/P3 fusion followed by shared local features and five-way spatial routing.

    Directional features average samples at 2 and 4 P2 cells. Out-of-image
    directions fall back to the local feature. No boxes are accepted by forward.
    """
    def __init__(self, channels, c2, gamma=0.1):
        super().__init__()
        self.fusion = Conv(sum(channels), c2, 1)
        self.local = Conv(c2, c2, 3, g=c2)
        self.projection = Conv(c2, c2, 1)
        self.gate = nn.Sequential(Conv(c2, max(c2 // 4, 8), 1), nn.Conv2d(max(c2 // 4, 8), 5, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        self.gate_logits = None

    def forward(self, features):
        p2, p3 = features
        fused = self.fusion(torch.cat((p2, F.interpolate(p3, p2.shape[-2:], mode="nearest")), 1))
        local = self.local(fused)
        logits = self.gate(fused)
        self.gate_logits = logits if self.training else None
        weights = logits.softmax(1)
        mixed = weights[:, :1] * local
        valid = local.new_ones((1, 1, *local.shape[-2:]))
        for k, (dx, dy) in enumerate(DIRECTIONS, 1):
            v2, v4 = spatial_sample(valid, 2 * dx, 2 * dy), spatial_sample(valid, 4 * dx, 4 * dy)
            count = v2 + v4
            directional = (spatial_sample(local, 2 * dx, 2 * dy) + spatial_sample(local, 4 * dx, 4 * dy)) / count.clamp_min(1)
            directional = torch.where(count > 0, directional, local)
            mixed = mixed + weights[:, k:k + 1] * directional
        return fused + self.gamma * self.projection(mixed)
