"""Small, conventional fusion controls. These are diagnostic experiments, not novelty claims."""
import torch
from torch import nn
from torch.nn import functional as F
from .nscr import NSCRFusion


class P2Fusion(NSCRFusion):
    """Change only layer 23, retaining the exact baseline Concat/1x1 projection.

    Extra parameters are constructed inside fork_rng: shared layers and Detect
    start with the same random tensors as M0 for a given seed. All variants
    also start with the same fusion function. No GT teacher or auxiliary loss.
    """
    def __init__(self, channels, c2, mode="semantic", strength=.1):
        super().__init__(channels, c2, mode="none")
        if mode not in {"semantic", "global_weight", "spatial", "detail_plain", "detail_gated"}:
            raise ValueError(mode)
        self.variant = mode
        self.strength = float(strength)
        self.last_weights = None
        with torch.random.fork_rng(devices=[]):
            if mode == "global_weight":
                self.scale_logits = nn.Parameter(torch.zeros(1, 2, 1, 1))
            if mode in {"semantic", "spatial", "detail_gated"}:
                self.semantic_gate = nn.Conv2d(channels[1], 2 if mode == "spatial" else 1, 1)
                nn.init.zeros_(self.semantic_gate.weight)
                nn.init.zeros_(self.semantic_gate.bias)
            if mode in {"detail_plain", "detail_gated"}:
                # A zero-initialized linear DW filter avoids adding another BN/SiLU variable.
                self.detail = nn.Conv2d(channels[0], channels[0], 3, padding=1,
                                        groups=channels[0], bias=False)
                nn.init.zeros_(self.detail.weight)

    def forward(self, features):
        p2, p3 = features
        if self.variant == "semantic":
            gate = self.semantic_gate(p3).sigmoid()
            self.last_weights = gate.detach() if self.training else None
            p2 = p2 * (1 + self.strength * (2 * F.interpolate(gate, p2.shape[-2:], mode="nearest") - 1))
        elif self.variant in {"global_weight", "spatial"}:
            logits = self.scale_logits if self.variant == "global_weight" else self.semantic_gate(p3)
            weights = logits.softmax(1)
            self.last_weights = weights.detach() if self.training else None
            # Scale at the input to the existing projection. Its two weight
            # blocks already align both branches to the same output channels.
            p2 = p2 * (2 * F.interpolate(weights[:, :1], p2.shape[-2:], mode="nearest"))
            p3 = p3 * (2 * weights[:, 1:])
        else:
            detail = self.detail(p2 - F.avg_pool2d(p2, 3, 1, 1, count_include_pad=False))
            if self.variant == "detail_gated":
                gate = self.semantic_gate(p3).sigmoid()
                self.last_weights = gate.detach() if self.training else None
                detail = detail * (2 * F.interpolate(gate, p2.shape[-2:], mode="nearest"))
            p2 = p2 + self.strength * detail
        return self.fusion(torch.cat((p2, F.interpolate(p3, p2.shape[-2:], mode="nearest")), 1))
