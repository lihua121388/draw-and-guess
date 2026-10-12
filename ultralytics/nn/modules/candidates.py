"""Five independent research candidates for P2/P3 fusion, all RGB-only at inference."""
import torch
from torch import nn
from torch.nn import functional as F
from .conv import Conv
from .dcr import DCRFusion, spatial_sample

KINDS = ("dcr", "csr", "brr", "gpr", "pcr")
PEER_OFFSETS = ((2, 0), (-2, 0), (0, 2), (0, -2), (2, 2), (-2, 2), (2, -2), (-2, -2))


class CandidateFusion(nn.Module):
    """One candidate at a time; retain auxiliary tensors only for training loss.

    dcr: directional clearance; csr: collision-safe rings;
    brr: boundary-residual recovery; gpr: grid-phase reliability;
    pcr: supervised peer-consensus rescue.
    """
    def __init__(self, channels, c2, kind, gamma=.1):
        super().__init__()
        if kind not in KINDS:
            raise ValueError(f"Unknown candidate kind: {kind}")
        self.kind, self.aux_state = kind, None
        if kind == "dcr":
            self.impl = DCRFusion(channels, c2, gamma)
            return
        self.fusion = Conv(sum(channels), c2, 1)
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        self.projection = Conv(c2, c2, 1)
        hidden = max(c2 // 4, 8)
        out = 4 if kind == "csr" else 2 if kind == "gpr" else 1
        self.head = nn.Sequential(Conv(c2, hidden, 1), nn.Conv2d(hidden, out, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        if kind == "gpr":
            self.detail = Conv(channels[0], c2, 1)
            self.semantic = Conv(channels[1], c2, 1)
        else:
            self.local = Conv(c2, c2, 3, g=c2)
        if kind == "pcr":
            self.embedding = nn.Conv2d(c2, 16, 1)
            self.affinity_temperature = .3

    def forward(self, features):
        if self.kind == "dcr":
            output = self.impl(features)
            self.aux_state = {"logits": self.impl.gate_logits} if self.training else None
            return output
        p2, p3 = features
        fused = self.fusion(torch.cat((p2, F.interpolate(p3, p2.shape[-2:], mode="nearest")), 1))
        logits = self.head(fused)
        state = {"logits": logits}
        if self.kind == "gpr":
            weights = logits.softmax(1)
            mixed = weights[:, :1] * self.detail(p2) + weights[:, 1:] * F.interpolate(
                self.semantic(p3), p2.shape[-2:], mode="nearest")
        elif self.kind == "brr":
            # Shared high-pass residual; learned contour trust is applied to detail.
            detail = fused - F.avg_pool2d(fused, 3, 1, 1, count_include_pad=False)
            mixed = self.local(fused) + logits.sigmoid() * detail
        elif self.kind == "csr":
            local = self.local(fused)
            weights = logits.softmax(1)
            mixed = weights[:, :1] * local
            valid = local.new_ones((1, 1, *local.shape[-2:]))
            for k, radius in enumerate((1, 3, 5), 1):
                outer, inner = 2 * radius + 1, 2 * radius - 1
                total = F.avg_pool2d(local, outer, 1, radius, divisor_override=1)
                total = total - F.avg_pool2d(local, inner, 1, radius - 1, divisor_override=1)
                count = F.avg_pool2d(valid, outer, 1, radius, divisor_override=1)
                count = count - F.avg_pool2d(valid, inner, 1, radius - 1, divisor_override=1)
                ring = torch.where(count > 0, total / count.clamp_min(1), local)
                mixed = mixed + weights[:, k:k+1] * ring
        else:
            local = self.local(fused)
            embedding = F.normalize(self.embedding(fused), dim=1, eps=1e-6)
            state["embedding"] = embedding
            trust = logits.float().sigmoid().to(local.dtype)
            candidates, scores = [local], [torch.zeros_like(trust)]
            valid = local.new_ones((1, 1, *local.shape[-2:]))
            for dx, dy in PEER_OFFSETS:
                peer = spatial_sample(embedding, dx, dy)
                cosine = (embedding * peer).sum(1, keepdim=True)
                peer_trust = spatial_sample(trust, dx, dy)
                score = cosine.float() / self.affinity_temperature + peer_trust.float().clamp_min(1e-6).log()
                score = score.masked_fill(spatial_sample(valid, dx, dy) == 0, -1e4)
                scores.append(score)
                candidates.append(spatial_sample(local, dx, dy))
            weights = torch.cat(scores, 1).softmax(1).to(local.dtype)
            mixed = sum(weights[:, k:k+1] * feature for k, feature in enumerate(candidates))
        self.aux_state = state if self.training else None
        return fused + self.gamma * self.projection(mixed)

    def consume_aux(self):
        state, self.aux_state = self.aux_state, None
        if self.kind == "dcr":
            self.impl.gate_logits = None
        return state
