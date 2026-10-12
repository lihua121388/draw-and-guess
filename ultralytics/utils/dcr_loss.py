"""Box-boundary directional clearance targets, recalculated after augmentation."""
import math
import torch
from torch.nn import functional as F
from ultralytics.nn.modules.dcr import DCRFusion
from ultralytics.utils.loss import v8DetectionLoss


@torch.no_grad()
def directional_clearance(boxes, image_size, cap=6.0, chunk_size=128):
    """Pixel xywh -> [N,4] normalized boundary gaps in R/L/D/U corridors.

    Neighbors must overlap the perpendicular box interval and lie in that
    center half-plane. Overlap gives zero gap. No neighbor gives capped
    clearance, limited by the image boundary. Working storage is O(chunk*N).
    """
    if not math.isfinite(cap) or cap <= 0 or chunk_size < 1:
        raise ValueError("cap and chunk_size must be positive")
    if not len(boxes):
        return boxes.new_empty((0, 4))
    b = boxes.float()
    center, size = b[:, :2], b[:, 2:]
    lo, hi = center - size / 2, center + size / 2
    scale = size.prod(1).clamp_min(1e-12).sqrt()
    height, width = image_size
    boundary = torch.stack((width - hi[:, 0], lo[:, 0], height - hi[:, 1], lo[:, 1]), 1).clamp_min(0)
    boundary = (boundary / scale[:, None]).clamp_max(cap)
    result = []
    for start in range(0, len(b), chunk_size):
        stop = min(start + chunk_size, len(b))
        overlaps_y = (lo[start:stop, None, 1] < hi[None, :, 1]) & (hi[start:stop, None, 1] > lo[None, :, 1])
        overlaps_x = (lo[start:stop, None, 0] < hi[None, :, 0]) & (hi[start:stop, None, 0] > lo[None, :, 0])
        dx = center[None, :, 0] - center[start:stop, None, 0]
        dy = center[None, :, 1] - center[start:stop, None, 1]
        # Identical centers are genuine blockers in all directions (except self).
        masks = (overlaps_y & (dx >= 0), overlaps_y & (dx <= 0), overlaps_x & (dy >= 0), overlaps_x & (dy <= 0))
        gaps = (lo[None, :, 0] - hi[start:stop, None, 0], lo[start:stop, None, 0] - hi[None, :, 0],
                lo[None, :, 1] - hi[start:stop, None, 1], lo[start:stop, None, 1] - hi[None, :, 1])
        denominator = .5 * (scale[start:stop, None] + scale[None]) + 1e-6
        directional = []
        for mask, gap in zip(masks, gaps):
            ratio = (gap.clamp_min(0) / denominator).masked_fill(~mask, float("inf"))
            ratio[torch.arange(stop-start, device=b.device), torch.arange(start, stop, device=b.device)] = float("inf")
            directional.append(ratio.min(1).values)
        result.append(torch.minimum(torch.stack(directional, 1), boundary[start:stop]))
    return torch.cat(result)


def clearance_distribution(q, tau=2.0, temperature=.5, supervision="directional"):
    """Local weight=1 and directional sigmoid weights, normalized to a distribution."""
    if temperature <= 0 or not all(math.isfinite(v) for v in (tau, temperature)):
        raise ValueError("Finite tau and positive temperature are required")
    wide = torch.sigmoid((q - tau) / temperature)
    if supervision == "isotropic":
        wide = wide.mean(1, keepdim=True).expand_as(wide)
    elif supervision == "shuffle":
        # Per-instance nonidentity direction permutation; preserve local/context
        # mass and each instance's directional label histogram exactly.
        offsets = torch.randint(1, 4, (len(wide), 1), device=wide.device)
        wide = wide.gather(1, (torch.arange(4, device=wide.device)[None] + offsets) % 4)
    elif supervision != "directional":
        raise ValueError(f"Unknown supervision: {supervision}")
    target = torch.cat((torch.ones_like(wide[:, :1]), wide), 1)
    return target / target.sum(1, keepdim=True)


def dcr_auxiliary(logits, batch, config):
    if logits is None:
        raise RuntimeError("DCR logits missing from training forward")
    if config.get("supervision") == "none":
        return logits.float().sum() * 0
    h, w = batch["img"].shape[-2:]
    boxes = batch["bboxes"].detach().to(logits.device).float()
    classes = batch["cls"].detach().to(logits.device).flatten()
    indices = batch["batch_idx"].to(logits.device).flatten()
    valid = torch.isfinite(boxes).all(1) & torch.isfinite(classes)
    valid &= (boxes >= 0).all(1) & (boxes <= 1).all(1) & (boxes[:, 2:] > 0).all(1)
    valid &= (classes >= 0) & (classes < config["nc"]) & (classes == classes.round())
    if "spacing_valid" in batch:
        valid &= batch["spacing_valid"].to(logits.device).flatten().bool()
    losses = []
    for image in range(len(logits)):
        b = boxes[valid & (indices == image)]
        if len(b) < 2:
            continue
        pixels = b * b.new_tensor([w, h, w, h])
        small = pixels[:, 2:].prod(1).sqrt() <= config.get("small_size", 32)
        if not small.any():
            continue
        q = directional_clearance(pixels, (h, w), config.get("cap", 6))[small]
        targets = clearance_distribution(q, config.get("tau", 2), config.get("temperature", .5), config["supervision"])
        grid = (b[small, :2] * 2 - 1).view(1, -1, 1, 2)
        sampled = F.grid_sample(logits[image:image+1].float(), grid, mode="bilinear", padding_mode="border", align_corners=False)[0, :, :, 0].T
        losses.append(-(targets * sampled.log_softmax(1)).sum(1).mean())
    return torch.stack(losses).mean() if losses else logits.float().sum() * 0


class DCRDetectionLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.config = dict(model.yaml["dcr"], nc=model.model[-1].nc)
        self.router = next(m for m in model.modules() if isinstance(m, DCRFusion))
        if not math.isfinite(self.config["weight"]) or self.config["weight"] < 0:
            raise ValueError("DCR weight must be finite and nonnegative")
        for key in ("cap", "temperature", "small_size"):
            if not math.isfinite(self.config[key]) or self.config[key] <= 0:
                raise ValueError(f"DCR {key} must be finite and positive")
        if not math.isfinite(self.config["tau"]):
            raise ValueError("DCR tau must be finite")
        if self.config["supervision"] not in {"none", "isotropic", "directional", "shuffle"}:
            raise ValueError("Unknown DCR supervision")

    def __call__(self, preds, batch):
        loss, items = super().__call__(preds, batch)
        aux = loss.new_zeros(())
        if self.router.training:
            logits, self.router.gate_logits = self.router.gate_logits, None
            aux = dcr_auxiliary(logits, batch, self.config) * self.config["weight"]
        return torch.cat((loss, (aux * len(batch["img"])).view(1))), torch.cat((items, aux.detach().view(1)))
