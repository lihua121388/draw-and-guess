"""Auxiliary routing supervision; the original YOLO criterion and assignment remain unchanged."""

import math

import torch
import torch.nn.functional as F

from ultralytics.nn.modules.nscr import NSCRFusion
from ultralytics.utils.loss import v8DetectionLoss


@torch.no_grad()
def normalized_spacing(boxes, chunk_size=256):
    """Return min_j distance(i,j) / mean(scale_i,scale_j) for pixel xywh boxes.

    Minimize the normalized ratio, as specified in the research equation, rather than
    selecting an Euclidean nearest neighbor first. Chunking limits intermediate memory.
    """
    n = len(boxes)
    if n < 2:
        return boxes.new_full((n,), float("inf"))
    centers = boxes[:, :2].float()
    scale = boxes[:, 2:].float().prod(1).clamp_min(0).sqrt()
    result = []
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        distance = torch.cdist(centers[start:stop], centers)
        ratio = distance / ((scale[start:stop, None] + scale[None]) * 0.5 + 1e-6)
        ratio[torch.arange(stop - start, device=boxes.device), torch.arange(start, stop, device=boxes.device)] = float("inf")
        result.append(ratio.min(1).values)
    return torch.cat(result)


def sample_gate(logits, centers):
    """Bilinearly sample a one-channel gate at normalized box centers, align_corners=False."""
    grid = (centers * 2 - 1).view(1, -1, 1, 2)
    return F.grid_sample(logits.float(), grid.float(), mode="bilinear", padding_mode="border", align_corners=False).flatten()


@torch.no_grad()
def gaussian_density_targets(centers, selected, shape, sigma=2.0, threshold=2.0, temperature=0.5):
    """Conventional fixed Gaussian center density, evaluated at supervised centers.

    The fixed bandwidth is in P2 cells. A decreasing sigmoid maps high density to a local
    branch preference; no object-size normalization or nearest-neighbor ratio is used.
    """
    scale = centers.new_tensor([shape[1], shape[0]])
    if sigma <= 0 or temperature <= 0 or not all(math.isfinite(x) for x in (sigma, threshold, temperature)):
        raise ValueError("Density sigma/temperature must be positive and density parameters finite")
    outputs = []
    for points in selected.split(256):
        d2 = torch.cdist(points * scale, centers * scale).square()
        density = torch.exp(-d2 / (2 * sigma ** 2)).sum(1)
        outputs.append(torch.sigmoid((threshold - density) / temperature))
    return torch.cat(outputs)


def spacing_auxiliary(logits, batch, config):
    """Construct targets exclusively from the current augmented batch, averaging per image."""
    if logits is None:
        raise RuntimeError("NSCR training forward did not retain gate logits")
    threshold = float(config.get("small_size", 32))
    tau, temperature = float(config.get("tau", 2)), float(config.get("temperature", 0.5))
    if threshold <= 0 or temperature <= 0 or not all(math.isfinite(x) for x in (threshold, tau, temperature)):
        raise ValueError("NSCR small_size and temperature must be positive and all thresholds finite")
    supervision = config.get("supervision", "spacing")
    if supervision not in {"spacing", "shuffle", "density", "none"}:
        raise ValueError(f"Unknown NSCR supervision: {supervision}")
    image_h, image_w = batch["img"].shape[-2:]
    boxes = batch["bboxes"].detach().to(logits.device).float()
    classes = batch["cls"].detach().to(logits.device).flatten()
    indices = batch["batch_idx"].to(logits.device).flatten()
    valid = torch.isfinite(boxes).all(1) & torch.isfinite(classes)
    valid &= (boxes[:, 2:] > 0).all(1) & (boxes >= 0).all(1) & (boxes <= 1).all(1)
    valid &= (classes >= 0) & (classes < config["nc"]) & (classes == classes.round())
    # Optional mask for datasets that propagate ignore eligibility through augmentation.
    if "spacing_valid" in batch:
        valid &= batch["spacing_valid"].to(logits.device).flatten().bool()
    sampled, targets, groups = [], [], []
    for image_index in range(len(logits)):
        b = boxes[valid & (indices == image_index)]
        if len(b) < 2:
            continue
        pixel_boxes = b * b.new_tensor([image_w, image_h, image_w, image_h])
        q = normalized_spacing(pixel_boxes)
        small = (pixel_boxes[:, 2:].prod(1).sqrt() <= threshold) & torch.isfinite(q)
        if not small.any():
            continue
        centers = b[small, :2]
        target = torch.sigmoid((q[small] - tau) / temperature)
        if supervision == "density":
            target = gaussian_density_targets(b[:, :2], centers, logits.shape[-2:],
                config.get("density_sigma", 2), config.get("density_threshold", 2), config.get("density_temperature", 0.5))
        sampled.append(sample_gate(logits[image_index:image_index + 1], centers))
        targets.append(target)
        groups.append(len(target))
    if not sampled or supervision == "none":
        return logits.float().sum() * 0
    predictions, targets = torch.cat(sampled), torch.cat(targets)
    if supervision == "shuffle" and len(targets) > 1:
        # Derangement across the batch preserves the label histogram and eligibility mask.
        order = torch.randperm(len(targets), device=targets.device)
        shuffled = targets.clone()
        shuffled[order] = targets[order.roll(1)]
        targets = shuffled
    elif supervision == "shuffle":
        return logits.float().sum() * 0  # no valid permutation exists for a single label
    terms = F.binary_cross_entropy_with_logits(predictions, targets, reduction="none")
    return torch.stack([term.mean() for term in terms.split(groups)]).mean()


class NSCRDetectionLoss(v8DetectionLoss):
    """Append a weighted spacing term to unchanged box/cls/DFL losses."""

    def __init__(self, model):
        super().__init__(model)
        self.config = dict(model.yaml["nscr"], nc=model.model[-1].nc)
        self.router = next(m for m in model.modules() if isinstance(m, NSCRFusion))
        weight = float(self.config.get("weight", 0.1))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("NSCR auxiliary weight must be finite and nonnegative")

    def __call__(self, preds, batch):
        loss, items = super().__call__(preds, batch)
        weight = float(self.config.get("weight", 0.1))
        aux = loss.new_zeros(())
        if self.router.training and self.router.mode == "gated":
            logits = self.router.gate_logits
            self.router.gate_logits = None  # release the graph before EMA/checkpoint deepcopy
            aux = spacing_auxiliary(logits, batch, self.config) * weight
        # Eval losses contain a zero auxiliary term. Evaluation never constructs GT routing targets.
        return torch.cat((loss, (aux * len(batch["img"])).reshape(1))), torch.cat((items, aux.detach().reshape(1)))
