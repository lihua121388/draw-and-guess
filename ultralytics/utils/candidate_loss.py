"""Augmentation-consistent auxiliary teachers for five research candidates."""
import math
import torch
from torch.nn import functional as F
from ultralytics.nn.modules.candidates import CandidateFusion
from ultralytics.utils.dcr_loss import dcr_auxiliary
from ultralytics.utils.loss import v8DetectionLoss


def sample_features(features, centers):
    grid = (centers * 2 - 1).reshape(1, -1, 1, 2)
    return F.grid_sample(features.float(), grid, mode="bilinear", padding_mode="border",
                         align_corners=False)[0, :, :, 0].T


def pixel_edges(boxes):
    return boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2


@torch.no_grad()
def max_cell_overlap(lo, hi, stride):
    """Maximum overlap with any one stride-wide lattice cell, separately per axis."""
    first, last = torch.floor(lo / stride), torch.floor(hi / stride)
    left = (torch.minimum(hi, (first + 1) * stride) - lo).clamp(0, stride)
    right = (hi - torch.maximum(lo, last * stride)).clamp(0, stride)
    return torch.where(last - first >= 2, torch.full_like(left, stride), torch.maximum(left, right))


@torch.no_grad()
def grid_phase_targets(boxes):
    lo, hi = pixel_edges(boxes)
    occupancy = torch.stack([max_cell_overlap(lo, hi, s).prod(1) / (s*s) for s in (4., 8.)], 1)
    return occupancy / occupancy.sum(1, keepdim=True).clamp_min(1e-9)


@torch.no_grad()
def ring_targets(boxes, image_size, stride=4., chunk_size=128):
    """[local, r1,r3,r5] utilities using neighbor collision mass outside the own box.

    Summed intersected neighbor areas are clipped to one; this is a collision
    proxy, not an exact union or visible-area estimate.
    """
    lo, hi = pixel_edges(boxes)
    scale = boxes[:, 2:].prod(1).sqrt()
    height, width = image_size
    limit = boxes.new_tensor([width, height])
    utilities = [torch.full_like(scale, .5)]
    for radius in (1., 3., 5.):
        outer_lo = (boxes[:, :2] - (radius + .5)*stride).clamp_min(0)
        outer_hi = torch.minimum(boxes[:, :2] + (radius + .5)*stride, limit)
        inner_lo = (boxes[:, :2] - (radius - .5)*stride).clamp_min(0)
        inner_hi = torch.minimum(boxes[:, :2] + (radius - .5)*stride, limit)
        outer_area = (outer_hi - outer_lo).clamp_min(0).prod(1)
        inner_area = (inner_hi - inner_lo).clamp_min(0).prod(1)
        own_outer = (torch.minimum(hi, outer_hi) - torch.maximum(lo, outer_lo)).clamp_min(0).prod(1)
        own_inner = (torch.minimum(hi, inner_hi) - torch.maximum(lo, inner_lo)).clamp_min(0).prod(1)
        context_area = (outer_area - inner_area - own_outer + own_inner).clamp_min(0)
        mass = []
        for start in range(0, len(boxes), chunk_size):
            stop = min(start + chunk_size, len(boxes))

            def outside_own(window_lo, window_hi):
                overlap = (torch.minimum(window_hi[start:stop, None], hi[None]) -
                           torch.maximum(window_lo[start:stop, None], lo[None])).clamp_min(0).prod(2)
                triple_lo = torch.maximum(torch.maximum(window_lo[start:stop], lo[start:stop])[:, None], lo[None])
                triple_hi = torch.minimum(torch.minimum(window_hi[start:stop], hi[start:stop])[:, None], hi[None])
                triple = (triple_hi - triple_lo).clamp_min(0).prod(2)
                return (overlap - triple).clamp_min(0)

            collisions = (outside_own(outer_lo, outer_hi) - outside_own(inner_lo, inner_hi)).clamp_min(0)
            collisions[torch.arange(stop-start, device=boxes.device), torch.arange(start, stop, device=boxes.device)] = 0
            mass.append(collisions.sum(1))
        collision = (torch.cat(mass) / context_area.clamp_min(1e-6)).clamp(0, 1)
        utility = torch.log1p(radius*stride / scale.clamp_min(1e-6)) * torch.exp(-4 * collision)
        utilities.append(utility.masked_fill(context_area <= 1e-6, -1e4))
    return torch.stack(utilities, 1).softmax(1)


@torch.no_grad()
def points_outside_boxes(points, boxes, chunk_size=128):
    lo, hi = pixel_edges(boxes)
    masks = []
    for start in range(0, len(points), chunk_size):
        p = points[start:start+chunk_size, None]
        inside = ((p >= lo[None]) & (p <= hi[None])).all(2).any(1)
        masks.append(~inside)
    return torch.cat(masks) if masks else torch.empty(0, dtype=torch.bool, device=points.device)


@torch.no_grad()
def boundary_samples(boxes, all_boxes, image_size):
    """Four inner edge midpoints vs four outer points; never label another object background."""
    height, width = image_size
    margin = (.1 * boxes[:, 2:].prod(1).sqrt()).clamp_min(2.)
    inner = (boxes[:, 2:] / 2 - torch.minimum(boxes[:, 2:] / 4, margin[:, None])).clamp_min(0)
    outer = boxes[:, 2:] / 2 + margin[:, None]
    directions = boxes.new_tensor([[1, 0], [-1, 0], [0, 1], [0, -1]])
    positives = (boxes[:, None, :2] + directions[None] * inner[:, None]).reshape(-1, 2)
    negatives = (boxes[:, None, :2] + directions[None] * outer[:, None]).reshape(-1, 2)
    limit = boxes.new_tensor([width, height])
    keep = ((negatives >= 0) & (negatives <= limit)).all(1) & points_outside_boxes(negatives, all_boxes)
    points = torch.cat((positives, negatives[keep])) / limit
    labels = torch.cat((torch.ones(len(positives), device=boxes.device), torch.zeros(int(keep.sum()), device=boxes.device)))
    return points, labels


def balanced_bce(logits, labels):
    terms = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    groups = [terms[labels == label].mean() for label in (0, 1) if (labels == label).any()]
    return torch.stack(groups).mean() if groups else logits.sum() * 0


def candidate_auxiliary(kind, state, batch, config):
    if state is None or state.get("logits") is None:
        raise RuntimeError("Missing candidate training auxiliary tensors")
    logits = state["logits"]
    if config["weight"] == 0:
        return logits.float().sum() * 0
    if kind == "dcr":
        return dcr_auxiliary(logits, batch, dict(config, supervision="directional"))
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
        mask = valid & (indices == image)
        b, cls = boxes[mask], classes[mask]
        if not len(b):
            continue
        pixels = b * b.new_tensor([w, h, w, h])
        small = pixels[:, 2:].prod(1).sqrt() <= config["small_size"]
        if not small.any():
            continue
        centers = b[small, :2]
        sampled = sample_features(logits[image:image+1], centers)
        if kind == "csr":
            if len(b) < 2:
                continue
            target = ring_targets(pixels, (h, w))[small]
            losses.append(-(target * sampled.log_softmax(1)).sum(1).mean())
        elif kind == "gpr":
            target = grid_phase_targets(pixels[small])
            losses.append(-(target * sampled.log_softmax(1)).sum(1).mean())
        elif kind == "brr":
            points, labels = boundary_samples(pixels[small], pixels, (h, w))
            scores = sample_features(logits[image:image+1], points).flatten()
            losses.append(balanced_bce(scores, labels))
        elif kind == "pcr":
            # Deterministic lattice negatives outside every positive box.
            yy, xx = torch.meshgrid(torch.linspace(.03125, .96875, 16, device=b.device),
                                    torch.linspace(.03125, .96875, 16, device=b.device), indexing="ij")
            background = torch.stack((xx.flatten(), yy.flatten()), 1)
            background = background[points_outside_boxes(background * b.new_tensor([w, h]), pixels)]
            foreground = sampled.flatten()
            bg_logits = sample_features(logits[image:image+1], background).flatten() if len(background) else foreground[:0]
            trust_loss = balanced_bce(torch.cat((foreground, bg_logits)),
                torch.cat((torch.ones_like(foreground), torch.zeros_like(bg_logits))))
            embed = F.normalize(sample_features(state["embedding"][image:image+1], centers), dim=1, eps=1e-6)
            pair_terms, pair_labels = [], []
            selected_cls, selected_pixels = cls[small], pixels[small]
            for start in range(0, len(embed), 128):
                stop = min(start + 128, len(embed))
                distance = (selected_pixels[start:stop, None, :2] - selected_pixels[None, :, :2]).square().sum(2).sqrt()
                ii = torch.arange(start, stop, device=b.device)[:, None]
                jj = torch.arange(len(embed), device=b.device)[None]
                eligible = (jj > ii) & (distance <= 16.)
                if eligible.any():
                    similarity = embed[start:stop] @ embed.T / .2
                    pair_terms.append(similarity[eligible])
                    pair_labels.append((selected_cls[start:stop, None] == selected_cls[None])[eligible].float())
            pair_loss = balanced_bce(torch.cat(pair_terms), torch.cat(pair_labels)) if pair_terms else embed.sum() * 0
            losses.append(.5 * (trust_loss + pair_loss))
        else:
            raise ValueError(f"Unknown candidate: {kind}")
    return torch.stack(losses).mean() if losses else logits.float().sum() * 0


class CandidateDetectionLoss(v8DetectionLoss):
    def __init__(self, model):
        super().__init__(model)
        self.config = dict(model.yaml["candidate"], nc=model.model[-1].nc)
        self.router = next(m for m in model.modules() if isinstance(m, CandidateFusion))
        for key in ("weight", "small_size", "tau", "temperature", "cap"):
            if not math.isfinite(self.config[key]):
                raise ValueError(f"Candidate {key} must be finite")
        if self.config["weight"] < 0 or min(self.config[k] for k in ("small_size", "temperature", "cap")) <= 0:
            raise ValueError("Aux weight nonnegative; size/temperature/cap positive")

    def __call__(self, preds, batch):
        loss, items = super().__call__(preds, batch)
        aux = loss.new_zeros(())
        if self.router.training:
            aux = candidate_auxiliary(self.router.kind, self.router.consume_aux(), batch, self.config) * self.config["weight"]
        return torch.cat((loss, (aux * len(batch["img"])).view(1))), torch.cat((items, aux.detach().view(1)))
