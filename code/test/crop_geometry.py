"""Original-image tiling, reversible coordinates and budgeted GT-only Oracle geometry."""
import numpy as np


def axis_starts(length, window=640, overlap=.2):
    if length <= 0 or window <= 0 or not 0 <= overlap < 1:
        raise ValueError("Positive dimensions and overlap in [0,1) required")
    if length <= window:
        return [0]
    step = max(1, round(window * (1 - overlap)))
    starts = list(range(0, length - window + 1, step))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


def fixed_windows(width, height, window=640, overlap=.2):
    return [(x, y, min(x + window, width), min(y + window, height))
            for y in axis_starts(height, window, overlap)
            for x in axis_starts(width, window, overlap)]


def restore_boxes(boxes, gain, padding, offset, image_shape):
    """Invert letterbox then ROI translation; accepts empty predictions; never uses GT."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - padding[0]) / gain + offset[0]
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - padding[1]) / gain + offset[1]
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, image_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, image_shape[0])
    return boxes


def coverage_matrix(windows, boxes, width, height, margin=8):
    """Full-box containment with margin only at INTERNAL crop edges, not image edges."""
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    rois = np.asarray(windows, dtype=float).reshape(-1, 4)
    if not len(rois):
        return np.zeros((0, len(boxes)), dtype=bool)
    lower = rois[:, :2] + margin * (rois[:, :2] > 0)
    upper = rois[:, 2:] - margin * (rois[:, 2:] < [width, height])
    return ((boxes[None, :, :2] >= lower[:, None]).all(2)
            & (boxes[None, :, 2:] <= upper[:, None]).all(2))


def oracle_windows(width, height, boxes, budget=4, window=640, margin=8):
    """GT Oracle ONLY: greedy maximum new small-GT coverage, no class/confidence input.

    Finite candidates = fixed grid + each small GT's center/top-left/bottom-right
    anchored windows. Prefer maximum new fully contained sqrt(wh)<32 GT count;
    ties favor newly covered valid targets, then lexicographic ROI coordinates.
    Stop at zero new SMALL targets. K=1/2/4 are nested prefixes, not unbounded crops.
    """
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    size = np.sqrt(np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(1))
    small = (size > 0) & (size < 32)
    if budget < 0 or margin < 0:
        raise ValueError("Nonnegative budget and margin required")
    if not small.any() or budget == 0:
        return [], dict(candidate_count=0, new_small_counts=[])
    rw, rh = min(width, window), min(height, window)
    candidates = set(fixed_windows(width, height, window, .2))
    for b in boxes[small]:
        cx, cy = (b[:2] + b[2:]) / 2
        for x, y in ((cx-rw/2, cy-rh/2), (b[0]-margin, b[1]-margin),
                     (b[2]+margin-rw, b[3]+margin-rh)):
            x = int(np.clip(round(x), 0, width-rw))
            y = int(np.clip(round(y), 0, height-rh))
            candidates.add((x, y, x+rw, y+rh))
    candidates = sorted(candidates)
    covers = coverage_matrix(candidates, boxes, width, height, margin)
    remaining = np.ones(len(boxes), dtype=bool)
    selected, gains = [], []
    for _ in range(budget):
        gain_small = (covers & remaining & small).sum(1)
        best_gain = int(gain_small.max())
        if best_gain == 0:
            break
        choices = np.flatnonzero(gain_small == best_gain)
        gain_all = (covers[choices] & remaining).sum(1)
        best = int(choices[np.argmax(gain_all)])
        selected.append(candidates[best])
        gains.append(best_gain)
        remaining &= ~covers[best]
    return selected, dict(candidate_count=len(candidates), new_small_counts=gains,
                          objective="new full-containment sqrt(wh)<32 targets", margin=margin)


def union_area(windows):
    """Exact union of axis-aligned windows; repeated area and unique coverage differ."""
    if not windows:
        return 0
    xs = sorted({r[i] for r in windows for i in (0, 2)})
    area = 0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted((r[1], r[3]) for r in windows if r[0] < right and r[2] > left)
        span = 0
        end = -1
        for lo, hi in intervals:
            span += max(0, hi - max(lo, end))
            end = max(end, hi)
        area += (right-left) * span
    return area
