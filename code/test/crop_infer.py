"""FP32 original-image inference using unchanged M0; explicit two-stage NMS and timing."""
import math
import time
import cv2
import numpy as np
from train_test import ROOT
import torch
import torchvision
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from crop_geometry import restore_boxes


def rect_shapes(metadata, imgsz, batch=8, stride=32):
    """Exactly the val loader's aspect-ratio sort / batch rectangle / pad=.5.

    Only original image dimensions are used, never GT. All images executed
    sequentially for true per-image latency, retaining historical batch-8 shapes.
    """
    ar = np.array([m['height']/m['width'] for m in metadata])
    indices = ar.argsort()
    result = {}
    for start in range(0, len(indices), batch):
        group = indices[start:start+batch]
        ratios = ar[group]
        shape = [ratios.max(), 1] if ratios.max() < 1 else [1, 1/ratios.min()] if ratios.min() > 1 else [1, 1]
        h, w = (np.ceil(np.array(shape)*imgsz/stride+.5).astype(int)*stride).tolist()
        for i in group:
            result[metadata[i]['id']] = (h, w)
    return result


def fuse_predictions(predictions, iou=.7, max_det=500):
    """Class-aware original-coordinate NMS. No confidence or class correction / GT input."""
    predictions = np.asarray(predictions, dtype=np.float32).reshape(-1, 6)
    if not len(predictions):
        return predictions.copy(), dict(candidates=0, after_nms=0, final=0, final_cap_removed=0)
    p = torch.from_numpy(np.ascontiguousarray(predictions))
    keep = torchvision.ops.batched_nms(p[:, :4], p[:, 4], p[:, 5], iou)
    count = len(keep)
    keep = keep[:max_det]
    return predictions[keep.numpy()], dict(candidates=len(p), after_nms=count, final=len(keep),
                                           final_cap_removed=max(0, count-max_det))


def boundary_flags(predictions, roi, image_shape, margin=2):
    """Geometry-only interface for a future edge-suppression experiment.

    Returns flags WITHOUT filtering/rescoring predictions. Original-image edges
    are excluded. Current experiment only counts/exports these flags; GT absent.
    """
    p=np.asarray(predictions,dtype=np.float32).reshape(-1,6)
    x,y,xe,ye=roi;h,w=image_shape
    edge=np.zeros(len(p),dtype=bool)
    for idx,boundary,internal in ((0,x,x>0),(1,y,y>0),(2,xe,xe<w),(3,ye,ye<h)):
        if internal:
            edge|=abs(p[:,idx]-boundary)<=margin
    return edge


class CropDetector:
    def __init__(self, checkpoint, device='0'):
        self.device = torch.device('cpu' if device == 'cpu' else f'cuda:{device}')
        wrapper = YOLO(str(checkpoint))
        self.names = wrapper.names
        self.model = wrapper.model.to(self.device).float().eval()
        self.model.fuse(verbose=False)
        self.calls = 0
        self.stride = int(self.model.stride.max())
        with torch.inference_mode():
            for _ in range(30):
                self.model(torch.zeros(1, 3, 640, 640, device=self.device))
        self.sync()

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def warm_shapes(self, shapes):
        with torch.inference_mode():
            for h,w in sorted(set(shapes)):
                for _ in range(3):
                    self.model(torch.zeros(1,3,h,w,device=self.device))
        self.sync()

    def infer(self, image, roi=None, imgsz=640, global_shape=None):
        """BGR original pixels; global follows val resize; local uses square LetterBox.

        Raw decoder multi_label=True matches val. Stage-1 NMS=.7/conf=.001;
        max_nms=30000, local/global intermediate cap=10000 (all drops recorded).
        Only the FINAL original-image fusion is capped to 500.
        """
        self.sync()
        t0 = time.perf_counter()
        h, w = image.shape[:2]
        x, y, xe, ye = roi if roi is not None else (0, 0, w, h)
        crop = np.ascontiguousarray(image[y:ye, x:xe])
        t_crop = time.perf_counter()
        ch, cw = crop.shape[:2]
        if global_shape is not None:
            r = imgsz/max(ch, cw)
            nh, nw = min(math.ceil(ch*r), imgsz), min(math.ceil(cw*r), imgsz)
            resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR) if r != 1 else crop
            # Val scaleup=False. Dataset's first resize always sets long side=imgsz.
            tensor_image = LetterBox(global_shape, auto=False, scaleup=False)(image=resized)
            pad = (round((global_shape[1]-nw)/2-.1), round((global_shape[0]-nh)/2-.1))
            gain = nh/ch  # upstream scale_boxes uses ratio_pad[0][0], faithfully retained
        else:
            gain = imgsz/max(ch, cw)
            nw, nh = round(cw*gain), round(ch*gain)
            pad = (round((imgsz-nw)/2-.1), round((imgsz-nh)/2-.1))
            tensor_image = LetterBox((imgsz, imgsz), auto=False, scaleup=True)(image=crop)
        tensor = torch.from_numpy(np.ascontiguousarray(tensor_image[..., ::-1].transpose(2, 0, 1)))
        tensor = tensor.unsqueeze(0).to(self.device).float()/255
        self.sync()
        t_pre = time.perf_counter()
        with torch.inference_mode():
            raw = self.model(tensor)[0][0].transpose(0, 1)
        self.sync()
        t_forward = time.perf_counter()
        boxes = ops.xywh2xyxy(raw[:, :4])
        bi, ci = torch.where(raw[:, 4:14] > .001)
        candidates = torch.cat((boxes[bi], raw[bi, ci+4, None], ci[:, None].float()), 1)
        before = len(candidates)
        if before > 30000:
            candidates = candidates[candidates[:, 4].argsort(descending=True)[:30000]]
        keep = torchvision.ops.batched_nms(candidates[:, :4], candidates[:, 4], candidates[:, 5], .7)
        after_nms = len(keep)
        candidates = candidates[keep[:10000]].detach().cpu().numpy()
        self.sync()
        t_nms = time.perf_counter()
        candidates[:, :4] = restore_boxes(candidates[:, :4], gain, pad, (x, y), (h, w))
        # Local NMS boxes clipped in ROI coordinates BEFORE translation, like predict.
        candidates[:, [0, 2]] = candidates[:, [0, 2]].clip(x, xe)
        candidates[:, [1, 3]] = candidates[:, [1, 3]].clip(y, ye)
        valid = (candidates[:, 2] > candidates[:, 0]) & (candidates[:, 3] > candidates[:, 1])
        invalid = int((~valid).sum())
        candidates = candidates[valid]
        edge = boundary_flags(candidates,(x,y,xe,ye),(h,w))
        elapsed = (time.perf_counter()-t0)*1000
        self.calls += 1
        info = dict(roi=[x,y,xe,ye], input_shape=list(tensor_image.shape[:2]),
            gain=gain, padding=list(pad), multi_label=True,
            pre_nms_candidates=before, max_nms_removed=max(0, before-30000),
            after_roi_nms=after_nms, intermediate_cap_removed=max(0, after_nms-10000),
            degenerate_removed=invalid, internal_boundary_candidates=int(edge.sum()),
            mapped_candidates=len(candidates),boundary_prediction_indices=np.flatnonzero(edge).tolist(),
            internal_boundary_high_conf=int((edge & (candidates[:,4]>=.25)).sum()),
            time_ms=dict(crop=(t_crop-t0)*1000, preprocess=(t_pre-t_crop)*1000,
                forward=(t_forward-t_pre)*1000, roi_nms=(t_nms-t_forward)*1000,
                map=(time.perf_counter()-t_nms)*1000, total=elapsed))
        return candidates, info
