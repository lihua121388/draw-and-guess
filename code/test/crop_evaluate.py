"""One offline scoring path for ALL crop schemes; no model inference or GT-driven filtering."""
import collections
import json
from pathlib import Path
import numpy as np
from train_test import ROOT
import torch
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import DetMetrics, box_iou
from evaluate_nscr import coco_metrics, box_iou_numpy
from resolution_diagnosis import diagnose, match_indices
from visdrone_protocol import evaluate as visdrone_reference


def ultra_metrics(gt, predictions):
    """Original positives and upstream Ultralytics IoU matching / AP integration.

    No extra ignore filtering: mirrors YOLO validation's converted positives.
    Supplemental COCO and fixed-size diagnosis retain their separate ignore rules.
    """
    anns, preds = collections.defaultdict(list), collections.defaultdict(list)
    for a in gt['annotations']:
        if not a.get('iscrowd', 0):
            anns[a['image_id']].append(a)
    for p in predictions:
        preds[p['image_id']].append(p)
    names = {c['id']-1: c['name'] for c in gt['categories']}
    metrics = DetMetrics(names=names)
    # Avoid constructing a validator (which creates an output directory).
    class Matcher:
        iouv = torch.linspace(.5, .95, 10)
    matcher = Matcher()
    for im in gt['images']:
        a = anns[im['id']]
        b = np.array([x['bbox'] for x in a], dtype=np.float32).reshape(-1, 4)
        b[:, 2:] += b[:, :2]
        cls = np.array([x['category_id']-1 for x in a], dtype=np.float32)
        p = preds[im['id']]
        d = np.array([[*x['bbox'], x['score'], x['category_id']-1] for x in p], dtype=np.float32).reshape(-1, 6)
        d[:, 2:4] += d[:, :2]
        correct = np.zeros((len(d), 10), dtype=bool)
        if len(b) and len(d):
            iou = box_iou(torch.from_numpy(b), torch.from_numpy(d[:, :4]))
            correct = DetectionValidator.match_predictions(matcher, torch.from_numpy(d[:, 5]),
                torch.from_numpy(cls), iou).numpy()
        metrics.update_stats(dict(tp=correct, conf=d[:, 4], pred_cls=d[:, 5],
                                  target_cls=cls, target_img=np.unique(cls)))
    metrics.process(plot=False)
    return dict(metrics=metrics.results_dict, per_class=[dict(class_id=int(cid), name=names[int(cid)],
        Precision=float(metrics.box.p[k]), Recall=float(metrics.box.r[k]),
        AP50=float(metrics.box.ap50[k]), AP50_95=float(metrics.box.ap[k]))
        for k, cid in enumerate(metrics.box.ap_class_index)])


def fixed_conf_quality(gt,predictions):
    """Retrospective FP proxies at PRESET conf.25/IoU.5; never edits predictions."""
    anns,regions,preds=(collections.defaultdict(list) for _ in range(3))
    for a in gt['annotations']:
        (regions if a.get('iscrowd') else anns)[a['image_id']].append(a)
    for p in predictions:preds[p['image_id']].append(p)
    counts=dict(TP=0,FP=0,targets=0,duplicates_proxy=0,wrong_class_proxy=0,
                localization_proxy=0,background_proxy=0,ignored_high_conf=0)
    for m in gt['images']:
        boxes=np.array([a['bbox'] for a in anns[m['id']]],dtype=float).reshape(-1,4)
        classes=np.array([a['category_id'] for a in anns[m['id']]])
        pp=[p for p in preds[m['id']] if p['score']>=.25]
        d=np.array([[*p['bbox'],p['score'],p['category_id']] for p in pp],dtype=float).reshape(-1,6)
        ignore=np.unique(np.array([a['bbox'] for a in regions[m['id']]],dtype=float).reshape(-1,4),axis=0)
        if len(d) and len(ignore):
            inter=np.maximum(0,np.minimum(d[:,None,:2]+d[:,None,2:4],ignore[None,:,:2]+ignore[None,:,2:4])-
                np.maximum(d[:,None,:2],ignore[None,:,:2])).prod(2)
            keep=(inter/np.maximum(d[:,2:4].prod(1)[:,None],1e-9)).max(1)<.5
            counts['ignored_high_conf']+=int((~keep).sum());d=d[keep]
        boxes[:,2:]+=boxes[:,:2];d[:,2:4]+=d[:,:2]
        iou=box_iou_numpy(boxes,d[:,:4])
        matches=match_indices(iou,classes,d)
        fp=np.ones(len(d),bool);fp[matches[matches>=0]]=False
        same=classes[:,None]==d[None,:,5]
        duplicate=fp&((iou>=.5)&same).any(0)
        wrong=fp&~duplicate&((iou>=.5)&~same).any(0)
        loc=fp&~duplicate&~wrong&((iou>=.1)&same).any(0)
        background=fp&~duplicate&~wrong&~loc
        for k,v in [('TP',matches>=0),('FP',fp),('duplicates_proxy',duplicate),('wrong_class_proxy',wrong),
                    ('localization_proxy',loc),('background_proxy',background)]:counts[k]+=int(v.sum())
        counts['targets']+=len(boxes)
    counts.update(precision=counts['TP']/max(1,counts['TP']+counts['FP']),recall=counts['TP']/max(1,counts['targets']),
        protocol='Fixed conf.25/IoU.5 one-to-one class-aware, converted positive GT, individual ignore-region IoA.5. '
        'FP categories are retrospective proxies, not established causes; neither GT nor these flags affect outputs.')
    return counts


def evaluate_saved(directory, gt, annotation_path, original_annotations, reference=True):
    directory = Path(directory)
    predictions = json.loads((directory/'coco_predictions.json').read_text(encoding='utf-8'))
    print(f'Scoring {directory.name}: {len(predictions)} predictions', flush=True)
    u = ultra_metrics(gt, predictions)
    (directory/'ultralytics_offline.json').write_text(json.dumps(u, indent=2), encoding='utf-8')
    c = coco_metrics(annotation_path, predictions, [m['id'] for m in gt['images']], 500)
    diag = diagnose(gt, predictions, 640)
    # Input-size bins would be meaningless for a global+ROI composite.
    diag.pop('input_size', None)
    (directory/'error_diagnosis.json').write_text(json.dumps(diag, indent=2), encoding='utf-8')
    result = dict(images=len(gt['images']), ultralytics=u['metrics'], per_class=u['per_class'],
                  coco_supplemental=c, original_size=diag['original_size'],
                  density_size=diag['density_size'], all_diagnostics=diag['all'],
                  protocol='Shared offline original-coordinate Ultralytics AP; COCO small and VisDrone reference separate.')
    if reference:
        pred_dir = directory/'visdrone_predictions'
        pred_dir.mkdir(exist_ok=True)  # deterministic derivative exports; supports interrupted scoring
        lookup = collections.defaultdict(list)
        for p in predictions:
            lookup[p['image_id']].append(p)
        for m in gt['images']:
            rows = sorted(lookup[m['id']], key=lambda x: -x['score'])
            lines = [','.join(f'{v:.6f}' for v in (*p['bbox'], p['score']))+f",{p['category_id']},-1,-1" for p in rows]
            (pred_dir/(Path(m['file_name']).stem+'.txt')).write_text('\n'.join(lines), encoding='utf-8')
        result['visdrone_reference_python'] = visdrone_reference(original_annotations, pred_dir,
            gt['images'], directory/'visdrone_reference_python.json')
    (directory/'evaluation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
