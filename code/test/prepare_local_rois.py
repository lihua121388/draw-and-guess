"""Training-split-only native RGB crops; unchanged S2 selector. NO training."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
from train_test import ROOT
import torch
import yaml
from run_crop_study import load,save
from nscr_experiments import sha256
from crop_infer import CropDetector,rect_shapes,fuse_predictions
from crop_proposals import predict_roi_windows,weighted_roi_windows
from crop_geometry import fixed_windows,coverage_matrix
from prepare_data import clip_box,inside_ignore,label_line
from run_source_study import DEFAULT


def read_annotations(path,width,height):
    positives=[];ignored=[];invalid=0
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():continue
        v=list(map(float,line.strip().rstrip(',').split(',')[:8]))
        b=clip_box(v[:4],width,height)
        if b is None or not np.isfinite(v).all():invalid+=1;continue
        c=int(v[5])
        if v[4]<=0 or c in (0,11):ignored.append(b)
        elif 1<=c<=10 and c==v[5]:positives.append(dict(class_id=c-1,bbox=b,occlusion=v[7],truncation=v[6]))
        else:invalid+=1
    positives=[p for p in positives if not inside_ignore(p['bbox'],ignored)]
    return positives,ignored,invalid


def crop_labels(image,roi,positives,ignored,visible_threshold=.70):
    """Clip, retain centered targets with >=70% area and sides>=2px.

    Stock YOLO has no ignore loss mask. Unknown/ignored/truncated-discarded
    objects are neutral-filled (114) in training ONLY, not labelled background.
    Positives with >=50% area masked are also discarded/neutral-filled to a fixed
    point. This is a disclosed pixel-masking approximation, not loss masking.
    """
    x,y,xe,ye=roi;cw,ch=xe-x,ye-y
    pad_x,pad_y=(640-cw)//2,(640-ch)//2
    result=np.full((640,640,3),114,np.uint8);result[pad_y:pad_y+ch,pad_x:pad_x+cw]=image[y:ye,x:xe]
    mask=np.zeros((640,640),np.uint8);retained=[];counts=collections.Counter()
    def rectangle(box):
        b=np.asarray(box,float);b[2:]+=b[:2]
        b[:2]=np.maximum(b[:2],[x,y]);b[2:]=np.minimum(b[2:],[xe,ye])
        b-=np.array([x-pad_x,y-pad_y,x-pad_x,y-pad_y]);return b
    def paint(b):
        x1,y1=np.maximum(0,np.floor(b[:2]).astype(int));x2,y2=np.minimum(640,np.ceil(b[2:]).astype(int))
        if x2>x1 and y2>y1:mask[y1:y2,x1:x2]=1
    for b in ignored:paint(rectangle(b));counts['ignore_regions']+=1
    for p in positives:
        b=rectangle(p['bbox']);wh=b[2:]-b[:2]
        if (wh<=0).any():continue
        counts['intersecting_targets']+=1
        area=p['bbox'][2]*p['bbox'][3];ratio=float(wh.prod()/area)
        cx=p['bbox'][0]+p['bbox'][2]/2;cy=p['bbox'][1]+p['bbox'][3]/2
        center=x<=cx<xe and y<=cy<ye
        if center and ratio>=visible_threshold and wh.min()>=2:
            retained.append((p['class_id'],b,ratio));counts['retained_partial' if ratio<1-1e-7 else 'retained_full']+=1
        else:paint(b);counts['discarded_boundary_or_tiny']+=1
    while True:
        kept=[];removed=False
        for cls,b,ratio in retained:
            ix1,iy1=np.maximum(0,np.floor(b[:2]).astype(int));ix2,iy2=np.minimum(640,np.ceil(b[2:]).astype(int))
            if mask[iy1:iy2,ix1:ix2].mean()>=.5:
                paint(b);removed=True;counts['discarded_mask_overlap']+=1
            else:kept.append((cls,b,ratio))
        retained=kept
        if not removed:break
    result[mask.astype(bool)]=114
    labels=[]
    for cls,b,ratio in retained:labels.append((cls,[b[0],b[1],b[2]-b[0],b[3]-b[1]]))
    counts['retained_final']=len(labels);counts['masked_pixels']=int(mask.sum());counts['padded_pixels']=640**2-ch*cw
    return result,labels,dict(counts)


def group_split(metadata):
    groups=sorted({Path(m['file_name']).stem.split('_')[0] for m in metadata})
    ranked=sorted(groups,key=lambda g:hashlib.sha256(('42:'+g).encode()).hexdigest())
    held=set(ranked[:max(1,round(.1*len(groups)))])
    return held,dict(seed=42,group_key='filename first underscore field (source sequence)',
        source_sequences=len(groups),dev_sequences=sorted(held),
        note='Holdout only for local fine-tuning/checkpoint selection. M0 already trained ALL official train images; '
             'this is not an independent detector-generalization test. Official val never makes training crops.')


def supplemental(width,height,positives,actual,class_counts):
    boxes=np.array([p['bbox'] for p in positives],float).reshape(-1,4);boxes[:,2:]+=boxes[:,:2]
    if not len(boxes):return []
    size=np.sqrt((boxes[:,2:]-boxes[:,:2]).prod(1))
    uncovered=~coverage_matrix(actual,boxes,width,height,8).any(0) if actual else np.ones(len(boxes),bool)
    rarity=np.array([np.sqrt(max(class_counts.values())/class_counts[p['class_id']]) for p in positives])
    difficult=np.array([1+float(p['occlusion']>0)+float(p['truncation']>0) for p in positives])
    weights=rarity*difficult*np.where(size<8,4.,np.where(size<16,2.,1.))
    weights*=uncovered&((size<16)|(rarity>2)|(difficult>1))
    roi,_=weighted_roi_windows(width,height,boxes,weights,1)
    return [r for r in roi if list(r) not in actual]


def negative_window(width,height,positives,predictions):
    boxes=np.array([p['bbox'] for p in positives],float).reshape(-1,4);boxes[:,2:]+=boxes[:,:2]
    best=None;bestscore=-1
    for roi in fixed_windows(width,height):
        x,y,xe,ye=roi
        if len(boxes) and (np.maximum(0,np.minimum(boxes[:,2:],[xe,ye])-np.maximum(boxes[:,:2],[x,y])).prod(1)>0).any():continue
        center=(predictions[:,:2]+predictions[:,2:4])/2
        inside=(center[:,0]>=x)&(center[:,0]<xe)&(center[:,1]>=y)&(center[:,1]<ye)
        score=float(predictions[inside&(predictions[:,4]>=.1),4].sum())
        if score>bestscore:bestscore=score;best=roi
    return best,bestscore


def build(output,device):
    a=load(output/'audit.json');data_root=Path(a['data_root']);metadata=load(data_root/'train_coco.json')['images']
    paths=[Path(p) for p in (data_root/'train.txt').read_text().splitlines() if p.strip()];by_name={p.name:p for p in paths}
    if len(paths)!=6471 or {m['file_name'] for m in metadata}!=set(by_name):raise RuntimeError('Official train membership mismatch')
    ann_root=Path(load(data_root/'preparation.json')['splits']['train']['original_annotations'])
    if len({p.name for p in paths}&{m['file_name'] for m in load(output/'image_manifest.json')}):raise RuntimeError('Train/val filename leak')
    d=output/'local_dataset';d.mkdir(exist_ok=False)
    if shutil.disk_usage(d).free<30*1024**3:raise RuntimeError('Need at least 30GiB free for lossless crops and metadata')
    for sub in ('images','labels','checks'):(d/sub).mkdir()
    held,split=group_split(metadata);save(d/'split.json',split)
    annotations={};class_counts=collections.Counter();digest=hashlib.sha256()
    for m in metadata:
        p=ann_root/(Path(m['file_name']).stem+'.txt');digest.update(p.name.encode());digest.update(p.read_bytes())
        positives,ignored,invalid=read_annotations(p,m['width'],m['height']);annotations[m['id']]=(positives,ignored,invalid)
        class_counts.update(p['class_id'] for p in positives)
    save(d/'generation_config.json',dict(source_split='official train ONLY',images=6471,checkpoint=a['checkpoint'],checkpoint_sha256=sha256(a['checkpoint']),
        annotation_sha256=digest.hexdigest(),selector=a['selector'],input_size=640,actual_budget=2,supplemental_budget=1,
        negative_sampling='Every fourth source by hash, highest frozen-M0 conf mass among grid windows with zero positive intersection',
        annotation_policy='center inside, visible area>=.70, clipped sides>=2; neutral fill114 for ignore/discarded objects; masked positive area>=.5 removed',
        resampling=False,format='lossless PNG; native RGB values retained outside explicit neutral masks; smaller crops centered pad114',
        class_counts_original=dict(class_counts),supplemental='Uncovered tiny/rare/occluded/truncated training GT only; separate from actual S2 selection'))
    detector=CropDetector(a['checkpoint'],device);shapes=rect_shapes(metadata,640)
    detector.warm_shapes(list(shapes.values()))
    manifests={'train':[],'dev':[]};stats=collections.Counter();categories=collections.Counter();sizes=collections.Counter();check_counts=collections.Counter();samples=[];start=time.perf_counter()
    with (d/'train_global_predictions.jsonl').open('w',encoding='utf-8') as predictions,(d/'samples.jsonl').open('w',encoding='utf-8') as samplesfile:
        for index,m in enumerate(metadata):
            source=by_name[m['file_name']];image=cv2.imread(str(source))
            if image is None or image.shape[:2]!=(m['height'],m['width']):raise RuntimeError('Train shape mismatch')
            positives,ignored,invalid=annotations[m['id']];g,info=detector.infer(image,imgsz=640,global_shape=shapes[m['id']]);g,_=fuse_predictions(g,.7,500)
            rois,selection=predict_roi_windows(m['width'],m['height'],g,2)
            predictions.write(json.dumps(dict(image_id=m['id'],file_name=m['file_name'],global_predictions=g.tolist(),rois=rois,selection=selection))+'\n')
            jobs=[('actual',r) for r in rois]+[('supplemental',list(r)) for r in supplemental(m['width'],m['height'],positives,rois,class_counts)]
            if int(hashlib.sha256(('42:'+m['file_name']).encode()).hexdigest()[:8],16)%4==0:
                neg,confidence=negative_window(m['width'],m['height'],positives,g)
                if neg is not None and list(neg) not in [r for _,r in jobs]:jobs.append(('negative',list(neg)))
            subset='dev' if Path(m['file_name']).stem.split('_')[0] in held else 'train'
            for j,(kind,roi) in enumerate(jobs):
                crop,labels,counters=crop_labels(image,roi,positives,ignored)
                name=f'{Path(m["file_name"]).stem}_{kind}_{j}'
                destination=d/'images'/(name+'.png');labelpath=d/'labels'/(name+'.txt')
                if not cv2.imwrite(str(destination),crop,[cv2.IMWRITE_PNG_COMPRESSION,1]):raise RuntimeError('PNG write failed')
                labelpath.write_text('\n'.join(label_line(c,b,640,640) for c,b in labels)+('\n' if labels else ''),encoding='utf-8')
                manifests[subset].append(destination.resolve().as_posix());stats.update({f'{subset}_{kind}':1,f'{subset}_images':1,f'{subset}_empty':int(not labels)})
                stats.update(counters)
                for c,b in labels:
                    categories[(subset,c)]+=1;size=np.sqrt(b[2]*b[3]);sizes[(subset,('<8','8-16','16-32','>=32')[np.searchsorted([8,16,32],size,side='right')])]+=1
                row=dict(source_image=str(source),source_annotation=str(ann_root/(source.stem+'.txt')),image=str(destination),label=str(labelpath),
                    source_image_id=m['id'],roi=roi,kind=kind,split=subset,objects=len(labels),image_sha256=sha256(destination),label_sha256=sha256(labelpath),**counters)
                samplesfile.write(json.dumps(row)+'\n')
                if check_counts[kind]<12:
                    marked=crop.copy()
                    for c,b in labels:
                        x,y,w,h=b;cv2.rectangle(marked,(round(x),round(y)),(round(x+w),round(y+h)),(0,255,0),1)
                        cv2.putText(marked,str(c+1),(round(x),max(12,round(y))),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,255,0),1)
                    cv2.imwrite(str(d/'checks'/(name+'.jpg')),marked);check_counts[kind]+=1
            if (index+1)%100==0 or index+1==len(metadata):
                save(d/'progress.json',dict(images_done=index+1,images_total=len(metadata),crops=sum(len(v) for v in manifests.values()),elapsed_seconds=time.perf_counter()-start))
                print(f'Prepare ROI {index+1}/6471, crops={sum(len(v) for v in manifests.values())}',flush=True)
    del detector;torch.cuda.empty_cache()
    for subset,values in manifests.items():(d/(subset+'.txt')).write_text('\n'.join(values)+'\n',encoding='utf-8')
    config=dict(path=str(d.resolve()),train='train.txt',val='dev.txt',names={c['id']-1:c['name'] for c in a['categories']})
    (d/'dataset.yaml').write_text(yaml.safe_dump(config,sort_keys=False,allow_unicode=True),encoding='utf-8')
    save(d/'statistics.json',dict(status='generated',counts=dict(stats),class_counts=[dict(split=s,class_id=c+1,count=v) for (s,c),v in sorted(categories.items())],
        size_counts=[dict(split=s,size=b,count=v) for (s,b),v in sorted(sizes.items())],check_images=dict(check_counts),seconds=time.perf_counter()-start,
        checkpoint_unchanged=sha256(a['checkpoint'])==a['immutable']['checkpoint_sha256']))


def verify(output):
    d=output/'local_dataset';rows=[json.loads(x) for x in (d/'samples.jsonl').read_text().splitlines()]
    groups={s:set() for s in ('train','dev')};counts=collections.Counter();digest=hashlib.sha256()
    for index,r in enumerate(rows):
        image=cv2.imread(r['image']);labels=np.loadtxt(r['label'],ndmin=2) if Path(r['label']).stat().st_size else np.empty((0,5))
        if image is None or image.shape!=(640,640,3):raise RuntimeError('Crop must be 640 RGB')
        if len(labels) and (not np.isfinite(labels).all() or not np.isin(labels[:,0],range(10)).all() or (labels[:,1:]<0).any() or (labels[:,1:]>1).any() or (labels[:,3:]<=0).any()):raise RuntimeError('Invalid YOLO label')
        if len(labels) and ((labels[:,1:3]-labels[:,3:]/2< -1e-7).any() or (labels[:,1:3]+labels[:,3:]/2>1+1e-7).any()):raise RuntimeError('Box outside crop')
        if len(labels)!=r['objects'] or sha256(r['image'])!=r['image_sha256'] or sha256(r['label'])!=r['label_sha256']:raise RuntimeError('Artifact corruption')
        groups[r['split']].add(Path(r['source_image']).stem.split('_')[0]);counts[r['split']]+=1
        digest.update((r['image_sha256']+r['label_sha256']).encode())
    if groups['train']&groups['dev']:raise RuntimeError('Source sequence leaks')
    checks=list((d/'checks').glob('*.jpg'))
    if len(checks)<30:raise RuntimeError('At least30 label overlays required')
    save(d/'verification.json',dict(status='complete',crops=len(rows),splits=dict(counts),all_640_rgb=True,labels_valid=True,
        source_group_leakage=False,official_val_used_for_training=False,overlays=len(checks),artifacts_sha256=digest.hexdigest()))
    print('Dataset verification complete',counts,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--stage',choices=['build','verify'],required=True);p.add_argument('--device',default='0');args=p.parse_args()
    cv2.setNumThreads(2);torch.set_num_threads(4)
    if args.stage=='build':build(args.output,args.device)
    else:verify(args.output)


if __name__=='__main__':main()
