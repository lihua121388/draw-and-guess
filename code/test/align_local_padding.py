"""One-time alignment of first prepared native crops to centered local padding.

Only owned new dataset outputs change, before certification/dry-run. Historical
images/models/ROI selections never change. All sample hashes and labels updated.
"""
import collections
import json
from pathlib import Path
import cv2
import numpy as np
from run_source_study import DEFAULT
from run_crop_study import load,save
from nscr_experiments import sha256


def main():
    d=DEFAULT/'local_dataset';cfg=load(d/'generation_config.json')
    if (d/'verification.json').exists():raise RuntimeError('Cannot change a certified dataset')
    if (d/'padding_alignment.json').exists():raise RuntimeError('Already aligned')
    rows=[json.loads(x) for x in (d/'samples.jsonl').read_text().splitlines()];modified=0;both_small=0
    for row in rows:
        x,y,xe,ye=row['roi'];w,h=xe-x,ye-y;px,py=(640-w)//2,(640-h)//2
        both_small+=int(max(w,h)<640)
        image=cv2.imread(row['image']);labels=np.loadtxt(row['label'],ndmin=2) if Path(row['label']).stat().st_size else np.empty((0,5))
        # Build process loaded the old top-left padding rule; never infer rule from pixels.
        if px or py:
            new=np.full((640,640,3),114,np.uint8);new[py:py+h,px:px+w]=image[:h,:w]
            if len(labels):labels[:,1]+=px/640;labels[:,2]+=py/640
            if not cv2.imwrite(row['image'],new,[cv2.IMWRITE_PNG_COMPRESSION,1]):raise RuntimeError('PNG update failed')
            Path(row['label']).write_text('\n'.join(str(int(p[0]))+' '+' '.join(f'{v:.9f}' for v in p[1:]) for p in labels)+('\n' if len(labels) else ''),encoding='utf-8')
            row['image_sha256']=sha256(row['image']);row['label_sha256']=sha256(row['label']);modified+=1
        row['training_padding_xy']=[px,py]
        if max(w,h)<640:row['small_source_note']='Native pixels centered padded, not upscaled; S2 inference LetterBox would upscale. This rare case is disclosed.'
    (d/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    cfg['format']='lossless PNG; native RGB outside neutral masks; centered114 padding. No resampling.'
    save(d/'generation_config.json',cfg)
    # Refresh the initial generated label checks so they do not show stale coordinates.
    lookup={Path(r['image']).stem:r for r in rows}
    for p in (d/'checks').glob('*.jpg'):
        if p.stem not in lookup:continue
        r=lookup[p.stem];image=cv2.imread(r['image']);labels=np.loadtxt(r['label'],ndmin=2) if Path(r['label']).stat().st_size else np.empty((0,5))
        for cls,cx,cy,w,h in labels:
            b=np.round(np.array([cx-w/2,cy-h/2,cx+w/2,cy+h/2])*640).astype(int)
            cv2.rectangle(image,tuple(b[:2]),tuple(b[2:]),(0,255,0),1);cv2.putText(image,str(int(cls)+1),tuple(np.maximum(b[:2],[0,12])),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,255,0),1)
        cv2.imwrite(str(p),image)
    save(d/'padding_alignment.json',dict(status='aligned_before_certification',crops=len(rows),modified=modified,
        both_dimensions_below640=both_small,source_pixels_not_resampled=True,
        rule='Centered padding matches S2 square LetterBox when at least one crop dimension is640. Smaller-than640 in both dimensions stays native, explicitly different from inference upscaling.',
        correction='First build was top-left native placement. Corrected owned PNG/labels only, ROI coordinates and source files unchanged.'))
    print('Centered native padding aligned',modified,flush=True)


if __name__=='__main__':main()
