"""Seeded independent label overlays and class/size plots, after data generation."""
import collections
import json
from pathlib import Path
import random
import cv2
import numpy as np
from train_test import ROOT
from run_crop_study import load,save
from run_source_study import DEFAULT


def main():
    d=DEFAULT/'local_dataset';rows=[json.loads(x) for x in (d/'samples.jsonl').read_text().splitlines()]
    chosen=random.Random(42).sample(rows,30)
    save(d/'checks/seeded_selection.json',dict(seed=42,samples=chosen))
    for r in chosen:
        image=cv2.imread(r['image']);label=Path(r['label'])
        labels=np.loadtxt(label,ndmin=2) if label.stat().st_size else np.empty((0,5))
        for cls,cx,cy,w,h in labels:
            b=np.array([cx-w/2,cy-h/2,cx+w/2,cy+h/2])*640;b=np.round(b).astype(int)
            cv2.rectangle(image,tuple(b[:2]),tuple(b[2:]),(0,255,0),1)
            cv2.putText(image,str(int(cls)+1),tuple(np.maximum(b[:2],[0,12])),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,0),1)
        cv2.putText(image,f'{r["kind"]} / {r["split"]} / objects={len(labels)}',(8,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,255),1)
        cv2.imwrite(str(d/'checks'/('seed42_'+Path(r['image']).stem+'.jpg')),image)
    import matplotlib.pyplot as plt
    stats=load(d/'statistics.json');fig,axes=plt.subplots(1,2,figsize=(12,4))
    for split in ['train','dev']:
        axes[0].plot(range(1,11),[next((x['count'] for x in stats['class_counts'] if x['split']==split and x['class_id']==c),0) for c in range(1,11)],marker='o',label=split)
        bins=['<8','8-16','16-32','>=32'];axes[1].plot(bins,[next((x['count'] for x in stats['size_counts'] if x['split']==split and x['size']==b),0) for b in bins],marker='o',label=split)
    axes[0].set_xlabel('Class ID');axes[0].set_ylabel('Crop labels');axes[0].set_yscale('log');axes[0].legend();axes[1].set_xlabel('Native crop sqrt(area) px');axes[1].set_ylabel('Crop labels');axes[1].legend();fig.tight_layout()
    for suffix in ['png','svg','pdf']:fig.savefig(d/('label_distribution.'+suffix),dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
