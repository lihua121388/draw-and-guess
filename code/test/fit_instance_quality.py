"""Quality-only fit entry. Requires prepared source-isolated TRAIN-SIDE data; no YOLO training."""
import argparse
import csv
import json
from pathlib import Path
import random
import numpy as np
from train_test import ROOT
import torch
from instance_quality import FEATURES,VARIANTS,QualityMLP,feature_mask,quality_loss
from run_crop_study import load,save
from nscr_experiments import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True,help='Directory with metadata.json and fit/early_stop/calibration.npz')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--variant',choices=VARIANTS,required=True)
    p.add_argument('--protocol',choices=['upstream_seen_development','upstream_out_of_fold','independent_external'],required=True)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch',type=int,default=4096);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--patience',type=int,default=5);p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='0');args=p.parse_args()
    meta=load(args.dataset/'metadata.json')
    if meta['features']!=FEATURES or meta['protocol']!=args.protocol:raise RuntimeError('Feature/protocol metadata mismatch')
    if not meta.get('labels_from_training_or_external_sources_only') or meta.get('validation_GT_used'):raise RuntimeError('Validation labels forbidden')
    groups=[set(meta['groups'][s]) for s in ['fit','early_stop','calibration']]
    if any(groups[i]&groups[j] for i in range(3) for j in range(i+1,3)):raise RuntimeError('Original-source group overlap')
    val_names={m['file_name'] for m in load(ROOT/'datasets/nscr_visdrone/val_coco.json')['images']}
    if val_names&{Path(x).name for s in ['fit','early_stop','calibration'] for x in meta['source_images'][s]}:raise RuntimeError('548 validation membership leaked into quality data')
    if args.protocol!='upstream_seen_development' and not meta.get('upstream_independence_verified'):raise RuntimeError('Missing upstream independence evidence')
    data={s:np.load(args.dataset/(s+'.npz')) for s in ['fit','early_stop']}
    for split,arr in data.items():
        x=arr['features'];y=arr['targets'];w=arr['weights']
        if x.ndim!=2 or x.shape[1]!=len(FEATURES) or y.shape!=(len(x),2) or w.shape!=(len(x),):raise RuntimeError('Dataset shape mismatch')
        if not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all() or (y<0).any() or (y>1).any() or (y[:,1]>y[:,0]+1e-6).any() or (w<0).any() or not np.isfinite(w).all():raise RuntimeError('Invalid targets/weights')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(4)
    target=torch.device('cpu' if args.device=='cpu' else f'cuda:{args.device}')
    mean=data['fit']['features'].mean(0);std=data['fit']['features'].std(0).clip(1e-3)
    binary=[i for i,n in enumerate(FEATURES) if n.startswith('class_') or n.endswith('_available') or n.endswith('_nearest') or n in ['class_agrees','supported','observed_no_support']]
    mean[binary]=0;std[binary]=1
    mask=feature_mask(args.variant);arrays={}
    for split,arr in data.items():
        arrays[split]=tuple(torch.from_numpy(np.ascontiguousarray(v,dtype=np.float32)).to(target) for v in [((arr['features']-mean)/std)*mask,arr['targets'],arr['weights']])
    args.output.mkdir(parents=True,exist_ok=False)
    gain=args.variant not in ['C2','C3a','C4_no_gain_loss']
    config=dict(arguments=vars(args),gain_loss_enabled=gain,protocol=meta['protocol'],source_groups=meta['groups'],
        metadata_sha256=sha256(args.dataset/'metadata.json'),dataset_sha256={s:sha256(args.dataset/(s+'.npz')) for s in ['fit','early_stop','calibration']},
        YOLO_frozen=True,validation_GT_used=False,calibration_used_for_early_stop=False)
    save(args.output/'config.json',json.loads(json.dumps(config,default=str)))
    model=QualityMLP().to(target);optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.0001)
    best=float('inf');bad=0;history=[]
    for epoch in range(1,args.epochs+1):
        model.train();x,y,w=arrays['fit'];order=torch.randperm(len(x),device=target);loss_total=0;weight_total=0
        for ids in order.split(args.batch):
            optimizer.zero_grad(set_to_none=True);loss=quality_loss(model(x[ids]),y[ids],w[ids],gain)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite fit loss')
            loss.backward();optimizer.step();mass=float(w[ids].sum());loss_total+=float(loss.detach())*mass;weight_total+=mass
        model.eval();vx,vy,vw=arrays['early_stop'];eval_sum=0;eval_mass=0
        with torch.inference_mode():
            for start in range(0,len(vx),args.batch):
                sl=slice(start,start+args.batch);mass=float(vw[sl].sum());eval_sum+=float(quality_loss(model(vx[sl]),vy[sl],vw[sl],gain))*mass;eval_mass+=mass
        row=dict(epoch=epoch,fit_loss=loss_total/max(1,weight_total),early_stop_loss=eval_sum/max(1,eval_mass));history.append(row)
        with (args.output/'results.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerows(history)
        print(args.variant,row,flush=True)
        state=dict(state_dict=model.cpu().state_dict(),mean=mean,std=std,mask=mask,features=FEATURES,epoch=epoch,variant=args.variant,gain_loss=gain,seed=args.seed,protocol=meta['protocol'])
        torch.save(state,args.output/'last.pt');model.to(target)
        if row['early_stop_loss']<best:
            best=row['early_stop_loss'];bad=0;torch.save(state,args.output/'best.pt')
        else:bad+=1
        if bad>=args.patience:break
    save(args.output/'completion.json',dict(status='fit_complete',epochs=len(history),best_early_stop_loss=best,
        beta_selected=False,probability_calibration_done=False,validation_evaluation_done=False,best_sha256=sha256(args.output/'best.pt')))


if __name__=='__main__':main()
