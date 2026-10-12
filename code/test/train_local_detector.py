"""M0-identical local detector. Default is dry-run; formal training is explicit."""
import argparse
import copy
import json
from pathlib import Path
import time
import cv2
import numpy as np
from train_test import ROOT
import torch
import yaml
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset,build_dataloader
from nscr_experiments import sha256
from run_crop_study import load,save
from run_source_study import DEFAULT


def configuration(output):
    a=load(output/'audit.json');d=output/'local_dataset'
    cfg=dict(model=a['checkpoint'],data=str((d/'dataset.yaml').resolve()),epochs=30,batch=8,workers=4,imgsz=640,
        device='0',optimizer='AdamW',lr0=.0001,lrf=.1,momentum=.9,weight_decay=.0005,
        warmup_epochs=1.,warmup_bias_lr=.0001,cos_lr=True,seed=42,deterministic=True,
        amp=True,cache=False,mosaic=0.,mixup=0.,copy_paste=0.,scale=.1,translate=.05,
        degrees=0.,shear=0.,perspective=0.,flipud=0.,fliplr=.5,hsv_h=.015,hsv_s=.3,hsv_v=.2,
        close_mosaic=0,multi_scale=False,rect=False,patience=31,save_period=5,max_det=500,iou=.7,
        project=str((output/'local_training').resolve()),name='A1_seed42',exist_ok=False,plots=True,val=True,
        resume=False,pretrained=True)
    path=output/'local_train.yaml'
    if path.exists():
        if yaml.safe_load(path.read_text())!=cfg:raise RuntimeError('Existing locked config differs')
    else:path.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
    return cfg


def dry_run(output,cfg,device):
    d=output/'local_dataset'
    if load(d/'verification.json')['status']!='complete':raise RuntimeError('Verify entire dataset first')
    a=load(output/'audit.json');before=sha256(cfg['model'])
    if before!=a['immutable']['checkpoint_sha256']:raise RuntimeError('M0 checkpoint changed')
    from ultralytics.utils.torch_utils import init_seeds
    init_seeds(cfg['seed'],deterministic=True)
    wrapper=YOLO(cfg['model']);model=wrapper.model.float()
    # Checkpoints are loaded with inference-frozen parameters. The real trainer
    # rebuilds the model and enables grads except its fixed DFL projection.
    for name,parameter in model.named_parameters():parameter.requires_grad_('.dfl.' not in name)
    reference=[type(m).__name__ for m in model.model]
    if len(reference)!=25 or model.model[-1].nl!=4 or model.yaml.get('nscr',{}).get('experiment')!='M0':raise RuntimeError('Expected M0 P2 four-head structure')
    # Exercise the actual upstream loader/augmentation/collation and original loss.
    rows=[json.loads(x) for x in (d/'samples.jsonl').read_text().splitlines()]
    # Deterministically include a negative and dense sample; fill total batch8.
    picks=[]
    for kind in ('negative','supplemental','actual'):
        choices=[r for r in rows if r['split']=='train' and r['kind']==kind]
        choices.sort(key=lambda r:-r['objects'])
        if choices:picks.append(choices[0]['image'])
    for r in rows:
        if len(picks)>=8:break
        if r['split']=='train' and r['image'] not in picks:picks.append(r['image'])
    manifest=output/'dry_run_images.txt';manifest.write_text('\n'.join(picks)+'\n',encoding='utf-8')
    options=get_cfg(overrides=cfg);data=yaml.safe_load((d/'dataset.yaml').read_text());data['nc']=10;data['channels']=3
    dataset=build_yolo_dataset(options,str(manifest),8,data,mode='train',rect=False,stride=32)
    loader=build_dataloader(dataset,8,0,False);batch=next(iter(loader))
    target=torch.device('cpu' if device=='cpu' else f'cuda:{device}');model.to(target).train()
    model.args=options;model.nc=10
    batch={k:v.to(target) if isinstance(v,torch.Tensor) else v for k,v in batch.items()};batch['img']=batch['img'].float()/255
    if target.type=='cuda':torch.cuda.reset_peak_memory_stats(target);torch.cuda.synchronize(target)
    start=time.perf_counter()
    with torch.autocast(target.type,enabled=target.type=='cuda'):
        loss,items=model.loss(batch)
    if not torch.isfinite(loss).all():raise RuntimeError('Nonfinite dry-run loss')
    loss.sum().backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Nonfinite dry-run gradient')
    if target.type=='cuda':torch.cuda.synchronize(target)
    elapsed=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated(target)/1024**3 if target.type=='cuda' else None
    model.zero_grad(set_to_none=True)
    if sha256(cfg['model'])!=before:raise RuntimeError('Checkpoint modified')
    save(output/'local_dry_run.json',dict(status='passed',formal_training_started=False,optimizer_step=False,epochs_executed=0,
        finite_forward_backward=True,loss=float(loss.detach().sum()),loss_items=items.detach().float().cpu().tolist(),
        batch_size=len(batch['img']),input_shape=list(batch['img'].shape),objects=len(batch['cls']),
        architecture=reference,architecture_identical_to_M0=True,checkpoint_sha256=before,checkpoint_unchanged=True,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        seconds=elapsed,peak_allocated_GiB=peak,device=str(target),dataset_samples=len(rows),
        resource_note='Peak is this one augmented batch with forward/backward, without AdamW states. Full epoch VRAM/runtime unmeasured.',
        dataset_verification_sha256=sha256(d/'verification.json'),config_sha256=sha256(output/'local_train.yaml')))
    print('Dry-run passed; no optimizer update or formal training',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--mode',choices=['config','dry-run','train'],default='dry-run');p.add_argument('--device',default='0')
    p.add_argument('--execute-training',action='store_true',help='Only use after user authorizes formal local training')
    args=p.parse_args();torch.set_num_threads(4);cv2.setNumThreads(2);cfg=configuration(args.output)
    if args.mode=='config':print(json.dumps(cfg,indent=2));return
    if args.mode=='dry-run':dry_run(args.output,cfg,args.device);return
    if not args.execute_training:p.error('Formal training requires explicit --execute-training AFTER user authorization')
    check=load(args.output/'local_dry_run.json')
    if check['status']!='passed' or check['config_sha256']!=sha256(args.output/'local_train.yaml') or check['checkpoint_sha256']!=sha256(cfg['model']):raise RuntimeError('Dry-run/config/M0 identity mismatch')
    if (Path(cfg['project'])/cfg['name']).exists():raise FileExistsError('Training run already exists; choose a fresh study, do not overwrite')
    if load(args.output/'local_dataset/verification.json')['status']!='complete':raise RuntimeError('Dataset verification missing')
    if check['dataset_verification_sha256']!=sha256(args.output/'local_dataset/verification.json'):raise RuntimeError('Dataset certificate changed after dry-run')
    model=YOLO(cfg['model']);arguments=copy.deepcopy(cfg);arguments.pop('model');arguments['device']=args.device
    model.train(**arguments)


if __name__=='__main__':main()
