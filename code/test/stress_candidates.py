"""Batch-8 CUDA AMP forward/backward with 512 valid GT boxes in every image."""
import json
import sys
import gc
from pathlib import Path
from types import SimpleNamespace
from train_test import ROOT
import torch
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT
from candidate_experiments import architecture
from nscr_experiments import transfer_pretrained


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the requested batch stress test")
    torch.set_num_threads(4)
    output = ROOT / "runs/candidates/stress_b8.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for experiment in ("M1", "M2", "M3", "M4", "M5", "M0"):
        torch.manual_seed(42)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = DetectionModel(architecture(experiment), verbose=False)
        transfer_pretrained(model, ROOT / "model/yolo11n.pt")
        model.args = SimpleNamespace(**DEFAULT_CFG_DICT)
        model = model.cuda().train()
        n = 512
        centers = torch.rand(8*n, 2, device="cuda")*.88+.06
        sizes = torch.rand(8*n, 2, device="cuda")*.025+.008
        batch = dict(img=torch.rand(8, 3, 640, 640, device="cuda"),
            bboxes=torch.cat((centers, sizes), 1), cls=torch.randint(0, 10, (8*n, 1), device="cuda").float(),
            batch_idx=torch.arange(8, device="cuda").repeat_interleave(n))
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss, items = model(batch)
        assert torch.isfinite(loss).all(), f"Nonfinite loss: {experiment}"
        loss.sum().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None), experiment
        torch.cuda.synchronize()
        record = dict(experiment=experiment, batch=8, imgsz=640, instances_per_image=n,
            AMP="FP16", loss_items=items.detach().cpu().tolist(),
            peak_allocated_GiB=torch.cuda.max_memory_allocated()/1024**3,
            peak_reserved_GiB=torch.cuda.max_memory_reserved()/1024**3, status="passed")
        records.append(record)
        output.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)
        del model, batch, loss, items, centers, sizes
        gc.collect()
    print("All candidates passed batch=8 AMP stress")


if __name__ == "__main__":
    main()
