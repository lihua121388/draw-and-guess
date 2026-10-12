"""Profile exact layer shapes and costs with paired, interleaved GPU forward timings; no training."""
import argparse
import json
import numpy as np
from train_test import ROOT
import torch
from fusion_experiments import EXPERIMENTS, architecture, write_architecture
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops
from nscr_experiments import transfer_pretrained


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=str, default=str(ROOT / "runs/fusion/profile.json"))
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    rows, models = {}, {}
    for experiment in EXPERIMENTS:
        torch.manual_seed(42)
        path = write_architecture(experiment, ROOT / "code/test/configs/generated")
        model = DetectionModel(architecture(experiment), verbose=False).eval()
        transfer = transfer_pretrained(model, ROOT / "model/yolo11n.pt")
        shape, handles = {}, []
        for idx in (2, 16, 19, 22, 23):
            def capture(module, inputs, output, idx=idx):
                shape[str(idx)] = list(output.shape)
            handles.append(model.model[idx].register_forward_hook(capture))
        with torch.inference_mode():
            model(torch.zeros(1, 3, 640, 640))
        for handle in handles:
            handle.remove()
        rows[experiment] = dict(config=str(path), params=sum(p.numel() for p in model.parameters()),
            GFLOPs=get_flops(model, 640), shapes=shape, weight_transfer=transfer)
        models[experiment] = model.fuse(verbose=False).to(device).eval()
    tensor = torch.zeros(1, 3, 640, 640, device=device)
    timings = {name: [] for name in models}
    with torch.inference_mode():
        reference = models["F0"](tensor)[0]
        for name, model in models.items():
            torch.testing.assert_close(model(tensor)[0], reference, rtol=1e-5, atol=1e-4)
            for _ in range(10):
                model(tensor)
        for repeat in range(30):
            names = list(models)
            if repeat % 2:
                names.reverse()
            for name in names:
                if device.type == "cuda":
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record(); models[name](tensor); end.record(); end.synchronize()
                    timings[name].append(start.elapsed_time(end))
                else:
                    import time
                    start = time.perf_counter(); models[name](tensor)
                    timings[name].append(1000*(time.perf_counter()-start))
    for name, values in timings.items():
        rows[name].update(mean_ms=float(np.mean(values)), p50_ms=float(np.median(values)),
                          delta_params=rows[name]["params"]-rows["F0"]["params"],
                          delta_GFLOPs=rows[name]["GFLOPs"]-rows["F0"]["GFLOPs"])
    report = dict(models=rows, protocol="Untrained new fusion, same local pretraining, seed42, "
        "FP32 batch1 640-square zero RGB, fused forward only; 10 warmup per model then 30 interleaved repeats, "
        "GPU event synchronized. No preprocessing/NMS/IO. This is a cost and initialization check, not detection accuracy.",
        device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None)
    from pathlib import Path
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
