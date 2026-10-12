"""Two real validation images exercise evaluation IO; this is not an accuracy experiment."""
import json
from pathlib import Path
from types import SimpleNamespace
import yaml
import torch
from evaluate_dcr import direction_diagnostics, ROOT
from evaluate_nscr import evaluate
from dcr_experiments import architecture
from nscr_experiments import transfer_pretrained
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG_DICT


def main():
    torch.set_num_threads(2)
    base = ROOT / "runs/dcr/evaluation_smoke"
    base.mkdir(parents=True, exist_ok=True)
    full = ROOT / "datasets/nscr_visdrone/dataset.yaml"
    cfg = yaml.safe_load(full.read_text(encoding="utf-8"))
    images = (Path(cfg["path"]) / cfg["val"]).read_text().splitlines()[:2]
    manifest = base / "val_two.txt"
    manifest.write_text("\n".join(images) + "\n", encoding="utf-8")
    cfg["val"] = manifest.as_posix()
    data = base / "data_two.yaml"
    data.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    model = DetectionModel(architecture("D3"), verbose=False)
    model.names = cfg["names"]
    transfer_pretrained(model, ROOT / "model/yolo11n.pt")
    model.eval()
    model.args = dict(DEFAULT_CFG_DICT)
    checkpoint = base / "untrained_dcr.pt"
    torch.save({"model": model, "train_args": dict(DEFAULT_CFG_DICT)}, checkpoint)
    args = SimpleNamespace(split="val", imgsz=640, device="cpu", batch=1, workers=0,
                           max_det=500, no_plots=True, tau=2., small_size=32., conf=.25)
    report = evaluate(checkpoint, data, base / "evaluation", args)
    report["directions"] = direction_diagnostics(checkpoint, data, args)
    assert report["images"] == 2
    assert report["directions"]["targets"] > 0
    assert report["directions"]["mean_KL_target_to_prediction"] >= 0
    report["purpose"] = "Untrained DCR IO/coordinate smoke test, not evidence of accuracy or measured GPU performance"
    (base / "smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Evaluation pipeline smoke passed")


if __name__ == "__main__":
    main()
