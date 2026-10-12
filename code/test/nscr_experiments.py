"""Reproducible architectures, local checkpoint transfer, and experiment metadata."""

import hashlib
import json
import platform
from copy import deepcopy
from pathlib import Path

import torch
import yaml

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = {
    "E0": (None, "none", 0.0),
    "E1": ("none", "none", 0.0),
    "E2": ("single", "none", 0.0),
    "E3": ("gated", "none", 0.0),
    "E4": ("gated", "spacing", 0.1),
    "E5": ("gated", "shuffle", 0.1),
    "E6": ("gated", "density", 0.1),
}


def architecture(experiment, nc=10, weight=0.1, tau=2.0, temperature=0.5, small_size=32.0,
                 density_sigma=2.0, density_threshold=2.0, density_temperature=0.5):
    """Produce each ablation from one versioned backbone/neck definition."""
    mode, supervision, default_weight = EXPERIMENTS[experiment]
    source = ROOT / "code/test/configs/yolo11n-nscr.yaml"
    cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    cfg["nc"] = nc
    if experiment == "E0":
        cfg.pop("nscr")
        cfg["head"] = cfg["head"][:-2] + [[[16, 19, 22], 1, "Detect", ["nc"]]]
    else:
        cfg["head"][-2][3][1] = mode
        cfg["nscr"].update(experiment=experiment, supervision=supervision,
                           weight=weight if default_weight else 0.0, tau=tau,
                           temperature=temperature, small_size=small_size, density_sigma=density_sigma,
                           density_threshold=density_threshold, density_temperature=density_temperature)
    return cfg


def write_architecture(experiment, directory, **kwargs):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"yolo11n-{experiment.lower()}.yaml"
    path.write_text(yaml.safe_dump(architecture(experiment, **kwargs), sort_keys=False), encoding="utf-8")
    return path


def transfer_pretrained(model, checkpoint):
    """Transfer unchanged layers and correctly remap old P3/P4/P5 detection branches.

    A naive YOLO.load would lose all detection weights after inserting layer 23.
    Classification output tensors are transferred only if class semantics match.
    """
    source = YOLO(str(checkpoint)).model
    src = source.state_dict()
    dst = model.state_dict()
    four_scales = len(model.stride) == 4
    same_classes = model.names == source.names
    mapped = {}
    for key, value in src.items():
        target = key
        if key.startswith("model.23."):
            parts = key.split(".")
            if parts[2] in {"cv3", "one2one_cv3"} and parts[4] == "2" and not same_classes:
                continue  # exclude class outputs; transfer compatible classification feature tensors
            if four_scales:
                parts[1] = "24"
                if parts[2] in {"cv2", "cv3", "one2one_cv2", "one2one_cv3"}:
                    parts[3] = str(int(parts[3]) + 1)
                target = ".".join(parts)
        if target in dst and dst[target].shape == value.shape:
            mapped[target] = value.float()
    model.load_state_dict(mapped, strict=False)
    return {"checkpoint": str(Path(checkpoint).resolve()), "transferred_tensors": len(mapped),
            "total_tensors": len(dst), "detect_branch_remap": four_scales,
            "classification_transfer": same_classes}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_manifest(path, args, data_path, cfg, transfer):
    """Record configuration, dataset manifests, implementation and local checkpoint identity."""
    dataset = yaml.safe_load(Path(data_path).read_text(encoding="utf-8"))
    data_root = Path(dataset["path"])
    files = [Path(data_path)]
    if getattr(args, "tuning_report", None):
        files.append(Path(args.tuning_report))
    for split in ("train", "val", "test"):
        if split in dataset:
            candidate = data_root / dataset[split]
            if candidate.is_file():
                files.append(candidate)
    labels_digest = {}
    for split in ("train", "val", "test"):
        if split not in dataset:
            continue
        manifest_path = data_root / dataset[split]
        if manifest_path.suffix != ".txt":
            continue
        digest = hashlib.sha256()
        for image in manifest_path.read_text(encoding="utf-8").splitlines():
            label = Path(image).parent.parent / "labels" / f"{Path(image).stem}.txt"
            digest.update(label.name.encode())
            digest.update(label.read_bytes())
        labels_digest[split] = digest.hexdigest()
    sources = [ROOT / p for p in (
        "ultralytics/nn/modules/nscr.py", "ultralytics/utils/nscr_loss.py",
        "code/test/nscr_experiments.py", "code/test/prepare_data.py", "code/test/train_test.py",
        "code/test/evaluate_nscr.py", "ultralytics/nn/tasks.py", "ultralytics/models/yolo/detect/train.py",
        "ultralytics/utils/checks.py")]
    if "dcr" in cfg:
        sources.extend(ROOT / p for p in ("ultralytics/nn/modules/dcr.py", "ultralytics/utils/dcr_loss.py",
                                           "code/test/dcr_experiments.py", "code/test/run_dcr.py"))
    if "candidate" in cfg or cfg.get("nscr", {}).get("experiment") == "M0":
        sources.extend(ROOT / p for p in ("ultralytics/nn/modules/dcr.py", "ultralytics/utils/dcr_loss.py",
            "ultralytics/nn/modules/candidates.py", "ultralytics/utils/candidate_loss.py",
            "code/test/candidate_experiments.py", "code/test/train_candidate.py", "code/test/run_five.py",
            "code/test/evaluate_candidates.py", "code/test/review_candidates.py", "code/test/progress_console.py",
            "code/test/handoff_candidates.py"))
    if "fusion_research" in cfg:
        sources.extend(ROOT / p for p in ("ultralytics/nn/modules/p2_fusion.py",
            "code/test/fusion_experiments.py", "code/test/train_fusion.py", "code/test/run_fusion.py",
            "code/test/diagnose_fusion.py", "code/test/analyze_fusion_results.py", "code/test/visdrone_protocol.py"))
    manifest = {"arguments": vars(args), "architecture": deepcopy(cfg), "weight_transfer": transfer,
                "checkpoint_sha256": sha256(args.weights), "python": platform.python_version(),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "dataset_files": {str(p): sha256(p) for p in files},
                "dataset_labels_sha256": labels_digest,
                "implementation_files": {str(p.relative_to(ROOT)): sha256(p) for p in sources}}
    Path(path).write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
