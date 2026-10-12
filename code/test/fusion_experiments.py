"""Three mature mechanisms, plus two necessary simple controls, based on the exact M0."""
from pathlib import Path
import yaml
from candidate_experiments import architecture as candidate_architecture

EXPERIMENTS = {
    "F0": None,
    "F1": "semantic",
    "F2_GLOBAL": "global_weight",
    "F2": "spatial",
    "F3_PLAIN": "detail_plain",
    "F3": "detail_gated",
}


def architecture(experiment, nc=10, **kwargs):
    cfg = candidate_architecture("M0", nc=nc)
    # Keep the zero fourth loss term and original training infrastructure identical to M0.
    cfg["nscr"].update(experiment=experiment, weight=0., supervision="none")
    if experiment != "F0":
        cfg["head"][-2] = [[2, 16], 1, "P2Fusion", [256, EXPERIMENTS[experiment], .1]]
    cfg["fusion_research"] = dict(experiment=experiment, mechanism=EXPERIMENTS[experiment],
                                  novelty_claim=False, auxiliary_supervision=False)
    return cfg


def write_architecture(experiment, directory, **kwargs):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"yolo11n-{experiment.lower()}.yaml"
    path.write_text(yaml.safe_dump(architecture(experiment, **kwargs), sort_keys=False), encoding="utf-8")
    return path
