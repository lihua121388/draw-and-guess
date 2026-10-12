"""DCR candidate and matched controls; original NSCR experiments remain available."""
from pathlib import Path
import yaml
from nscr_experiments import architecture as nscr_architecture

EXPERIMENTS = ("E0", "E1", "E3", "E4", "D1", "D2", "D3", "D4")
SUPERVISION = {"D1": "none", "D2": "isotropic", "D3": "directional", "D4": "shuffle"}


def architecture(experiment, nc=10, weight=.1, tau=2., temperature=.5, small_size=32., cap=6., **kwargs):
    if experiment not in SUPERVISION:
        return nscr_architecture(experiment, nc, weight, tau, temperature, small_size, **kwargs)
    cfg = nscr_architecture("E1", nc=nc)
    cfg.pop("nscr")
    cfg["head"][-2] = [[2, 16], 1, "DCRFusion", [256, .1]]
    cfg["dcr"] = dict(experiment=experiment, supervision=SUPERVISION[experiment],
                      weight=0. if experiment == "D1" else weight, tau=tau,
                      temperature=temperature, small_size=small_size, cap=cap)
    return cfg


def write_architecture(experiment, directory, **kwargs):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"yolo11n-{experiment.lower()}.yaml"
    path.write_text(yaml.safe_dump(architecture(experiment, **kwargs), sort_keys=False), encoding="utf-8")
    return path
