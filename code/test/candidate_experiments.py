"""Independent candidate architectures and matched P2 control."""
from pathlib import Path
import yaml
from nscr_experiments import architecture as nscr_architecture

EXPERIMENTS = {
    "M0": dict(kind=None, name="P2 baseline"),
    "M1": dict(kind="dcr", name="Directional Clearance Routing"),
    "M2": dict(kind="csr", name="Collision-Safe Ring Routing"),
    "M3": dict(kind="brr", name="Boundary-Residual Recovery"),
    "M4": dict(kind="gpr", name="Grid-Phase Reliability Fusion"),
    "M5": dict(kind="pcr", name="Peer-Consensus Rescue"),
}


def architecture(experiment, nc=10, weight=.1, tau=2., temperature=.5, small_size=32., cap=6., **kwargs):
    cfg = nscr_architecture("E1", nc=nc)
    if experiment == "M0":
        cfg["nscr"]["experiment"] = "M0"
        return cfg
    kind = EXPERIMENTS[experiment]["kind"]
    cfg.pop("nscr")
    cfg["head"][-2] = [[2, 16], 1, "CandidateFusion", [256, kind, .1]]
    cfg["candidate"] = dict(experiment=experiment, kind=kind, name=EXPERIMENTS[experiment]["name"],
        weight=weight, tau=tau, temperature=temperature, small_size=small_size, cap=cap)
    return cfg


def write_architecture(experiment, directory, **kwargs):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"yolo11n-{experiment.lower()}.yaml"
    path.write_text(yaml.safe_dump(architecture(experiment, **kwargs), sort_keys=False), encoding="utf-8")
    return path
