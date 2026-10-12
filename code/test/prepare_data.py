"""Prepare local VisDrone or official AI-TOD-v2 COCO splits without random resplitting."""

import argparse
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]


def clip_box(box, width, height):
    """Clip xywh to an image and reject nonfinite, empty or reversed annotations."""
    x, y, w, h = map(float, box)
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        return None
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(width, x + w), min(height, y + h)
    return [x1, y1, x2 - x1, y2 - y1] if x2 > x1 and y2 > y1 else None


def inside_ignore(box, ignored):
    x, y = box[0] + box[2] / 2, box[1] + box[3] / 2
    return any(r[0] <= x <= r[0] + r[2] and r[1] <= y <= r[1] + r[3] for r in ignored)


def link_image(source, destination):
    """Use same-volume hard links, falling back to a copy; never modify source images."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise FileExistsError(f"Existing processed image differs: {destination}")
        return
    try:
        os.link(str(source), str(destination))
    except OSError:
        shutil.copy2(str(source), str(destination))


def label_line(category, box, width, height):
    x, y, w, h = box
    values = [(x + w / 2) / width, (y + h / 2) / height, w / width, h / height]
    return f"{category} " + " ".join(f"{v:.9f}" for v in values)


def write_splits(output, names, splits, train_limit=0, val_limit=0, seed=42):
    """Deterministic subset selection within the official training split only."""
    suffix = f"-subset-{train_limit}-{val_limit}-s{seed}" if train_limit or val_limit else ""
    for split, images in splits.items():
        chosen = sorted(images)
        limit = train_limit if split == "train" else val_limit if split == "val" else 0
        if limit:
            # Hash ranking avoids selecting one contiguous video/city sequence.
            chosen = sorted(chosen, key=lambda p: hashlib.sha256(f"{seed}:{p.name}".encode()).hexdigest())[:limit]
            chosen.sort()
        (output / f"{split}{suffix}.txt").write_text("\n".join(p.resolve().as_posix() for p in chosen) + "\n", encoding="utf-8")
    cfg = {"path": output.resolve().as_posix(), **{k: f"{k}{suffix}.txt" for k in splits}, "names": dict(enumerate(names))}
    path = output / f"dataset{suffix}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def prepare_visdrone(source, output, train_limit=0, val_limit=0, seed=42):
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    splits, report = {}, {"dataset": "VisDrone2019-DET", "source": str(source), "splits": {}}
    for split, raw_name in (("train", "train"), ("val", "val"), ("test", "test-dev")):
        name = f"VisDrone2019-DET-{raw_name}"
        candidates = [source / name, source / name / name, source]  # common nested extraction layouts
        raw = next((p for p in candidates if (p / "images").is_dir() and (p / "annotations").is_dir()), None)
        if raw is None:
            if split == "test":
                continue
            raise FileNotFoundError(f"Missing {name}/images and annotations under {source}")
        images = sorted((raw / "images").glob("*.jpg"))
        if not images:
            raise ValueError(f"No images in {raw}")
        target = output / split
        (target / "labels").mkdir(parents=True, exist_ok=True)
        coco = {"info": {"description": "Supplemental COCO protocol; use original annotations for official VisDrone evaluation"},
                "images": [], "annotations": [], "categories": [{"id": i + 1, "name": n} for i, n in enumerate(NAMES)]}
        stats = {"images": len(images), "valid_boxes": 0, "invalid_boxes": 0, "ignored_boxes": 0,
                 "boxes_in_ignore": 0, "original_annotations": str(raw / "annotations")}
        processed = []
        for image_id, image in enumerate(images, 1):
            with Image.open(image) as img:
                width, height = img.size
            annotation = raw / "annotations" / f"{image.stem}.txt"
            if not annotation.is_file():
                raise FileNotFoundError(annotation)
            rows, ignored = [], []
            for number, line in enumerate(annotation.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.strip().rstrip(",").split(",")
                if len(fields) < 8:
                    raise ValueError(f"Malformed annotation {annotation}:{number}")
                values = [float(x) for x in fields[:8]]
                box = clip_box(values[:4], width, height)
                if box is None or not all(math.isfinite(v) for v in values):
                    stats["invalid_boxes"] += 1
                    continue
                score, category = values[4], int(values[5])
                if score <= 0 or category in (0, 11):
                    ignored.append(box)
                    stats["ignored_boxes"] += 1
                elif 1 <= category <= 10 and values[5] == category:
                    rows.append((category, box))
                else:
                    stats["invalid_boxes"] += 1
            lines = []
            for category, box in rows:
                if inside_ignore(box, ignored):
                    stats["boxes_in_ignore"] += 1
                    continue
                lines.append(label_line(category - 1, box, width, height))
                coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": image_id,
                    "category_id": category, "bbox": box, "area": box[2] * box[3], "iscrowd": 0})
                stats["valid_boxes"] += 1
            # Crowd entries are only a supplemental COCO approximation to official ignored-region handling.
            for box in ignored:
                for category in range(1, 11):
                    coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": image_id,
                        "category_id": category, "bbox": box, "area": box[2] * box[3], "iscrowd": 1})
            destination = target / "images" / image.name
            link_image(image, destination)
            (target / "labels" / f"{image.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            coco["images"].append({"id": image_id, "file_name": image.name, "width": width, "height": height})
            processed.append(destination)
        (output / f"{split}_coco.json").write_text(json.dumps(coco), encoding="utf-8")
        splits[split], report["splits"][split] = processed, stats
        print(f"{split}: {stats}")
    report.update(train_limit=train_limit, val_limit=val_limit, seed=seed,
                  training_ignore_policy="omit ignored boxes and positive boxes whose centers are inside ignored regions; YOLO background loss unchanged")
    (output / "preparation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return write_splits(output, NAMES, splits, train_limit, val_limit, seed)


def prepare_aitod(images_root, annotation_paths, output, train_limit=0, val_limit=0, seed=42):
    """Accept the official v2 annotations and preserve their image membership/category IDs."""
    images_root, output = Path(images_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    names, categories, splits, reports = None, None, {}, {}
    split_image_sets = []
    for split, annotation in annotation_paths.items():
        if not annotation:
            continue
        payload = json.loads(Path(annotation).read_text(encoding="utf-8"))
        cats = sorted(payload["categories"], key=lambda c: c["id"])
        current = [(c["id"], c["name"]) for c in cats]
        if categories is not None and categories != current:
            raise ValueError("AI-TOD-v2 category mapping differs between splits")
        categories, names = current, [c["name"] for c in cats]
        mapping = {c["id"]: i for i, c in enumerate(cats)}
        by_image = {}
        for ann in payload["annotations"]:
            by_image.setdefault(ann["image_id"], []).append(ann)
        target = output / split
        (target / "labels").mkdir(parents=True, exist_ok=True)
        processed, used, membership = [], set(), set()
        for img in payload["images"]:
            relative = Path(img["file_name"])
            candidates = [images_root / relative, images_root / split / relative,
                          images_root / split / "images" / relative]
            source = next((p for p in candidates if p.is_file()), None)
            if source is None:
                raise FileNotFoundError(f"Missing AI-TOD-v2 image {relative} under {images_root}")
            membership.add(str(source.resolve()))
            filename = relative.name
            if filename in used:
                raise ValueError(f"Duplicate image basename: {filename}")
            used.add(filename)
            with Image.open(source) as pil:
                if pil.size != (img["width"], img["height"]):
                    raise ValueError(f"Image/annotation dimensions differ: {source}")
            rows, ignored = [], []
            for ann in by_image.get(img["id"], []):
                box = clip_box(ann["bbox"], img["width"], img["height"])
                if box is None:
                    continue
                if ann.get("iscrowd", 0) or ann.get("ignore", 0):
                    ignored.append(box)
                else:
                    rows.append((mapping[ann["category_id"]], box))
            lines = [label_line(c, b, img["width"], img["height"]) for c, b in rows if not inside_ignore(b, ignored)]
            destination = target / "images" / filename
            link_image(source, destination)
            (target / "labels" / f"{Path(filename).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            img["file_name"] = filename
            processed.append(destination)
        if any(membership & previous for previous in split_image_sets):
            raise ValueError("AI-TOD-v2 splits overlap; supply distinct official train/val/test splits")
        split_image_sets.append(membership)
        (output / f"{split}_coco.json").write_text(json.dumps(payload), encoding="utf-8")
        splits[split] = processed
        reports[split] = {"source_annotations": str(Path(annotation).resolve()), "images": len(processed)}
    if "train" not in splits or "val" not in splits:
        raise ValueError("Both official training and validation v2 annotations are required")
    (output / "preparation.json").write_text(json.dumps({"dataset": "AI-TOD-v2", "splits": reports}, indent=2), encoding="utf-8")
    return write_splits(output, names, splits, train_limit, val_limit, seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("visdrone", "aitod"), default="visdrone")
    parser.add_argument("--source", type=Path, default=ROOT / "datasets/VisDrone2019-DET")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coco-train", type=Path)
    parser.add_argument("--coco-val", type=Path)
    parser.add_argument("--coco-test", type=Path)
    args = parser.parse_args()
    if args.train_limit < 0 or args.val_limit < 0:
        parser.error("Subset limits must be nonnegative")
    output = args.output or ROOT / f"datasets/nscr_{args.dataset}"
    if args.dataset == "visdrone":
        path = prepare_visdrone(args.source, output, args.train_limit, args.val_limit, args.seed)
    else:
        path = prepare_aitod(args.source, {"train": args.coco_train, "val": args.coco_val, "test": args.coco_test},
                             output, args.train_limit, args.val_limit, args.seed)
    print(f"Dataset YAML: {path}")


if __name__ == "__main__":
    main()
