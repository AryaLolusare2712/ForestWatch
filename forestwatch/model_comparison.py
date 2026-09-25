"""Spatially held-out CNN and YOLO comparison for the labelled Gorewada pair.

This is deliberately an experiment, not a production accuracy claim: the
project currently supplies one labelled 2024–2025 change period.  Patches are
split spatially so test tiles are not reused during training.
"""
from __future__ import annotations
import json
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
from torch import nn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from .config import settings


def _paths():
    root = settings.project_data_path
    return root / "Gorewada_Temporal_Dataset_2024_2025.npy", root / "Gorewada_HighConfidence_Change_Label_2024_2025.npy"


def _change_rgb(before, after):
    delta = after - before
    def norm(a):
        lo, hi = np.nanpercentile(a, [2, 98]); return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.dstack([norm(before), norm(after), norm(np.abs(delta))])


def _tiles(size=32):
    cube_path, labels_path = _paths()
    if not cube_path.exists() or not labels_path.exists():
        raise ValueError("The aligned 2024–2025 temporal cube and high-confidence label are required.")
    cube = np.load(cube_path, mmap_mode="r")
    labels = np.asarray(np.load(labels_path, mmap_mode="r"))
    before, after = np.asarray(cube[..., 2], dtype=np.float32), np.asarray(cube[..., 5], dtype=np.float32)
    image = (_change_rgb(before, after) * 255).astype(np.uint8)
    valid, changed = labels != 255, labels == 1
    rows = []
    for top in range(0, image.shape[0] - size + 1, size):
        for left in range(0, image.shape[1] - size + 1, size):
            valid_tile, change_tile = valid[top:top+size, left:left+size], changed[top:top+size, left:left+size]
            if valid_tile.mean() < .40: continue
            rows.append((top, left, image[top:top+size, left:left+size], change_tile & valid_tile, int((change_tile & valid_tile).mean() >= .005)))
    if len(rows) < 8: raise ValueError("Not enough valid spatial tiles for a held-out comparison.")
    return rows


class TinyChangeCNN(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 2))
    def forward(self, value): return self.net(value)


def _cnn_metrics(rows, epochs):
    torch.set_num_threads(1)
    # Checkerboard-style spatial holdout prevents adjacent tiles from simply
    # being copies of training context, while retaining rare change tiles in
    # both sets for evaluable recall.
    test = [row for row in rows if ((row[0] // 32) + (row[1] // 32)) % 5 == 0]
    test_coordinates = {(row[0], row[1]) for row in test}
    train = [row for row in rows if (row[0], row[1]) not in test_coordinates]
    x = torch.tensor(np.stack([row[2].transpose(2, 0, 1) / 255 for row in train]), dtype=torch.float32)
    y = torch.tensor([row[4] for row in train], dtype=torch.long)
    model = TinyChangeCNN(); optimiser = torch.optim.Adam(model.parameters(), lr=.001)
    class_counts = torch.bincount(y, minlength=2).float().clamp_min(1)
    # Change patches are rare. Weighted loss prevents a trivial all-stable CNN
    # from looking accurate while detecting no changes at all.
    loss_fn = nn.CrossEntropyLoss(weight=class_counts.sum() / (2 * class_counts))
    model.train()
    for _ in range(max(1, int(epochs))):
        optimiser.zero_grad(); loss_fn(model(x), y).backward(); optimiser.step()
    tx = torch.tensor(np.stack([row[2].transpose(2, 0, 1) / 255 for row in test]), dtype=torch.float32)
    truth = np.array([row[4] for row in test]); model.eval()
    with torch.no_grad(): predicted = model(tx).argmax(1).numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted, average="binary", zero_division=0)
    return {"model":"Tiny CNN", "evaluation":"held-out spatial tiles", "accuracy":float(accuracy_score(truth,predicted)), "precision":float(precision), "recall":float(recall), "f1":float(f1), "test_tiles":len(test), "training_change_tiles":int(y.sum()), "test_change_tiles":int(truth.sum())}


def _write_yolo_dataset(rows, root: Path):
    # This folder is generated from the source .npy files.  Rebuild it on every
    # run so images/labels from an earlier experiment can never leak into the
    # current validation result.
    if root.exists():
        shutil.rmtree(root)
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True); (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    augmented_positive_tiles = 0
    for index, (_, _, image, mask, _) in enumerate(rows):
        top, left = rows[index][0], rows[index][1]
        split = "val" if ((top // 32) + (left // 32)) % 5 == 0 else "train"; stem = f"tile_{index:03d}"
        cv2.imwrite(str(root / "images" / split / f"{stem}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # High-confidence change labels may be only a few pixels wide on the
        # 10 m grid. Expand them by two pixels to give an object detector a
        # learnable local context, without changing which pixels are labelled
        # as the original change source.
        context_mask = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1)
        count, _, stats, _ = cv2.connectedComponentsWithStats(context_mask, connectivity=8)
        boxes = []
        height, width = mask.shape
        for x, y, w, h, area in stats[1:]:
            if area >= 1: boxes.append(((x + w / 2) / width, (y + h / 2) / height, w / width, h / height))
        def write_sample(name, sample, sample_boxes):
            cv2.imwrite(str(root / "images" / split / f"{name}.png"), cv2.cvtColor(sample, cv2.COLOR_RGB2BGR))
            text = "\n".join(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for cx, cy, bw, bh in sample_boxes)
            (root / "labels" / split / f"{name}.txt").write_text(text, encoding="utf-8")
        # Original sample, then positive-only geometric augmentation in the
        # training split. Validation remains completely untouched.
        write_sample(stem, image, boxes)
        if split == "train" and boxes:
            write_sample(f"{stem}_flip_h", cv2.flip(image, 1), [(1 - cx, cy, bw, bh) for cx, cy, bw, bh in boxes])
            write_sample(f"{stem}_flip_v", cv2.flip(image, 0), [(cx, 1 - cy, bw, bh) for cx, cy, bw, bh in boxes])
            write_sample(f"{stem}_rot_180", cv2.rotate(image, cv2.ROTATE_180), [(1 - cx, 1 - cy, bw, bh) for cx, cy, bw, bh in boxes])
            augmented_positive_tiles += 3
    yaml = root / "data.yaml"
    yaml.write_text(f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: vegetation_change\n", encoding="utf-8")
    return yaml, augmented_positive_tiles


def _yolo_metrics(rows, epochs):
    config_dir = settings.output_path / "models" / "ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    try:
        from ultralytics import YOLO
    except Exception as exc:
        return {"model":"YOLOv8n", "status":"Unavailable", "detail":f"YOLO could not start: {exc}"}
    root = settings.output_path / "models" / "yolo_change_dataset"; yaml, augmentations = _write_yolo_dataset(rows, root)
    try:
        model = YOLO("yolov8n.pt")
        # The 32px source tiles are upscaled to give YOLO enough feature-map
        # resolution for compact changes.  The pretrained backbone, AdamW and
        # positive-only copies make this a meaningful fine-tuning experiment,
        # rather than a detector trained from scratch on five rare positives.
        results = model.train(data=str(yaml), epochs=max(1, int(epochs)), imgsz=128,
                              batch=8, device="cpu", workers=0, optimizer="AdamW",
                              lr0=0.002, cos_lr=True, fliplr=0.5, flipud=0.5,
                              mosaic=0.0, project=str(settings.output_path / "models" / "yolo_runs"),
                              name="gorewada", exist_ok=True, verbose=False)
        # Validate the actual best checkpoint.  Calling val() on the original
        # model object can otherwise evaluate its pre-training weights.
        trained_model = YOLO(str(Path(results.save_dir) / "weights" / "best.pt"))
        metrics = trained_model.val(data=str(yaml), imgsz=128, device="cpu", workers=0, verbose=False)
        tile_size = rows[0][2].shape[0]
        validation_rows = [row for row in rows if ((row[0] // tile_size) + (row[1] // tile_size)) % 5 == 0]
        validation_coordinates = {(row[0], row[1]) for row in validation_rows}
        training_rows = [row for row in rows if (row[0], row[1]) not in validation_coordinates]
        return {"model":"YOLOv8n", "evaluation":"held-out spatial tiles", "mAP50":float(metrics.box.map50), "mAP50_95":float(metrics.box.map), "precision":float(metrics.box.mp), "recall":float(metrics.box.mr), "status":"Completed", "training_change_tiles":int(sum(row[4] for row in training_rows)), "test_change_tiles":int(sum(row[4] for row in validation_rows)), "positive_tile_augmentations":augmentations, "run_directory":str(results.save_dir)}
    except Exception as exc:
        return {"model":"YOLOv8n", "status":"Not completed", "detail":str(exc)}


def compare_cnn_yolo(epochs=5):
    settings.ensure_output_dirs(); rows = _tiles()
    cnn = _cnn_metrics(rows, epochs); yolo = _yolo_metrics(rows, epochs)
    table = pd.DataFrame([cnn, yolo]).fillna("—")
    change_tiles = int(sum(row[4] for row in rows))
    note = (f"### Experimental comparison\nThe supplied label set contains only {change_tiles} changed tiles across one 2024–2025 period. "
            "Zero precision, recall, or mAP means the model did not detect a labelled held-out change tile; it does not mean there was no forest change. Add independent labelled dates before operational use.")
    output = settings.output_path / "metrics" / "cnn_yolo_comparison.json"; output.write_text(json.dumps(table.to_dict("records"), indent=2), encoding="utf-8")
    return table, note
