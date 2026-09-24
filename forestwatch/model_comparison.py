"""Spatially held-out CNN and YOLO comparison for the labelled Gorewada pair.

This is deliberately an experiment, not a production accuracy claim: the
project currently supplies one labelled 2024–2025 change period.  Patches are
split spatially so test tiles are not reused during training.
"""
from __future__ import annotations
import json
import os
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
    for _ in range(max(1, min(int(epochs), 50))):
        optimiser.zero_grad(); loss_fn(model(x), y).backward(); optimiser.step()
    tx = torch.tensor(np.stack([row[2].transpose(2, 0, 1) / 255 for row in test]), dtype=torch.float32)
    truth = np.array([row[4] for row in test]); model.eval()
    with torch.no_grad(): predicted = model(tx).argmax(1).numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(truth, predicted, average="binary", zero_division=0)
    return {"model":"Tiny CNN", "evaluation":"held-out spatial tiles", "accuracy":float(accuracy_score(truth,predicted)), "precision":float(precision), "recall":float(recall), "f1":float(f1), "test_tiles":len(test)}


def _write_yolo_dataset(rows, root: Path):
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True); (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for index, (_, _, image, mask, _) in enumerate(rows):
        top, left = rows[index][0], rows[index][1]
        split = "val" if ((top // 32) + (left // 32)) % 5 == 0 else "train"; stem = f"tile_{index:03d}"
        cv2.imwrite(str(root / "images" / split / f"{stem}.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        labels = []
        height, width = mask.shape
        for x, y, w, h, area in stats[1:]:
            if area >= 8: labels.append(f"0 {(x+w/2)/width:.6f} {(y+h/2)/height:.6f} {w/width:.6f} {h/height:.6f}")
        (root / "labels" / split / f"{stem}.txt").write_text("\n".join(labels), encoding="utf-8")
    yaml = root / "data.yaml"
    yaml.write_text(f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: vegetation_change\n", encoding="utf-8")
    return yaml


def _yolo_metrics(rows, epochs):
    config_dir = settings.output_path / "models" / "ultralytics"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
    try:
        from ultralytics import YOLO
    except Exception as exc:
        return {"model":"YOLOv8n", "status":"Unavailable", "detail":f"YOLO could not start: {exc}"}
    root = settings.output_path / "models" / "yolo_change_dataset"; yaml = _write_yolo_dataset(rows, root)
    try:
        model = YOLO("yolov8n.pt")
        results = model.train(data=str(yaml), epochs=max(1, min(int(epochs), 50)), imgsz=32, batch=4, device="cpu", workers=0, project=str(settings.output_path / "models" / "yolo_runs"), name="gorewada", exist_ok=True, verbose=False)
        metrics = model.val(data=str(yaml), imgsz=32, device="cpu", workers=0, verbose=False)
        return {"model":"YOLOv8n", "evaluation":"held-out spatial tiles", "mAP50":float(metrics.box.map50), "mAP50_95":float(metrics.box.map), "precision":float(metrics.box.mp), "recall":float(metrics.box.mr), "status":"Completed", "run_directory":str(results.save_dir)}
    except Exception as exc:
        return {"model":"YOLOv8n", "status":"Not completed", "detail":str(exc)}


def compare_cnn_yolo(epochs=5):
    settings.ensure_output_dirs(); rows = _tiles()
    cnn = _cnn_metrics(rows, epochs); yolo = _yolo_metrics(rows, epochs)
    table = pd.DataFrame([cnn, yolo]).fillna("—")
    note = ("### Experimental comparison\nBoth models use the supplied labelled 2024–2025 Gorewada pair and a spatially held-out tile split. "
            "One labelled time period is insufficient to claim general accuracy; add independent labelled dates before operational use.")
    output = settings.output_path / "metrics" / "cnn_yolo_comparison.json"; output.write_text(json.dumps(table.to_dict("records"), indent=2), encoding="utf-8")
    return table, note
