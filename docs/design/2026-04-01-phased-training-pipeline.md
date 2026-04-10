# 跌倒检测分阶段训练流程实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use @superpowers:subagent-driven-development (recommended) or @superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭建完整的分阶段训练 Pipeline，包含检测器训练、关键点微调、跟踪器调参、分类器训练、端到端集成与阈值搜索，以及对应的训练脚本和验证脚本。

**Architecture:** 基于已有的 inference pipeline，新增 `train/` 目录放置各阶段训练脚本与工具。检测器和关键点采用 YOLOv8 系列在公开数据集上预训练后再做分辨率迁移/领域微调；分类器采用缓存中间特征的方式离线训练；端到端阶段做规则阈值网格搜索。所有训练脚本支持通过 CLI 参数指定数据集路径和输出目录。

**Tech Stack:** Python 3.10+, PyTorch 2.x, Ultralytics YOLOv8, OpenCV, NumPy, pytest, ONNX Runtime, PyYAML

---

## 文件结构总览

```
fall_detection/
├── src/fall_detection/          # 已有的 inference 模块
├── scripts/
│   ├── benchmark_speed.py
│   ├── run_pipeline_demo.py
│   ├── train_detector.py        # 阶段1：检测器训练
│   ├── train_pose.py            # 阶段2：关键点微调
│   ├── tune_tracker.py          # 阶段3：跟踪器调参
│   ├── train_classifier.py      # 阶段4：分类器训练
│   ├── extract_features.py      # 阶段4辅助：提取中间特征
│   └── evaluate_pipeline.py     # 阶段5：端到端评估与阈值搜索
├── train/                       # 训练输出目录（.gitignore）
│   ├── detector/
│   ├── pose/
│   ├── classifier/
│   └── cache/
├── tests/
│   └── test_training_scripts.py # 训练脚本接口级单元测试
├── configs/
│   └── default.yaml             # 阈值配置
└── requirements.txt
```

---

## Task 1: 训练输出目录结构与配置扩展

**Files:**
- Create: `.gitignore`（若不存在则新增，忽略 `train/`、`*.pt`、`*.onnx`、`__pycache__`）
- Modify: `configs/default.yaml`

- [ ] **Step 1: 更新 configs/default.yaml 增加训练相关字段**

在 `configs/default.yaml` 末尾追加：

```yaml
training:
  detector:
    data_yaml: "data/fall_detection.yaml"
    epochs: 50
    imgsz: 832
    batch: 16
    output_dir: "train/detector"
  pose:
    data_yaml: "data/fall_pose.yaml"
    epochs: 100
    imgsz: 128
    batch: 64
    output_dir: "train/pose"
  classifier:
    feature_cache: "train/cache"
    epochs: 100
    batch_size: 32
    lr: 0.001
    output_dir: "train/classifier"
  tracker:
    video_dir: "data/videos"
    gt_dir: "data/tracker_gt"
    output_dir: "train/tracker"
  pipeline:
    val_video_dir: "data/videos"
    val_gt_dir: "data/event_gt"
    output_dir: "train/eval"
```

- [ ] **Step 2: 新增 .gitignore**

```gitignore
# Training outputs
train/
*.pt
*.onnx
*.engine
*.mlmodel

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
.pytest_cache/

# IDE
.vscode/
.idea/
```

- [ ] **Step 3: Commit**

```bash
git add configs/default.yaml .gitignore
git commit -m "chore: add training config fields and .gitignore"
```

---

## Task 2: 检测器训练脚本 (train_detector.py)

**Files:**
- Create: `scripts/train_detector.py`
- Create: `tests/test_training_scripts.py`

- [ ] **Step 1: 写失败测试**

```python
import subprocess
import sys

def test_train_detector_help():
    result = subprocess.run([sys.executable, "scripts/train_detector.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--data" in result.stdout
```

Run: `PYTHONPATH=src pytest tests/test_training_scripts.py::test_train_detector_help -v`
Expected: FAIL (train_detector.py not found)

- [ ] **Step 2: 实现 train_detector.py**

脚本职责：
1. 解析 CLI 参数：`--data`, `--epochs`, `--imgsz`, `--batch`, `--model`, `--project`, `--name`
2. 加载预训练 `YOLOv8n.pt`（默认）
3. 调用 `model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, project=args.project, name=args.name)`
4. 将最佳权重复制/软链到 `output_dir/best.pt`

代码骨架：

```python
import argparse
import os
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/fall_detection.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=832)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--project", default="train/detector")
    parser.add_argument("--name", default="exp")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )
    best_path = os.path.join(args.project, args.name, "weights", "best.pt")
    out_path = os.path.join(args.project, args.name, "best.pt")
    if os.path.exists(best_path):
        os.link(best_path, out_path)
        print(f"Best weights saved to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 运行测试通过**

Run: `PYTHONPATH=src pytest tests/test_training_scripts.py::test_train_detector_help -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/train_detector.py tests/test_training_scripts.py
git commit -m "feat: add detector training script"
```

---

## Task 3: 关键点训练/微调脚本 (train_pose.py)

**Files:**
- Create: `scripts/train_pose.py`
- Modify: `tests/test_training_scripts.py`

- [ ] **Step 1: 写失败测试**

```python
def test_train_pose_help():
    result = subprocess.run([sys.executable, "scripts/train_pose.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--data" in result.stdout
```

Run: `PYTHONPATH=src pytest tests/test_training_scripts.py::test_train_pose_help -v`
Expected: FAIL

- [ ] **Step 2: 实现 train_pose.py**

与检测器类似，但加载 `yolov8n-pose.pt`，默认 `imgsz=128`：

```python
import argparse
import os
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/fall_pose.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--model", default="yolov8n-pose.pt")
    parser.add_argument("--project", default="train/pose")
    parser.add_argument("--name", default="exp")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )
    best_path = os.path.join(args.project, args.name, "weights", "best.pt")
    out_path = os.path.join(args.project, args.name, "best.pt")
    if os.path.exists(best_path):
        os.link(best_path, out_path)
        print(f"Best weights saved to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 运行测试通过**

Run: `PYTHONPATH=src pytest tests/test_training_scripts.py::test_train_pose_help -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/train_pose.py tests/test_training_scripts.py
git commit -m "feat: add pose training/fine-tune script"
```

---

## Task 4: 跟踪器调参脚本 (tune_tracker.py)

**Files:**
- Create: `scripts/tune_tracker.py`
- Modify: `tests/test_training_scripts.py`

- [ ] **Step 1: 写失败测试**

```python
def test_tune_tracker_help():
    result = subprocess.run([sys.executable, "scripts/tune_tracker.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--video-dir" in result.stdout
```

Run: `PYTHONPATH=src pytest tests/test_training_scripts.py::test_tune_tracker_help -v`
Expected: FAIL

- [ ] **Step 2: 实现 tune_tracker.py**

提供一个简化的网格搜索，评估 tracker 在当前视频上的稳定性（ID persistence 得分）。

脚本职责：
1. 遍历 `video_dir` 下所有视频
2. 对每组 `(track_thresh, match_thresh, max_age, min_hits)` 运行 pipeline 的 tracker 部分
3. 输出各参数组合下的平均 track 持续时间（survival frames）和 ID switch 次数到 CSV
4. 保存最佳参数到 JSON

**实现细节**：
- 用固定 mock detector（每帧返回一个框）配合实际 video 跑 tracker，统计 ID persistence
- 为简化脚本体积，使用 `itertools.product` 生成参数组合

```python
import argparse
import json
import itertools
import os
from collections import defaultdict
import cv2
import numpy as np

import sys
sys.path.insert(0, "src")
from fall_detection.tracker import ByteTrackLite, Detection


def evaluate_tracker(video_path, detector_fn, cfg):
    cap = cv2.VideoCapture(video_path)
    tracker = ByteTrackLite(
        track_thresh=cfg["track_thresh"],
        match_thresh=cfg["match_thresh"],
        max_age=cfg["max_age"],
        min_hits=cfg["min_hits"],
    )
    id_history = defaultdict(int)
    switches = 0
    prev_active_ids = set()
    frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        dets = detector_fn(frame)
        detections = [Detection(d["bbox"], d["conf"]) for d in dets]
        active = tracker.update(detections)
        active_ids = {t.track_id for t in active}
        for tid in active_ids:
            id_history[tid] += 1
        # 简单 heuristic：如果当前活跃 IDs 集合与上一帧无交集，视为一次切换
        if prev_active_ids and not prev_active_ids.intersection(active_ids):
            switches += 1
        prev_active_ids = active_ids
    cap.release()

    avg_life = sum(id_history.values()) / max(1, len(id_history))
    return {
        "frames": frames,
        "avg_life": avg_life,
        "id_switches": switches,
        "score": avg_life - 10 * switches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", default="data/videos")
    parser.add_argument("--output", default="train/tracker/tune_result.json")
    args = parser.parse_args()

    # mock detector：固定框（用于 tracker 参数调优演示）
    h, w = 480, 640
    mock_dets = [{"bbox": [w * 0.3, h * 0.2, w * 0.7, h * 0.9], "conf": 0.9}]
    detector_fn = lambda frame: mock_dets

    grid = {
        "track_thresh": [0.4, 0.5, 0.6],
        "match_thresh": [0.7, 0.8, 0.9],
        "max_age": [20, 30, 40],
        "min_hits": [2, 3],
    }
    keys = list(grid.keys())
    best_score = -1e9
    best_cfg = None
    results = []
    video_files = [
        os.path.join(args.video_dir, f)
        for f in os.listdir(args.video_dir)
        if f.endswith((".mp4", ".avi"))
    ] if os.path.isdir(args.video_dir) else []

    if not video_files:
        # 无视频时生成一段 50 帧空白测试序列
        video_files = [None]

    for values in itertools.product(*[grid[k] for k in keys]):
        cfg = dict(zip(keys, values))
        scores = []
        for vf in video_files:
            if vf is None:
                # 空白序列 mock
                cap_info = {"frames": 50}
                tracker = ByteTrackLite(**cfg)
                id_history = defaultdict(int)
                for _ in range(cap_info["frames"]):
                    dets = [Detection(mock_dets[0]["bbox"], mock_dets[0]["conf"])]
                    active = tracker.update(dets)
                    for t in active:
                        id_history[t.track_id] += 1
                avg_life = sum(id_history.values()) / max(1, len(id_history))
                scores.append({"avg_life": avg_life, "id_switches": 0, "score": avg_life})
            else:
                scores.append(evaluate_tracker(vf, detector_fn, cfg))
        avg_score = sum(s["score"] for s in scores) / len(scores)
        cfg_score = {**cfg, "avg_score": avg_score}
        results.append(cfg_score)
        if avg_score > best_score:
            best_score = avg_score
            best_cfg = cfg
        print(cfg_score)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"best": best_cfg, "results": results}, f, indent=2)
    print(f"Best config: {best_cfg} saved to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 运行测试通过**

Run: `PYTHONPATH=src pytest tests/test_training_scripts.py::test_tune_tracker_help -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/tune_tracker.py tests/test_training_scripts.py
git commit -m "feat: add tracker parameter tuning script"
```

---

## Task 5: 分类器中间特征提取脚本 (extract_features.py)

**Files:**
- Create: `scripts/extract_features.py`
- Modify: `tests/test_training_scripts.py`

- [ ] **Step 1: 写失败测试**

```python
def test_extract_features_help():
    result = subprocess.run([sys.executable, "scripts/extract_features.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--video-dir" in result.stdout
```

Run: `PYTHONPATH=src pytest tests/test_training_scripts.py::test_extract_features_help -v`
Expected: FAIL

- [ ] **Step 2: 实现 extract_features.py**

脚本职责：遍历训练视频，用已有 pipeline（检测+跟踪+关键点）提取每帧的 ROI 图、关键点、运动特征，并保存为 `.npz`。

数据结构：每条保存 `roi(uint8, 3×96×96), kpts(float32, 17×3), motion(float32, 8), label(int)`。
标签来源：假设视频文件名或同级目录下有一个 `labels.json`：

```json
{"fall_001.mp4": {"label": 1, "frames": [120, 180]}, "normal_002.mp4": {"label": 0}}
```

- `label=1` 的 clip：在标注时间段内的所有帧标记为 1
- `label=0` 的 clip：全部帧标记为 0

为了控制缓存大小，对高帧率视频进行 **5 fps 降采样**。

实现代码：

```python
import argparse
import json
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, "src")
from fall_detection.pipeline import FallDetectionPipeline


def parse_label(label_path, video_name, frame_idx):
    if not os.path.exists(label_path):
        return 0
    with open(label_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    info = labels.get(video_name, {})
    label = info.get("label", 0)
    segments = info.get("frames", [])
    if len(segments) == 2 and isinstance(segments[0], int):
        start, end = segments
        return label if start <= frame_idx <= end else 0
    return label


def extract(video_path, pipeline, label, out_dir, sample_fps=5):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    skip = max(1, int(round(video_fps / sample_fps)))
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip != 0:
            frame_idx += 1
            continue

        results = pipeline.process_frame(frame)
        for track in results.get("tracks", []):
            tid = track.track_id
            kpts = results["track_kpts"].get(tid)
            if kpts is None:
                continue
            scores = results.get("track_scores", {}).get(tid, {})
            motion = pipeline._extract_motion(tid, kpts, track.to_tlbr().tolist(),
                                              {"centers": list(pipeline._track_history[tid])})
            # ROI
            x1, y1, x2, y2 = map(int, track.to_tlbr())
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            roi = cv2.resize(frame[y1:y2, x1:x2], (96, 96))
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

            file_name = f"{base_name}_f{frame_idx}_t{tid}.npz"
            np.savez(
                os.path.join(out_dir, file_name),
                roi=roi,
                kpts=kpts.astype(np.float32),
                motion=motion.astype(np.float32),
                label=np.int64(label),
            )
            saved += 1
        frame_idx += 1
    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", default="data/videos")
    parser.add_argument("--label-file", default="data/labels.json")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out-dir", default="train/cache")
    parser.add_argument("--sample-fps", type=int, default=5)
    args = parser.parse_args()

    pipeline = FallDetectionPipeline(args.config)
    os.makedirs(args.out_dir, exist_ok=True)

    video_files = []
    if os.path.isdir(args.video_dir):
        video_files = [
            os.path.join(args.video_dir, f)
            for f in os.listdir(args.video_dir)
            if f.endswith((".mp4", ".avi", ".mov"))
        ]

    total_saved = 0
    for vpath in video_files:
        name = os.path.basename(vpath)
        # 若 labels.json 不存在，默认 label=0
        label = 0
        if os.path.exists(args.label_file):
            label = parse_label(args.label_file, name, 0)
        n = extract(vpath, pipeline, label, args.out_dir, args.sample_fps)
        total_saved += n
        print(f"Processed {name}: saved {n} samples")

    print(f"Total saved: {total_saved}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 运行测试通过**

Run: `PYTHONPATH=src pytest tests/test_training_scripts.py::test_extract_features_help -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/extract_features.py tests/test_training_scripts.py
git commit -m "feat: add classifier feature extraction script"
```

---

## Task 6: 分类器训练脚本 (train_classifier.py)

**Files:**
- Create: `scripts/train_classifier.py`
- Modify: `tests/test_training_scripts.py`

- [ ] **Step 1: 写失败测试**

```python
def test_train_classifier_help():
    result = subprocess.run([sys.executable, "scripts/train_classifier.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--cache-dir" in result.stdout
```

Run: `PYTHONPATH=src pytest tests/test_training_scripts.py::test_train_classifier_help -v`
Expected: FAIL

- [ ] **Step 2: 实现 train_classifier.py**

脚本职责：从 `train/cache` 加载 `.npz`，划分 train/val，训练 `FallClassifier`，保存最佳 `state_dict` 到 `train/classifier/best.pt`。

```python
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, "src")
from fall_detection.classifier import FallClassifier


class FeatureDataset(Dataset):
    def __init__(self, cache_dir):
        self.files = glob.glob(os.path.join(cache_dir, "*.npz"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        roi = torch.from_numpy(data["roi"]).permute(2, 0, 1).float() / 255.0
        kpts = torch.from_numpy(data["kpts"]).float()
        motion = torch.from_numpy(data["motion"]).float()
        label = torch.tensor(float(data["label"]), dtype=torch.float32)
        return roi, kpts, motion, label


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for roi, kpts, motion, label in loader:
        roi, kpts, motion, label = roi.to(device), kpts.to(device), motion.to(device), label.to(device)
        optimizer.zero_grad()
        out = model(roi, kpts, motion).squeeze()
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for roi, kpts, motion, label in loader:
            roi, kpts, motion, label = roi.to(device), kpts.to(device), motion.to(device), label.to(device)
            out = model(roi, kpts, motion).squeeze()
            loss = criterion(out, label)
            total_loss += loss.item()
            pred = (out >= 0.5).float()
            correct += (pred == label).sum().item()
            total += label.size(0)
    return total_loss / len(loader), correct / max(1, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="train/cache")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--output-dir", default="train/classifier")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FeatureDataset(args.cache_dir)
    n_val = int(len(dataset) * args.val_ratio)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = FallClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        t_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_acc = eval_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs}  train_loss={t_loss:.4f}  val_loss={v_loss:.4f}  val_acc={v_acc:.4f}")
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best.pt"))

    print(f"Training done. Best val_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 运行测试通过**

Run: `PYTHONPATH=src pytest tests/test_training_scripts.py::test_train_classifier_help -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/train_classifier.py tests/test_training_scripts.py
git commit -m "feat: add classifier training script"
```

---

## Task 7: 端到端评估与阈值搜索脚本 (evaluate_pipeline.py)

**Files:**
- Create: `scripts/evaluate_pipeline.py`
- Modify: `tests/test_training_scripts.py`

- [ ] **Step 1: 写失败测试**

```python
def test_evaluate_pipeline_help():
    result = subprocess.run([sys.executable, "scripts/evaluate_pipeline.py", "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--video-dir" in result.stdout
```

Run: `PYTHONPATH=src pytest tests/test_training_scripts.py::test_evaluate_pipeline_help -v`
Expected: FAIL

- [ ] **Step 2: 实现 evaluate_pipeline.py**

脚本职责：
1. 加载已有 pipeline
2. 对 `video_dir` 下每个视频跑完整链路（可 mock 检测器加速）
3. 统计事件级 TP/FP/FN（单视频只要有任何 `is_falling=True` 且落在 GT 时间窗口内，即算 TP）
4. 对规则阈值 `T_trigger` 和融合阈值 `T_alarm` 做简单网格搜索
5. 输出最佳阈值组合与评估指标到 JSON

代码骨架：

```python
import argparse
import json
import os
import sys
import itertools

import cv2
import numpy as np

sys.path.insert(0, "src")
from fall_detection.pipeline import FallDetectionPipeline


def evaluate_video(pipeline, video_path, gt_segments, cfg):
    """gt_segments: List[(start_frame, end_frame)] or []."""
    # 临时覆盖阈值
    pipeline.trigger_thresh = cfg["trigger_thresh"]
    for f in pipeline.fusion.values():
        f.alarm_thresh = cfg["alarm_thresh"]

    cap = cv2.VideoCapture(video_path)
    fall_frames = 0
    total_frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1
        results = pipeline.process_frame(frame)
        if any(results.get("track_falling", {}).values()):
            fall_frames += 1
    cap.release()

    predicted = fall_frames > 0
    actual = len(gt_segments) > 0
    tp = int(predicted and actual)
    fp = int(predicted and not actual)
    fn = int(not predicted and actual)
    return {"tp": tp, "fp": fp, "fn": fn, "fall_frames": fall_frames, "total_frames": total_frames}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", default="data/videos")
    parser.add_argument("--gt-file", default="data/event_gt.json")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default="train/eval/eval_result.json")
    parser.add_argument("--mock-detector", action="store_true", help="Use fixed mock detector for speed")
    args = parser.parse_args()

    pipeline = FallDetectionPipeline(args.config)
    if args.mock_detector:
        pipeline.detector = lambda img, conf_thresh=0.3: [
            {"bbox": [100.0, 100.0, 200.0, 400.0], "conf": 0.9, "class_id": 0}
        ]

    gt_data = {}
    if os.path.exists(args.gt_file):
        with open(args.gt_file, "r", encoding="utf-8") as f:
            gt_data = json.load(f)

    video_files = []
    if os.path.isdir(args.video_dir):
        video_files = sorted([
            os.path.join(args.video_dir, f)
            for f in os.listdir(args.video_dir)
            if f.endswith((".mp4", ".avi", ".mov"))
        ])

    if not video_files:
        # dummy data for test run
        video_files = [None]

    grid = {
        "trigger_thresh": [0.5, 0.6, 0.7],
        "alarm_thresh": [0.6, 0.7, 0.8],
    }

    best_score = -1e9
    best_cfg = None
    all_results = []
    for trigger_thresh, alarm_thresh in itertools.product(*grid.values()):
        cfg = {"trigger_thresh": trigger_thresh, "alarm_thresh": alarm_thresh}
        tps = fps = fns = 0
        for vpath in video_files:
            if vpath is None:
                # dummy skip
                continue
            name = os.path.basename(vpath)
            segments = gt_data.get(name, [])
            res = evaluate_video(pipeline, vpath, segments, cfg)
            tps += res["tp"]
            fps += res["fp"]
            fns += res["fn"]
        precision = tps / max(1, tps + fps)
        recall = tps / max(1, tps + fns)
        f1 = 2 * precision * recall / max(1e-6, precision + recall)
        row = {**cfg, "precision": precision, "recall": recall, "f1": f1, "tp": tps, "fp": fps, "fn": fns}
        all_results.append(row)
        if f1 > best_score:
            best_score = f1
            best_cfg = cfg
        print(row)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"best": best_cfg, "results": all_results}, f, indent=2)
    print(f"Best config saved to {args.output}: {best_cfg}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 运行测试通过**

Run: `PYTHONPATH=src pytest tests/test_training_scripts.py::test_evaluate_pipeline_help -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/evaluate_pipeline.py tests/test_training_scripts.py
git commit -m "feat: add end-to-end evaluation and threshold search script"
```

---

## Task 8: 训练脚本集成测试

**Files:**
- Modify: `tests/test_training_scripts.py`

- [ ] **Step 1: 写集成测试**

```python
def test_feature_dataset_empty():
    import tempfile
    import os
    import sys
    sys.path.insert(0, "src")
    from scripts.train_classifier import FeatureDataset
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = FeatureDataset(tmpdir)
        assert len(ds) == 0
```

- [ ] **Step 2: 运行测试通过**

Run: `PYTHONPATH=src pytest tests/test_training_scripts.py -v`
Expected: PASS（所有 help 测试 + 集成测试通过）

- [ ] **Step 3: Commit**

```bash
git add tests/test_training_scripts.py
git commit -m "test: add training script integration tests"
```

---

## Task 9: 端到端训练链路走通验证

- [ ] **Step 1: 创建 dummy 数据验证 extract + train_classifier**

```bash
mkdir -p data/train_dummy
python -c "
import numpy as np
import os
os.makedirs('train/cache', exist_ok=True)
for i in range(20):
    np.savez(f'train/cache/sample_{i:03d}.npz',
        roi=np.random.randint(0,255,(96,96,3), dtype=np.uint8),
        kpts=np.random.rand(17,3).astype(np.float32),
        motion=np.random.rand(8).astype(np.float32),
        label=np.int64(i % 2))
"
```

- [ ] **Step 2: 跑分类器训练**

```bash
python scripts/train_classifier.py --cache-dir train/cache --epochs 2 --batch-size 4 --output-dir train/classifier
```

Expected: 成功跑完 2 个 epoch，输出 `train/classifier/best.pt`

- [ ] **Step 3: Commit dummy 数据清理说明（不提交 train 目录）**

确保 `train/` 已在 `.gitignore` 中。如果 `.gitignore` 之前未添加，先追加再提交。

```bash
git add .gitignore
git commit -m "docs: verify training pipeline with dummy data"
```

---

## 完成标准

- [ ] 所有 `tests/test_training_scripts.py` 用例通过
- [ ] `scripts/train_detector.py` 可以通过 `--help` 并具备 Ultralytics 训练调用能力
- [ ] `scripts/train_pose.py` 可以通过 `--help` 并具备关键点微调调用能力
- [ ] `scripts/tune_tracker.py` 可以输出 tracker 参数网格搜索结果 JSON
- [ ] `scripts/extract_features.py` 可以解析视频并输出 `.npz` 缓存
- [ ] `scripts/train_classifier.py` 可以从 `.npz` 缓存训练并保存 `best.pt`
- [ ] `scripts/evaluate_pipeline.py` 可以做阈值网格搜索并输出评估 JSON
