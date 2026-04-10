# 跌倒检测端到端测试 Pipeline 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use @superpowers:subagent-driven-development (recommended) or @superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从零搭建可运行的端到端跌倒检测测试 pipeline，包含检测、跟踪、关键点、规则、分类器、融合决策模块，以及完整的单元测试和集成测试。

**Architecture:** 采用 YOLOv8n（检测）+ YOLOv8n-pose（关键点）作为开源预训练 backbone 快速冷启动，自研 ByteTrack-lite、规则引擎、轻量融合分类器和端到端 pipeline。所有模块配有独立单元测试和可视化验证脚本。

**Tech Stack:** Python 3.10+, PyTorch 2.x, Ultralytics YOLOv8, OpenCV, NumPy, pytest, ONNX Runtime

---

## 文件结构总览

```
fall_detection/
├── configs/
│   └── default.yaml            # 默认超参数和阈值配置
├── src/
│   └── fall_detection/
│       ├── __init__.py
│       ├── detector.py         # YOLOv8n 检测器封装
│       ├── pose_estimator.py   # YOLOv8n-pose 关键点封装
│       ├── tracker.py          # ByteTrack-lite 实现
│       ├── rules.py            # 规则引擎 A/B/C
│       ├── classifier.py       # 融合特征姿态分类器
│       ├── fusion.py           # 融合决策器
│       ├── pipeline.py         # 端到端推理 pipeline
│       ├── utils.py            # 画图/坐标/IO 工具
│       └── export.py           # ONNX 导出脚本
├── tests/
│   ├── test_detector.py
│   ├── test_tracker.py
│   ├── test_pose_estimator.py
│   ├── test_rules.py
│   ├── test_classifier.py
│   ├── test_fusion.py
│   └── test_pipeline.py
├── scripts/
│   ├── run_pipeline_demo.py    # 端到端视频 demo
│   └── benchmark_speed.py      # 速度 benchmark
├── requirements.txt
└── README.md
```

---

## Task 1: 项目骨架与基础依赖

**Files:**
- Create: `requirements.txt`
- Create: `configs/default.yaml`
- Create: `src/fall_detection/__init__.py`
- Create: `README.md`

- [ ] **Step 1: 写 requirements.txt**

内容包含：`torch`, `ultralytics`, `opencv-python`, `numpy`, `pytest`, `onnx`, `onnxruntime`, `pyyaml`, `scipy`

- [ ] **Step 2: 安装依赖**

```bash
# 推荐：自动检测CUDA版本
bash scripts/shell/install.sh

# 或指定CUDA版本
pip install -e ".[torch-cu124,dev]"
```

Expected: 成功安装，无报错。

- [ ] **Step 3: 创建 src/fall_detection/__init__.py**

空文件即可。

- [ ] **Step 4: 写默认配置 configs/default.yaml**

明确键名和值：
- detector: {conf_thresh: 0.3}
- tracker: {track_thresh: 0.5, match_thresh: 0.8, max_age: 30, min_hits: 3}
- rules: {h_ratio_thresh: 0.5, n_ground_min: 3, motion_window_seconds: 1.5, trigger_thresh: 0.6}
- fusion: {alpha: 0.35, beta: 0.45, gamma: 0.20, alarm_thresh: 0.70, alarm_min_frames: 5, reset_seconds: 3.0}
- pipeline: {skip_frames: 2, fps: 25}

- [ ] **Step 5: Commit**

```bash
git add .
git commit -m "chore: init project skeleton with deps and config"
```

---

## Task 2: 检测器模块与单元测试

**Files:**
- Create: `src/fall_detection/detector.py`
- Create: `tests/test_detector.py`

- [ ] **Step 1: 写失败测试 test_detector.py**

```python
from fall_detection.core.detector import PersonDetector

def test_detector_loads():
    det = PersonDetector(model_name='yolov8n')
    assert det is not None

def test_detector_inference_shape():
    import numpy as np
    det = PersonDetector(model_name='yolov8n')
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    boxes = det(img)
    assert isinstance(boxes, list)
```

Run: `pytest tests/test_detector.py -v`
Expected: FAIL (detector module not found)

- [ ] **Step 2: 实现 detector.py**

封装 `ultralytics.YOLO('yolov8n.pt')`，提供 `PersonDetector.__call__(img) -> List[Dict]` 接口。每个 Dict 包含 `bbox: [x1,y1,x2,y2]`, `conf: float`, `class_id: int`（只返回 person 类）。输入支持 numpy array (HWC, BGR)。

- [ ] **Step 3: 运行测试通过**

Run: `pytest tests/test_detector.py -v`
Expected: PASS（第一次运行会自动下载 yolov8n.pt，需要有网；若离线，则 agent 需先确认环境）

- [ ] **Step 4: Commit**

```bash
git add src/fall_detection/detector.py tests/test_detector.py
git commit -m "feat: add person detector wrapper with yolov8n"
```

---

## Task 3: 跟踪器模块与单元测试

**Files:**
- Create: `src/fall_detection/tracker.py`
- Create: `tests/test_tracker.py`

- [ ] **Step 1: 写失败测试 test_tracker.py**

```python
from fall_detection.core.tracker import ByteTrackLite, Detection

def test_tracker_update():
    tracker = ByteTrackLite()
    dets = [Detection([10,10,30,30], 0.8)]
    tracks = tracker.update(dets)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
```

Run: `pytest tests/test_tracker.py -v`
Expected: FAIL

- [ ] **Step 2: 实现 tracker.py**

实现最小可用的 ByteTrack-lite：
- `Detection` dataclass（bbox，conf，可选嵌入为空）
- `Track` 类（维护 bbox、状态、丢失帧计数、track_id）
- `ByteTrackLite` 类：`update(detections) -> List[Track]`
- 包含卡尔曼滤波（可用 `filterpy` 或纯 numpy 自研 8 状态常速模型）
- 匈牙利匹配用 `scipy.optimize.linear_sum_assignment` 或纯 numpy 实现
- 参数：`track_thresh=0.5`, `match_thresh=0.8`, `max_age=30`, `min_hits=3`

注：使用手写卡尔曼（8 状态常速模型，纯 numpy）+ numpy IoU + scipy.optimize.linear_sum_assignment。

- [ ] **Step 3: 运行测试通过**

Run: `pytest tests/test_tracker.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/fall_detection/tracker.py tests/test_tracker.py
git commit -m "feat: add byte-track-lite tracker"
```

---

## Task 4: 关键点估计模块与单元测试

**Files:**
- Create: `src/fall_detection/pose_estimator.py`
- Create: `tests/test_pose_estimator.py`

- [ ] **Step 1: 写失败测试 test_pose_estimator.py**

```python
from fall_detection.core.pose_estimator import PoseEstimator
import numpy as np

def test_pose_inference():
    est = PoseEstimator(model_name='yolov8n-pose')
    img = np.zeros((480,640,3), dtype=np.uint8)
    bboxes = [[10,10,100,200]]
    kpts_list = est(img, bboxes)
    assert len(kpts_list) == 1
    assert kpts_list[0].shape == (17, 3)
```

Run: `pytest tests/test_pose_estimator.py -v`
Expected: FAIL

- [ ] **Step 2: 实现 pose_estimator.py**

封装 `ultralytics.YOLO('yolov8n-pose.pt')`，接口 `PoseEstimator.__call__(img, bboxes: List[List]) -> List[np.ndarray]`，每个结果 shape=(17,3)，格式：`[x, y, conf]`。处理逻辑（快速冷启动方案）：
1. 对当前帧整张图只跑一次 YOLOv8n-pose（batch=1），得到图中所有人的 pose boxes 和对应的 17 个关键点
2. 对每个输入的 track bbox，按 IoU 与 pose 输出中的 person box 做匈牙利/贪心匹配，关联到最佳 pose 结果
3. 返回与输入 bboxes 一一对应的关键点列表；未匹配到的返回全零数组
注：本阶段采用整图批量推理以快速验证 pipeline；后续训练阶段会替换为 ROI crop + 轻量自定义网络的方案。

- [ ] **Step 3: 运行测试通过**

Run: `pytest tests/test_pose_estimator.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/fall_detection/pose_estimator.py tests/test_pose_estimator.py
git commit -m "feat: add pose estimator wrapper with yolov8n-pose"
```

---

## Task 5: 规则引擎与单元测试

**Files:**
- Create: `src/fall_detection/rules.py`
- Create: `tests/test_rules.py`

- [ ] **Step 1: 写失败测试 test_rules.py**

测试三组场景：
1. 正常站立 → S_rule 低
2. 模拟跌倒（H_ratio 低 + N_ground≥3 + 在地面 ROI 内 + 由动到静）→ S_rule≥0.6
3. 蹲下但未跌倒（H_ratio 低但无持续静止）→ S_rule 不够触发

```python
from fall_detection.core.rules import RuleEngine
import numpy as np

def test_fall_detected():
    engine = RuleEngine()
    # 构造 17 个关键点，模拟人平躺
    kpts = np.zeros((17,3))
    # ... 设置头、肩、膝、踝坐标 ...
    score, flags = engine.evaluate(kpts, bbox=[0,0,100,200], history={...})
    assert score >= 0.6
```

Run: `pytest tests/test_rules.py -v`
Expected: FAIL

- [ ] **Step 2: 实现 rules.py**

`RuleEngine` 类：
- `__init__(self, config=None)` 从 dict 加载阈值
- `evaluate(self, kpts(17,3), bbox, history) -> (S_rule, dict_flags)`
- A 判定：计算 `head_y`（nose/eyes 平均）和 `ankle_y`（两踝平均），`H_ratio = (head_y - ankle_y) / bbox_h`。`H_ratio<0.5` 且 `N_ground≥3` 则 A 触发
- B 判定：地面 ROI 为可配置的多边形列表（默认全图即始终通过，方便测试）。统计 `y` 最低的三个关键点是否在 ROI 内则 B 触发
- C 判定：`history` 传入最近 1.5s 的 track 质心列表。计算前半段（0~0.5s）和后半段（0.5~1.5s）位移标准差，判断由动到静+持续静止
- 返回 S_rule 和 dict `{'A':bool, 'B':bool, 'C':bool}`

- [ ] **Step 3: 运行测试通过**

Run: `pytest tests/test_rules.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/fall_detection/rules.py tests/test_rules.py
git commit -m "feat: add rule engine for fall detection"
```

---

## Task 6: 姿态分类器与单元测试

**Files:**
- Create: `src/fall_detection/classifier.py`
- Create: `tests/test_classifier.py`

- [ ] **Step 1: 写失败测试 test_classifier.py**

```python
from fall_detection.models.classifier import FallClassifier
import numpy as np

def test_classifier_forward():
    clf = FallClassifier()
    roi = np.zeros((3,96,96), dtype=np.float32)
    kpts = np.zeros((17,3), dtype=np.float32)
    motion = np.zeros(8, dtype=np.float32)
    prob = clf(roi, kpts, motion)
    assert 0.0 <= prob <= 1.0
```

Run: `pytest tests/test_classifier.py -v`
Expected: FAIL

- [ ] **Step 2: 实现 classifier.py**

`FallClassifier` nn.Module：
- 图像分支：2 层 Conv(3→16→32, 3×3, stride2/stride2) + ReLU + GAP → 32-d
- 关键点分支：Flatten(17×3=51) → Linear(51→32) → ReLU → 32-d
- 运动分支：Linear(8→8) → ReLU → 8-d
- fusion: concat(32+32+8=72) → Linear(72→32) → Dropout(0.3) → Linear(32→1) → Sigmoid
- 提供 `forward` 和 `__call__` 接口
- 初始化随机权重即可（至此只做结构验证）

- [ ] **Step 3: 运行测试通过**

Run: `pytest tests/test_classifier.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/fall_detection/classifier.py tests/test_classifier.py
git commit -m "feat: add lightweight fall classifier network"
```

---

## Task 7: 融合决策器与单元测试

**Files:**
- Create: `src/fall_detection/fusion.py`
- Create: `tests/test_fusion.py`

- [ ] **Step 1: 写失败测试 test_fusion.py**

```python
from fall_detection.core.fusion import FusionDecision

def test_alarm_triggered():
    fd = FusionDecision()
    for _ in range(10):
        fd.update(rule_score=0.5, cls_score=0.9)
    is_fall = fd.decide()
    assert is_fall is True
```

Run: `pytest tests/test_fusion.py -v`
Expected: FAIL

- [ ] **Step 2: 实现 fusion.py**

`FusionDecision` 类：
- `__init__` 加载配置（α=0.35, β=0.45, γ=0.20, T_alarm=0.70, N_alarm_frames=5, reset_seconds=3.0, fps=25）
- `update(rule_score, cls_score)`：维护最近 K 帧分类器得分的滑动窗口，计算 `S_temporal = mean(S_cls_history)`，然后 `S_final = α*rule_score + β*cls_score + γ*S_temporal`
- `decide() -> bool`：若 `S_final >= T_alarm` 且连续满足 N_alarm_frames，返回 True；若自上次满足后已连续 `reset_seconds*fps` 帧不满足，则清除状态返回 False
- 提供 `get_state()` 返回当前 S_final, S_temporal, alarm_frames, is_falling

- [ ] **Step 3: 运行测试通过**

Run: `pytest tests/test_fusion.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/fall_detection/fusion.py tests/test_fusion.py
git commit -m "feat: add fusion decision module"
```

---

## Task 8: 端到端 Pipeline + 可视化 Demo

**Files:**
- Create: `src/fall_detection/pipeline.py`
- Create: `src/fall_detection/utils.py`
- Create: `scripts/run_pipeline_demo.py`

- [ ] **Step 1: 先写 utils.py（画图辅助）**

```python
def draw_results(frame, tracks, kpts, rule_score, cls_score, is_falling):
    # 画 bbox + track_id
    # 画 17 关键点骨架（按 COCO 格式连线）
    # 在左上角 overlay 规则分、分类器分、跌倒状态
```

- [ ] **Step 2: 写 pipeline.py**

`FallDetectionPipeline` 类：
- `__init__(self, config_path)`：加载配置，初始化 detector/tracker/pose/rules/classifier/fusion
- `process_frame(self, frame) -> dict`：
  1. 按抽帧间隔决定是否跑检测；非检测帧直接调用 `tracker.predict()` 补位，不跑检测和 pose
  2. 检测 → `tracker.update(detections)` → 得到 `active_tracks: List[Track]`
  3. 收集所有 active_tracks 的 bbox 列表，调用 `pose_estimator(frame, bboxes)` 一键批量获取全部关键点；返回结果按顺序映射回 `track_id`（建立 `track_id -> kpts(17,3)` 字典）。若 track bbox 尺寸过小（如高<20 像素）则跳过 pose，对应 keypoints 填零
  4. 对每个 active_track，用 rules.evaluate(kpts, bbox, history[track_id]) 计算 S_rule
  5. 若 `S_rule >= T_trigger`，提取该 track 的 ROI 图 + kpts + motion 特征，过分类器 forward 得到 S_cls；否则 S_cls=0
     - motion 特征 8-d 定义：`[vx, vy, ax, ay, bbox_w, bbox_h, H_ratio, N_ground]`，其中 v/a 从 track 历史质心在最近 0.5s 内的一阶/二阶差分得到；若历史不足则补零
  6. 对每个 track 调用 `fusion.update(rule_score, cls_score)` 和 `decide()`，输出 `track_id -> is_falling` 状态
  7. 返回结果 dict（tracks, track_kpts Dict[int, np.ndarray], track_scores Dict[int, dict], track_falling Dict[int, bool]）

- [ ] **Step 3: 写 demo 脚本 run_pipeline_demo.py**

```python
import cv2
from fall_detection.pipeline import FallDetectionPipeline

pipeline = FallDetectionPipeline('configs/default.yaml')
cap = cv2.VideoCapture('data/sample.mp4')  # 若不存在则生成一张空白图做测试
while True:
    ret, frame = cap.read()
    if not ret: break
    results = pipeline.process_frame(frame)
    # 调用 utils.draw_results
    cv2.imshow('demo', frame)
    if cv2.waitKey(1) == 27: break
```

测试时可用：如果没有 sample.mp4，先生成一段 10fps 的空白测试帧序列。

- [ ] **Step 4: 写集成测试 test_pipeline.py**

为确保检测器至少返回一个框从而验证完整链路，测试中 mock detector 返回固定框：
```python
import numpy as np
from fall_detection.pipeline import FallDetectionPipeline

def test_pipeline_runs_3_frames_with_mock():
    pipe = FallDetectionPipeline('configs/default.yaml')
    # 临时替换 detector 为 mock，返回一个固定人体框
    pipe.detector = lambda img: [{'bbox': [100,100,200,400], 'conf': 0.9, 'class_id': 0}]
    for i in range(3):
        frame = np.random.randint(0,255,(480,640,3), dtype=np.uint8)
        out = pipe.process_frame(frame)
        assert 'tracks' in out
        assert 'track_kpts' in out
        assert len(out['tracks']) >= 1
```

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fall_detection/pipeline.py src/fall_detection/utils.py scripts/run_pipeline_demo.py tests/test_pipeline.py
git commit -m "feat: add end-to-end pipeline and demo script"
```

---

## Task 9: ONNX 导出与量化部署模拟测试

**Files:**
- Create: `src/fall_detection/export.py`
- Create: `tests/test_export.py`

- [ ] **Step 1: 写 export.py**

提供 `export_classifier_onnx(out_path='fall_classifier.onnx')`：
- 加载 `FallClassifier()`
- 构造 dummy inputs（roi: (1,3,96,96), kpts: (1,17,3), motion: (1,8)）
- `torch.onnx.export` 导出 ONNX

- [ ] **Step 2: 写 test_export.py**

```python
from fall_detection.utils.export import export_classifier_onnx
import onnxruntime as ort
import numpy as np

def test_classifier_onnx_export():
    export_classifier_onnx('/tmp/fall_classifier.onnx')
    sess = ort.InferenceSession('/tmp/fall_classifier.onnx')
    names = [i.name for i in sess.get_inputs()]
    dummy = {
        names[0]: np.zeros((1,3,96,96), dtype=np.float32),
        names[1]: np.zeros((1,17,3), dtype=np.float32),
        names[2]: np.zeros((1,8), dtype=np.float32),
    }
    out = sess.run(None, dummy)
    assert len(out) == 1
    assert 0 <= out[0][0][0] <= 1
```

Run: `pytest tests/test_export.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add src/fall_detection/export.py tests/test_export.py
git commit -m "feat: add onnx export and deployment simulation test"
```

---

## Task 10: 运行全部测试并修复问题

- [ ] **Step 1: 运行全部测试**

```bash
pytest tests/ -v
```

Expected: 全部 PASS。如果有失败，定位并修复。

- [ ] **Step 2: Commit 最终修复**

```bash
git add .
git commit -m "fix: all tests green for end-to-end pipeline"
```

---

## Task 11: benchmark 速度测试脚本

**Files:**
- Create: `scripts/benchmark_speed.py`

- [ ] **Step 1: 实现 benchmark_speed.py**

生成 100 帧 640×480 随机图像，跑 pipeline process_frame，统计：
- 总耗时、平均每帧耗时（ms）、FPS
- 检测耗时、跟踪耗时、关键点耗时、规则+分类+融合耗时

- [ ] **Step 2: 运行 benchmark**

```bash
PYTHONPATH=src python scripts/benchmark_speed.py
```

Expected: 成功输出 benchmark 结果到终端。

- [ ] **Step 3: Commit**

```bash
git add scripts/benchmark_speed.py
git commit -m "feat: add speed benchmark script"
```

---

## 完成标准

- [ ] 所有 `tests/` 下的 pytest 通过
- [ ] `scripts/run_pipeline_demo.py` 能在单张/视频上跑通并可视化
- [ ] `scripts/benchmark_speed.py` 能输出 FPS 和延迟
- [ ] `export.py` 能成功导出 ONNX，且 ONNX Runtime 推理结果与 PyTorch 一致
