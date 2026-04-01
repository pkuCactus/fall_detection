# 边缘AI跌倒检测系统

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

专为海思 HiSilicon 3516C 平台设计的纯视觉边缘跌倒检测系统。在 0.5T INT8 算力、15M DDR、30M Flash 的极端资源约束下，实现实时多人跌倒检测。

## 系统特性

- **纯视觉方案**: 仅依赖 RGB 视频输入，无需深度传感器或毫米波雷达
- **边缘优化**: 总模型权重 ≤ 6M，适配 3516C 0.5T INT8 算力
- **多人跟踪**: ByteTrack-lite 跟踪器，支持多人场景同时检测
- **融合决策**: 规则引擎 + 轻量分类器 + 时序融合，降低误报率
- **模块化设计**: 检测/跟踪/姿态/规则/分类器可独立训练优化
- **DDP支持**: 分类器训练支持多卡分布式训练

## 硬件要求

### 目标平台 (海思 3516C)
| 指标 | 规格 |
|------|------|
| 算力 | 0.5T INT8 |
| 内存 | 15M DDR |
| Flash | 30M |
| 输入 | 1920x1080 @ 25fps |

### 开发环境
- Python 3.10+
- PyTorch 2.x
- CUDA (可选，用于训练)

## 系统架构

```
输入视频 (1920x1080@25fps)
    ↓
[抽帧: 1/2 或 1/3]
    ↓
┌─────────────────────────────────────┐
│ 模块1: 人体检测器 (YOLOv8n)          │
│ - 输入: 832x448 INT8                │
│ - 输出: 检测框 + 置信度              │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 模块2: ByteTrack-lite 跟踪器         │
│ - 卡尔曼滤波 + 匈牙利匹配            │
│ - 无 ReID 网络，输出 Track ID        │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 模块3: 姿态估计器 (YOLOv8n-pose)     │
│ - 17点 COCO 关键点                  │
│ - 输入: 96x96 (ROI裁剪)             │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 模块4: 规则引擎 (Rule Engine)        │
│ - A: 高度压缩比 + 多点贴地           │
│ - B: 地面 ROI 判定                   │
│ - C: 运动到静止方差                  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ 模块5: 融合决策器 (Fusion)           │
│ S_final = α·S_rule + β·S_cls + γ·S_temporal │
│ - 时序滑动窗口平滑                   │
│ - 连续 N 帧触发告警                  │
└─────────────────────────────────────┘
    ↓
输出: 跌倒告警 (Track ID + 置信度)
```

## 项目结构

```
fall_detection/
├── configs/
│   └── default.yaml          # 系统配置（阈值、参数）
├── src/fall_detection/       # 核心推理模块
│   ├── detector.py           # YOLOv8 人体检测器
│   ├── tracker.py            # ByteTrack-lite 跟踪器
│   ├── pose_estimator.py     # YOLOv8-pose 姿态估计
│   ├── rules.py              # 规则引擎
│   ├── classifier.py         # 轻量融合分类器
│   ├── fusion.py             # 融合决策器
│   ├── pipeline.py           # 端到端 Pipeline
│   ├── export.py             # ONNX 导出工具
│   └── utils.py              # 可视化工具
├── scripts/                  # 训练与评估脚本
│   ├── train_detector.py     # 检测器训练
│   ├── train_pose.py         # 姿态估计器微调
│   ├── train_classifier.py   # 分类器训练（支持DDP）
│   ├── extract_features.py   # 特征提取
│   ├── tune_tracker.py       # 跟踪器参数调优
│   ├── evaluate_pipeline.py  # 端到端评估
│   ├── benchmark_speed.py    # 速度基准测试
│   ├── run_pipeline_demo.py  # 演示脚本
│   └── shell/                # Shell 脚本集合
│       ├── run_train_detector.sh
│       ├── run_train_pose.sh
│       ├── run_train_classifier.sh
│       ├── run_extract_features.sh
│       ├── run_tune_tracker.sh
│       ├── run_evaluate_pipeline.sh
│       ├── run_tests.sh
│       ├── run_all_training.sh
│       └── run_superpowers_server.sh
├── tests/                    # 测试用例
├── docs/                     # 文档
└── train/                    # 训练输出目录（.gitignore）
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

requirements.txt 主要依赖：
- torch >= 2.0
- ultralytics >= 8.0
- opencv-python
- numpy
- pyyaml
- pytest

### 2. 下载预训练权重

```bash
# 自动下载（首次运行时会自动下载）
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt'); YOLO('yolov8n-pose.pt')"
```

### 3. 运行演示

```bash
# 单视频演示
python scripts/run_pipeline_demo.py --video data/videos/test.mp4 --output output.mp4

# 摄像头实时演示
python scripts/run_pipeline_demo.py --video 0
```

### 4. 运行测试

```bash
# 运行所有测试
bash scripts/shell/run_tests.sh

# 或指定测试文件
PYTHONPATH=src pytest tests/test_pipeline.py -v
```

## 分阶段训练流程

### 阶段1: 人体检测器训练

```bash
# 单卡训练
bash scripts/shell/run_train_detector.sh

# 多卡DDP训练
bash scripts/shell/run_train_detector.sh --ngpus 2 --batch 8
```

参数说明：
- `--data`: 数据集配置文件 (默认: data/fall_detection.yaml)
- `--epochs`: 训练轮数 (默认: 50)
- `--imgsz`: 输入尺寸 (默认: 832)
- `--batch`: 批次大小 (默认: 16)
- `--model`: 预训练模型 (默认: yolov8n.pt)

### 阶段2: 姿态估计器微调

```bash
# 单卡训练
bash scripts/shell/run_train_pose.sh

# 多卡DDP训练
bash scripts/shell/run_train_pose.sh --ngpus 2 --batch 32
```

参数说明：
- `--data`: 数据集配置文件 (默认: data/fall_pose.yaml)
- `--epochs`: 训练轮数 (默认: 100)
- `--imgsz`: 输入尺寸 (默认: 128)
- `--batch`: 批次大小 (默认: 64)
- `--model`: 预训练模型 (默认: yolov8n-pose.pt)

### 阶段3: 跟踪器参数调优

```bash
bash scripts/shell/run_tune_tracker.sh --video-dir data/videos
```

网格搜索参数：
- `track_thresh`: [0.4, 0.5, 0.6]
- `match_thresh`: [0.7, 0.8, 0.9]
- `max_age`: [20, 30, 40]
- `min_hits`: [2, 3]

结果保存在 `train/tracker/tune_result.json`

### 阶段4: 特征提取（分类器训练数据准备）

```bash
bash scripts/shell/run_extract_features.sh \
  --video-dir data/videos \
  --label-file data/labels.json \
  --out-dir train/cache
```

特征缓存格式（.npz）：
- `roi`: 裁剪ROI图像 (96x96x3, uint8)
- `kpts`: 17点关键点 (17x3, float32)
- `motion`: 8维运动特征 (8,), float32
- `label`: 跌倒标签 (0/1, int64)

### 阶段5: 分类器训练

```bash
# 单卡训练
bash scripts/shell/run_train_classifier.sh

# 多卡DDP训练
bash scripts/shell/run_train_classifier.sh --ngpus 2 --batch-size 16
```

参数说明：
- `--cache-dir`: 特征缓存目录 (默认: train/cache)
- `--epochs`: 训练轮数 (默认: 100)
- `--batch-size`: 批次大小 (默认: 32)
- `--lr`: 学习率 (默认: 0.001)
- `--val-ratio`: 验证集比例 (默认: 0.2)

最佳权重保存为 `train/classifier/best.pt`

### 阶段6: 端到端评估与阈值搜索

```bash
bash scripts/shell/run_evaluate_pipeline.sh \
  --video-dir data/videos \
  --gt-file data/event_gt.json

# 使用 mock 检测器加速评估（无需真实模型）
bash scripts/shell/run_evaluate_pipeline.sh --mock-detector
```

网格搜索阈值：
- `trigger_thresh`: [0.5, 0.6, 0.7] (规则触发阈值)
- `alarm_thresh`: [0.6, 0.7, 0.8] (融合告警阈值)

结果保存在 `train/eval/eval_result.json`

### 一键运行完整训练流程

```bash
# 使用环境变量指定GPU数量
NGPUS=2 bash scripts/shell/run_all_training.sh
```

## 模型导出与部署

### ONNX 导出

```python
from src.fall_detection.classifier import FallClassifier
from src.fall_detection.export import export_classifier_onnx

model = FallClassifier()
model.load_state_dict(torch.load('train/classifier/best.pt'))
export_classifier_onnx(model, 'fall_classifier.onnx')
```

### 部署检查清单

- [ ] 检测器权重 ≤ 3M (YOLOv8n INT8)
- [ ] 姿态估计器权重 ≤ 2.5M (YOLOv8n-pose INT8)
- [ ] 分类器权重 ≤ 0.15M (FallClassifier)
- [ ] 总模型权重 ≤ 6M
- [ ] 峰值内存 ≤ 15M DDR
- [ ] 单帧推理 ≤ 40ms (25fps)

## 性能基准

在 3516C 平台（模拟）上的性能：

| 模块 | 耗时 | 内存 |
|------|------|------|
| 检测器 (YOLOv8n) | ~15ms | ~6M |
| 跟踪器 (ByteTrack) | ~2ms | ~0.5M |
| 姿态估计 (YOLOv8n-pose) | ~12ms | ~5M |
| 规则引擎 | ~1ms | 可忽略 |
| 分类器 | ~3ms | ~0.5M |
| 融合决策 | ~1ms | 可忽略 |
| **总计** | **~34ms** | **~12M** |

运行基准测试：
```bash
python scripts/benchmark_speed.py --video data/videos/test.mp4 --num-frames 100
```

## 配置说明

编辑 `configs/default.yaml` 调整系统参数：

```yaml
detector:
  conf_thresh: 0.3          # 检测置信度阈值

tracker:
  track_thresh: 0.5         # 跟踪置信度阈值
  match_thresh: 0.8         # 匈牙利匹配阈值
  max_age: 30               # 最大丢失帧数
  min_hits: 3               # 最小确认帧数

rules:
  h_ratio_thresh: 0.5       # 高度压缩比阈值
  n_ground_min: 3           # 最小贴地点数
  trigger_thresh: 0.6       # 规则触发阈值

fusion:
  alpha: 0.35               # 规则分权重
  beta: 0.45                # 分类器分权重
  gamma: 0.20               # 时序分权重
  alarm_thresh: 0.70        # 告警阈值
  alarm_min_frames: 5       # 最小告警帧数

pipeline:
  skip_frames: 2            # 检测间隔帧数 (每3帧检测1次)
  fps: 25                   # 输入视频帧率
```

## 数据集格式

### 检测/姿态训练数据

使用 Ultralytics YOLO 格式，参见 [Ultralytics 文档](https://docs.ultralytics.com/datasets/)

### 分类器标签文件 (labels.json)

```json
{
  "fall_001.mp4": {
    "label": 1,
    "frames": [120, 180]
  },
  "normal_002.mp4": {
    "label": 0
  }
}
```

### 事件评估标签文件 (event_gt.json)

```json
{
  "test_video.mp4": [
    [100, 150],
    [200, 250]
  ]
}
```

## 开发文档

### 预览 Superpowers HTML 文档

```bash
bash scripts/shell/run_superpowers_server.sh 8080
# 浏览器打开 http://localhost:8080
```

### 添加新的规则

编辑 `src/fall_detection/rules.py`：

```python
def evaluate(self, kpts, bbox, history):
    # 添加新的判定逻辑
    flags["D"] = self._check_new_rule(kpts, bbox)
    # ...
```

### 自定义分类器网络

编辑 `src/fall_detection/classifier.py`：

```python
class FallClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 修改网络结构
        # 注意保持输入输出维度一致
```

## 常见问题

**Q: 检测不到人体？**
- 检查 `detector.conf_thresh` 是否设置过高
- 确认输入视频分辨率不低于 640x480

**Q: 跟踪ID频繁切换？**
- 调整 `tracker.max_age` 和 `tracker.match_thresh`
- 运行 `run_tune_tracker.sh` 寻找最优参数

**Q: 误报率高？**
- 提高 `rules.trigger_thresh` 或 `fusion.alarm_thresh`
- 增加 `fusion.alarm_min_frames`
- 收集更多困难样本重新训练分类器

**Q: 内存不足？**
- 增大 `pipeline.skip_frames` 降低检测频率
- 减小 `detector.imgsz` 降低输入分辨率
- 使用更小的 YOLO 模型

## 许可证

MIT License

## 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- COCO 关键点格式规范

## 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。
