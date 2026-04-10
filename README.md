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
├── configs/                          # 配置文件
 │   ├── pipeline/                     # Pipeline配置
 │   │   └── default.yaml              # 系统默认配置
 │   ├── training/                     # 训练配置
 │   │   ├── detector.yaml             # 检测器训练配置
 │   │   ├── pose.yaml                 # 姿态估计训练配置
 │   │   ├── classifier.yaml           # 融合分类器训练配置
 │   │   ├── simple_classifier.yaml    # 简单分类器训练配置
 │   │   ├── simple_classifier_voc.yaml # VOC数据集分类器配置
 │   │   └── yoloworld.yaml            # YOLOWorld训练配置
 │   └── tools/                        # 工具配置
 │       └── voc_to_yolo_example.yaml  # VOC转YOLO示例配置
 │
├── src/fall_detection/               # 核心推理模块
│   ├── core/                         # 核心推理组件
│   │   ├── detector.py               # YOLOv8 人体检测器
│   │   ├── tracker.py                # ByteTrack-lite 跟踪器
│   │   ├── pose_estimator.py         # YOLOv8-pose 姿态估计
│   │   ├── rules.py                  # 规则引擎
│   │   └── fusion.py                 # 融合决策器
│   ├── models/                       # 模型定义
│   │   ├── classifier.py             # 融合分类器（3分支）
│   │   └── simple_classifier.py      # 简单图像分类器
│   ├── pipeline/                     # Pipeline
│   │   └── pipeline.py               # 端到端 Pipeline
│   ├── data/                         # 数据处理
│   │   ├── augmentation.py           # 数据增强
│   │   └── datasets.py               # 数据集类
│   └── utils/                        # 工具函数
 │       ├── visualization.py          # 可视化
 │       ├── export.py                 # 模型导出
 │       ├── scheduler.py              # 学习率调度器
 │       ├── geometry.py               # 几何计算
 │       ├── training_common.py        # 训练通用工具
 │       └── common.py                 # 通用工具
│
├── scripts/                          # 脚本
 │   ├── shell/                        # Shell脚本
 │   │   ├── run_train_detector.sh
 │   │   ├── run_train_pose.sh
 │   │   ├── run_train_classifier.sh
 │   │   ├── run_train_simple_classifier.sh
 │   │   ├── run_extract_features.sh
 │   │   ├── run_evaluate_pipeline.sh
 │   │   ├── run_tune_tracker.sh
 │   │   ├── run_pipeline_demo.sh
 │   │   ├── run_tracker_demo.sh
 │   │   ├── run_tests.sh
 │   │   ├── run_all_training.sh
│   │   ├── run_export_yoloworld.sh
│   │   ├── run_validate_yoloworld.sh
│   │   ├── download_yoloworld_models.sh
 │   │   └── install.sh
 │   ├── train/                        # 训练脚本
 │   │   ├── train_detector.py
 │   │   ├── train_pose.py
 │   │   ├── train_classifier.py
 │   │   ├── train_simple_classifier.py
 │   │   └── validate_yoloworld.py
 │   ├── eval/                         # 评估脚本
 │   │   ├── evaluate_pipeline.py
 │   │   ├── benchmark_speed.py
 │   │   └── tune_tracker.py
 │   ├── tools/                        # 数据处理工具
 │   │   ├── convert_voc_to_yolo.py
 │   │   ├── extract_and_detect.py
 │   │   ├── extract_features.py
 │   │   ├── export_yoloworld.py
 │   │   └── export_yoloworld.py
 │   └── demo/                         # 演示脚本
 │       ├── run_pipeline_demo.py
 │       ├── demo_tracker.py
 │       └── video_to_frames.py
│
├── tools/                            # 辅助工具
│   └── annotate/                     # 标注工具
│       ├── yolo_annotate.py          # YOLO自动标注
│       └── vlm_annotate.py           # VLM自动标注
│
├── tests/                            # 测试用例
│   ├── unit/                         # 单元测试
│   │   ├── test_detector.py
│   │   ├── test_tracker.py
│   │   ├── test_pose_estimator.py
│   │   ├── test_rules.py
│   │   ├── test_fusion.py
│   │   ├── test_classifier.py
│   │   ├── test_simple_classifier.py
│   │   ├── test_augmentation.py
│   │   ├── test_datasets.py
│   │   ├── test_pipeline.py
│   │   ├── test_export.py
│   │   ├── test_scheduler.py
│   │   └── test_utils_*.py
│   ├── integration/                  # 集成测试
 │   │   ├── test_training_detector.py
 │   │   ├── test_training_pose.py
 │   │   ├── test_training_classifier.py
 │   │   ├── test_training_simple_classifier.py
 │   │   ├── test_training_scripts.py
 │   │   └── test_extract_features.py
 │   └── conftest.py                   # 测试配置
│
├── data/                             # 数据目录
│   ├── mini/                         # 示例数据集
│   ├── videos/                       # 视频文件
│   └── annotations/                  # 标注文件
│
├── outputs/                          # 训练输出目录
│   ├── detector/
│   ├── pose/
│   ├── classifier/
│   ├── simple_classifier/
│   ├── tracker/
│   ├── cache/
│   └── eval/
│
├── docs/                             # 文档
│   ├── api/                          # API文档
│   ├── design/                       # 设计文档
│   ├── development/                  # 开发指南
│   └── troubleshooting/              # 故障排查
│
├── INSTALL.md                        # 详细安装指南
├── requirements.txt                  # 快速安装入口（CUDA 12.4）
├── pyproject.toml                    # 项目配置和依赖定义
├── README.md                         # 项目说明
├── CLAUDE.md                         # AI助手指引
├── CONTRIBUTING.md                   # 贡献指南
├── CHANGELOG.md                      # 版本历史
└── LICENSE
```

## 快速开始

> 📖 **详细安装步骤请参考 [INSTALL.md](INSTALL.md)**

### 1. 安装依赖

```bash
# 推荐：自动检测CUDA版本安装
bash scripts/shell/install.sh

# 或指定CUDA版本
pip install -e ".[torch-cu124,dev]"

# CPU only
pip install -e ".[torch-cpu,dev]"
```

### 2. 验证安装

```bash
# 检查依赖
python -c "import torch, ultralytics; print('✓ Dependencies OK')"

# 运行测试
PYTHONPATH=src pytest tests/unit/test_detector.py -v
```

### 3. 运行演示

```bash
# 视频演示
PYTHONPATH=src python scripts/demo/run_pipeline_demo.py \
    --video data/video/test.mp4 --output output.mp4

# 摄像头实时演示
PYTHONPATH=src python scripts/demo/run_pipeline_demo.py --video 0
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

结果保存在 `outputs/tracker/tune_result.json`

### 阶段4: 特征提取（分类器训练数据准备）

```bash
bash scripts/shell/run_extract_features.sh \
  --video-dir data/videos \
  --label-file data/labels.json \
  --out-dir outputs/cache
```

特征缓存格式（.npz）：
- `roi`: 裁剪ROI图像 (96x96x3, uint8)
- `kpts`: 17点关键点 (17x3, float32)
- `motion`: 8维运动特征 (8,), float32)
- `label`: 跌倒标签 (0/1, int64)

### 阶段5: 融合分类器训练

 ```bash
 # 单卡训练
 bash scripts/shell/run_train_classifier.sh
 
 # 多卡DDP训练
 bash scripts/shell/run_train_classifier.sh --ngpus 2 --batch-size 16
 ```
 
 参数说明：
 - `--cache-dir`: 特征缓存目录 (默认: outputs/cache)
 - `--epochs`: 训练轮数 (默认: 100)
 - `--batch-size`: 批次大小 (默认: 32)
 - `--lr`: 学习率 (默认: 0.001)
 - `--val-ratio`: 验证集比例 (默认: 0.2)
 
 最佳权重保存为 `outputs/classifier/best.pt`
 
### 阶段6: 简单图像分类器训练

 ```bash
 # 使用配置文件训练
 bash scripts/shell/run_train_simple_classifier.sh \
   --config configs/training/simple_classifier.yaml
 
 # DDP多GPU训练
 bash scripts/shell/run_train_simple_classifier.sh \
   --config configs/training/simple_classifier.yaml \
   --ngpus 2
 
 # 覆盖配置参数
 bash scripts/shell/run_train_simple_classifier.sh \
   --config configs/training/simple_classifier.yaml \
   --override "lr=0.01,batch_size=128"
 ```
 
 ### 阶段7: 端到端评估与阈值搜索

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

结果保存在 `outputs/eval/eval_result.json`

### 一键运行完整训练流程

```bash
# 使用环境变量指定GPU数量
NGPUS=2 bash scripts/shell/run_all_training.sh
```

## 模型导出与部署

 ### ONNX 导出

 ```python
 from fall_detection.models import FallClassifier, SimpleFallClassifier
 from fall_detection.utils.export import export_classifier_onnx, export_simple_classifier_onnx
 
 # 导出融合分类器
 model = FallClassifier()
 model.load_state_dict(torch.load('outputs/classifier/best.pt'))
 export_classifier_onnx(model, 'fall_classifier.onnx')
 
 # 导出简单分类器
 model = SimpleFallClassifier()
 model.load_state_dict(torch.load('outputs/simple_classifier/best.pt'))
 export_simple_classifier_onnx(model, 'simple_fall_classifier.onnx')
 ```
 
### YOLOWorld 模型导出与验证

```bash
# 导出YOLOWorld模型（支持非正方形分辨率）
bash scripts/shell/run_export_yoloworld.sh \
  --weights outputs/yoloworld/best.pt \
  --imgsz 832x448 \
  --format onnx

# 验证YOLOWorld模型性能
bash scripts/shell/run_validate_yoloworld.sh \
  --weights outputs/yoloworld/best.pt \
  --data data/configs/fall_detection_yoloworld.yaml

# 下载预训练YOLOWorld模型
bash scripts/shell/download_yoloworld_models.sh
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
python scripts/eval/benchmark_speed.py --video data/videos/test.mp4 --num-frames 100
```

## 配置说明

编辑 `configs/pipeline/default.yaml` 调整系统参数：

```yaml
detector:
  conf_thresh: 0.3          # 检测置信度阈值
  model_path: "data/models/pretrained/yolov8n.pt"
  classes: null             # 开放词汇检测器文本提示

pose_estimator:
  model_path: "data/models/pretrained/yolov8n-pose.pt"

classifier:
  type: "simple"            # 分类器类型: "fusion" 或 "simple"
  model_path: "outputs/simple_classifier/best.pt"
  fall_class_idx: 1         # 跌倒类别索引

tracker:
  track_thresh: 0.5         # 跟踪置信度阈值
  match_thresh: 0.8         # 匈牙利匹配阈值
  max_age: 30               # 最大丢失帧数
  min_hits: 3               # 最小确认帧数

rules:
  h_ratio_thresh: 0.60      # 高度压缩比阈值
  n_ground_min: 2           # 最小贴地点数
  trigger_thresh: 0.60      # 规则触发阈值
  motion_thresh: 50.0       # 运动期位移阈值
  static_thresh: 20.0       # 静止期位移阈值
  fall_vy_thresh: 200.0     # 垂直下降速度阈值
  visible_ratio_min: 0.50   # 关键点可见比例下限
  ground_ratio: 0.40        # 贴地判定区域比例
  fps: 25                   # 规则引擎归一化帧率

fusion:
  alpha: 0.5                # 规则分权重
  beta: 0.3                 # 分类器分权重
  gamma: 0.2                # 时序分权重
  alarm_thresh: 0.50        # 告警阈值
  alarm_min_frames: 3       # 最小告警帧数
  cooldown_seconds: 3.0     # 冷却期
  recovery_seconds: 0.5     # 恢复确认期

pipeline:
  skip_frames: 2            # 检测间隔帧数 (每3帧检测1次)
  fps: 25                   # 输入视频帧率
```

## 自动标注工具

项目提供两种自动标注方式，用于快速生成训练数据：

### YOLO自动标注

使用预训练YOLOv8模型检测人体，生成Pascal VOC格式标注：

```bash
# 单张图像
python tools/annotate/yolo_annotate.py \
    --input data/my_image.jpg \
    --output-dir outputs/Annotations \
    --vis-dir outputs/Visualizations

# 批量处理
python tools/annotate/yolo_annotate.py \
    --input data/images/ \
    --output-dir outputs/Annotations \
    --model yolov8n.pt \
    --conf-threshold 0.3
```

详细文档：[YOLO_ANNOTATION_README.md](docs/api/YOLO_ANNOTATION_README.md)

### VLM自动标注

使用视觉语言模型(Claude/阿里百炼)分析图像，识别人物状态并生成标注：

```bash
# 设置API密钥
export ANTHROPIC_API_KEY="your-api-key"
# 或 export DASHSCOPE_API_KEY="your-api-key" (国内推荐)

# 运行标注
python tools/annotate/vlm_annotate.py \
    --input data/images/ \
    --output-dir outputs/Annotations \
    --vis-dir outputs/Visualizations \
    --model claude
```

详细文档：[VLM_ANNOTATION_README.md](docs/api/VLM_ANNOTATION_README.md)

## 数据集格式

### 检测/姿态训练数据

使用 Ultralytics YOLO 格式，参见 [Ultralytics 文档](https://docs.ultralytics.com/datasets/)

### 简单分类器训练数据 (COCO格式)

```json
{
  "images": [{"id": 1, "file_name": "img1.jpg", "height": 1080, "width": 1920}],
  "annotations": [
    {"image_id": 1, "bbox": [100, 200, 50, 100], "category_id": 1}
  ],
  "categories": [{"id": 1, "name": "fall"}]
}
```

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

### 模块导入示例

```python
# 导入核心组件
from fall_detection.core import PersonDetector, ByteTrackLite, PoseEstimator, RuleEngine, FusionDecision

# 导入模型
from fall_detection.models import FallClassifier, SimpleFallClassifier

# 导入Pipeline
from fall_detection.pipeline import FallDetectionPipeline

# 导入工具函数
from fall_detection.utils import draw_results, load_config
from fall_detection.utils.export import export_classifier_onnx
```

### 添加新的规则

编辑 `src/fall_detection/core/rules.py`：

```python
def evaluate(self, kpts, bbox, history):
    # 添加新的判定逻辑
    flags["D"] = self._check_new_rule(kpts, bbox)
    # ...
```

### 自定义分类器网络

编辑 `src/fall_detection/models/classifier.py`：

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

## 📚 文档资源

| 文档 | 说明 |
|------|------|
| [INSTALL.md](INSTALL.md) | 详细安装指南和故障排查 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 贡献指南和开发设置 |
| [CHANGELOG.md](CHANGELOG.md) | 版本历史和变更记录 |
| [CLAUDE.md](CLAUDE.md) | AI助手指引和编码规范 |
| [docs/api/](docs/api/) | API使用文档 |
| [docs/troubleshooting/](docs/troubleshooting/) | 故障排查指南 |

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

## 🙏 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- COCO 关键点格式规范

## 联系方式

如有问题或建议，欢迎提交 Issue 或 Pull Request。
