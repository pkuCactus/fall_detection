# 安装指南

本文档提供从零开始安装和使用跌倒检测系统的完整步骤。

## 目录

- [环境要求](#环境要求)
- [快速安装](#快速安装)
- [手动安装](#手动安装)
- [验证安装](#验证安装)
- [下载预训练模型](#下载预训练模型)
- [快速开始](#快速开始)
- [故障排查](#故障排查)

---

## 环境要求

### 操作系统
- Linux (推荐 Ubuntu 20.04+)
- macOS
- Windows (WSL2 推荐)

### Python
- Python 3.10+
- pip 21.0+

### 硬件
| 用途 | 最低配置 | 推荐配置 |
|------|----------|----------|
| **训练** | 8GB RAM, GPU 6GB | 32GB RAM, GPU 12GB+ |
| **推理** | 4GB RAM | 8GB RAM |
| **存储** | 10GB | 50GB+ (含数据集) |

### CUDA (可选)
| CUDA版本 | PyTorch变体 | 驱动版本 |
|----------|-------------|----------|
| 12.4 | torch-cu124 | ≥ 525 |
| 12.1 | torch-cu121 | ≥ 520 |
| 11.8 | torch-cu118 | ≥ 450 |
| CPU | torch-cpu | N/A |

---

## 快速安装

### 方法1: 自动检测CUDA版本 (推荐)

```bash
# 克隆仓库
git clone https://github.com/pkuCactus/fall_detection.git
cd fall_detection

# 自动检测CUDA版本并安装
bash scripts/shell/install.sh
```

安装脚本会：
1. 检测NVIDIA驱动版本
2. 自动选择合适的PyTorch变体
3. 安装所有依赖

### 方法2: 指定CUDA版本

```bash
# 克隆仓库
git clone https://github.com/pkuCactus/fall_detection.git
cd fall_detection

# 安装特定CUDA版本
pip install -e ".[torch-cu124,dev]"  # CUDA 12.4

# 其他版本
# pip install -e ".[torch-cu121,dev]"  # CUDA 12.1
# pip install -e ".[torch-cu118,dev]"  # CUDA 11.8
# pip install -e ".[torch-cpu,dev]"    # CPU only
```

### 方法3: 传统方式

```bash
# 克隆仓库
git clone https://github.com/pkuCactus/fall_detection.git
cd fall_detection

# 使用requirements.txt (默认CUDA 12.4)
pip install -r requirements.txt
```

---

## 手动安装

### 步骤1: 创建虚拟环境

```bash
# 使用venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows

# 或使用conda
conda create -n fall_detection python=3.10
conda activate fall_detection
```

### 步骤2: 安装PyTorch

根据你的CUDA版本选择：

```bash
# CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 步骤3: 安装项目依赖

```bash
# 安装项目包（包含所有核心依赖）
pip install -e "."

# 安装开发工具（可选）
pip install -e ".[dev]"
```

### 步骤4: 验证安装

```bash
# 检查Python版本
python --version  # 应显示 3.10+

# 检查PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 检查项目包
python -c "from fall_detection import __version__; print(f'fall_detection: {__version__}')"
```

---

## 验证安装

运行以下命令验证安装是否成功：

```bash
# 1. 检查核心依赖
python -c "
import torch
import ultralytics
import cv2
import numpy as np
print('✓ All core dependencies installed')
print(f'  PyTorch: {torch.__version__}')
print(f'  Ultralytics: {ultralytics.__version__}')
print(f'  OpenCV: {cv2.__version__}')
print(f'  NumPy: {np.__version__}')
"

# 2. 检查CUDA (如果使用GPU)
python -c "
import torch
if torch.cuda.is_available():
    print(f'✓ CUDA available: {torch.version.cuda}')
    print(f'  Device: {torch.cuda.get_device_name(0)}')
else:
    print('⚠ CUDA not available, using CPU')
"

# 3. 运行测试
PYTHONPATH=src pytest tests/unit/test_detector.py -v
```

---

## 下载预训练模型

首次运行时会自动下载，也可以手动下载：

```bash
# 创建模型目录
mkdir -p data/models/pretrained

# 下载YOLOv8n (人体检测器)
wget -P data/models/pretrained https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt

# 下载YOLOv8n-pose (姿态估计器)
wget -P data/models/pretrained https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n-pose.pt
```

或使用Python自动下载：

```bash
python -c "
from ultralytics import YOLO
YOLO('yolov8n.pt')
YOLO('yolov8n-pose.pt')
print('✓ Pretrained models downloaded')
"
```

---

## 快速开始

### 1. 准备测试数据

```bash
# 创建测试视频目录
mkdir -p data/video

# 将测试视频放入该目录
# 或使用示例视频（如果有）
```

### 2. 运行演示

```bash
# 视频文件演示
PYTHONPATH=src python scripts/demo/run_pipeline_demo.py \
    --video data/video/test.mp4 \
    --output output.mp4

# 摄像头实时演示
PYTHONPATH=src python scripts/demo/run_pipeline_demo.py --video 0
```

### 3. 运行测试

```bash
# 安装测试依赖
pip install -e ".[dev]"

# 运行所有测试
PYTHONPATH=src pytest tests/ -v

# 运行带覆盖率的测试
PYTHONPATH=src pytest tests/ --cov=src/fall_detection --cov-fail-under=90
```

### 4. 训练模型

```bash
# 训练检测器
bash scripts/shell/run_train_detector.sh --config configs/training/detector.yaml

# 训练姿态估计器
bash scripts/shell/run_train_pose.sh --config configs/training/pose.yaml

# 训练分类器
bash scripts/shell/run_train_simple_classifier.sh \
    --config configs/training/simple_classifier.yaml \
    --ngpus 2
```

---

## 故障排查

### 常见问题

#### 1. PyTorch CUDA版本不匹配

**症状**: `RuntimeError: CUDA out of memory` 或 CUDA不可用

**解决方案**:
```bash
# 检查CUDA版本
nvidia-smi

# 重新安装匹配的PyTorch
bash scripts/shell/install.sh --variant cu124  # 根据你的CUDA版本
```

#### 2. 找不到模块 `fall_detection`

**症状**: `ModuleNotFoundError: No module named 'fall_detection'`

**解决方案**:
```bash
# 确保已安装项目包
pip install -e "."

# 或设置PYTHONPATH
export PYTHONPATH=src
```

#### 3. OpenCV导入错误

**症状**: `ImportError: libGL.so.1: cannot open shared object file`

**解决方案** (Linux):
```bash
# Ubuntu/Debian
sudo apt-get install libgl1-mesa-glx libglib2.0-0

# 或使用headless版本
pip install opencv-python-headless
```

#### 4. Ultralytics下载模型失败

**症状**: 下载超时或网络错误

**解决方案**:
```bash
# 手动下载模型
wget -P data/models/pretrained https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt

# 或设置代理
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
```

#### 5. 测试覆盖率不达标

**症状**: `FAILED tests/ - coverage 85% < 90%`

**解决方案**:
```bash
# 查看缺失覆盖率的代码
PYTHONPATH=src pytest tests/ --cov=src/fall_detection --cov-report=html

# 打开HTML报告
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### 获取帮助

- **文档**: [README.md](README.md)
- **问题**: [GitHub Issues](https://github.com/pkuCactus/fall_detection/issues)
- **贡献**: [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 下一步

安装完成后，你可以：

1. 📖 阅读 [README.md](README.md) 了解系统架构
2. 🚀 运行 [演示脚本](scripts/demo/) 快速体验
3. 🧪 运行 [测试](tests/) 验证功能
4. 🏋️ 按照 [训练流程](README.md#分阶段训练流程) 训练模型
5. 📚 查看 [API文档](docs/api/) 了解更多

---

## 安装验证清单

- [ ] Python 3.10+ 已安装
- [ ] 虚拟环境已激活
- [ ] PyTorch 2.5+ 已安装
- [ ] CUDA 可用 (如果使用GPU)
- [ ] 项目包已安装 (`pip install -e "."`)
- [ ] 核心依赖已安装
- [ ] 预训练模型已下载
- [ ] 测试通过 (`pytest tests/ -v`)

完成以上步骤后，你的环境已准备就绪！