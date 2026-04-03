# YOLOv8 自动标注工具

使用预训练 YOLOv8 模型自动检测图像中的人，生成 Pascal VOC 格式标注。比 VLM 标注更快更准确。

## 特点

- **速度快**: 本地 GPU/CPU 推理，无需 API 调用
- **准确率高**: YOLOv8 专门优化的人体检测
- **离线可用**: 无需网络连接
- **标准格式**: 输出 Pascal VOC XML 标注

## 安装依赖

```bash
pip install ultralytics opencv-python pillow numpy

# 或安装项目全部依赖
pip install -r requirements.txt
```

## 使用方式

### 单张图像

```bash
# 仅生成标注（默认）
python deployment/yolo_annotate.py \
    --input data/my_image.jpg \
    --output-dir outputs/Annotations

# 生成标注 + 可视化
python deployment/yolo_annotate.py \
    --input data/my_image.jpg \
    --output-dir outputs/Annotations \
    --vis-dir outputs/Visualizations
```

### 批量处理目录

```bash
# 仅生成标注（默认）
python deployment/yolo_annotate.py \
    --input data/images/ \
    --output-dir outputs/Annotations

# 生成标注 + 可视化
python deployment/yolo_annotate.py \
    --input data/images/ \
    --output-dir outputs/Annotations \
    --vis-dir outputs/Visualizations
```

### 模型选择

```bash
# 使用不同大小的 YOLOv8 模型
python deployment/yolo_annotate.py -i data/ -o outputs/ --model yolov8n.pt  # Nano - 最快
python deployment/yolo_annotate.py -i data/ -o outputs/ --model yolov8s.pt  # Small - 平衡
python deployment/yolo_annotate.py -i data/ -o outputs/ --model yolov8m.pt  # Medium - 更准
python deployment/yolo_annotate.py -i data/ -o outputs/ --model yolov8l.pt  # Large - 最准但慢
```

模型会自动下载到当前目录。

### 调整检测阈值

```bash
python deployment/yolo_annotate.py \
    --input data/images/ \
    --output-dir outputs/Annotations \
    --conf-threshold 0.5 \  # 只保留置信度 > 0.5 的检测结果
    --iou-threshold 0.45    # NMS IoU 阈值
```

### 指定运行设备

```bash
# 自动选择 (默认)
python deployment/yolo_annotate.py -i data/ -o outputs/ --device auto

# 强制 CPU
python deployment/yolo_annotate.py -i data/ -o outputs/ --device cpu

# 指定 GPU
python deployment/yolo_annotate.py -i data/ -o outputs/ --device cuda
python deployment/yolo_annotate.py -i data/ -o outputs/ --device 0
```

## 输出结构

```
outputs/
├── Annotations/           # VOC格式XML标注
│   ├── image001.xml
│   └── image002.xml
└── Visualizations/        # 可视化结果（仅当指定 --vis-dir 时生成）
    ├── image001.jpg
    └── image002.jpg
```

## 完整工作流示例

```bash
# 1. 视频拆帧
python deployment/video_to_frames.py \
    --input videos/ \
    --output frames/ \
    --fps 1

# 2. YOLO自动标注 (无需API密钥，默认不生成可视化)
python deployment/yolo_annotate.py \
    --input frames/ \
    --output-dir outputs/Annotations \
    --model yolov8n.pt \
    --conf-threshold 0.3

# 3. 整理图像
mkdir -p outputs/JPEGImages
cp frames/*.jpg outputs/JPEGImages/

# 4. 划分数据集
python deployment/split_dataset.py \
    --image-dir outputs/JPEGImages \
    --anno-dir outputs/Annotations \
    --output data/VOC_fall \
    --train-ratio 0.8 \
    --val-ratio 0.2

# 5. 训练分类器
bash scripts/shell/run_train_simple_classifier.sh \
    --config configs/training/simple_classifier_voc.yaml
```

## 对比 VLM 标注

| 特性 | YOLO 标注 | VLM 标注 |
|------|-----------|----------|
| 速度 | 快 (GPU实时) | 慢 (API调用) |
| 准确率 | 高 (专用检测) | 中 (通用理解) |
| 成本 | 免费 | 按量付费 |
| 离线使用 | 是 | 否 |
| 类别信息 | 仅检测人 | 可识别状态 |

**建议**: 
- 仅需人体位置框 → 用 YOLO
- 需要识别跌倒/站立等状态 → 用 VLM 或先用 YOLO 检测再分类

## 常见问题

### Q: 检测不到人？
A: 降低置信度阈值：`--conf-threshold 0.1`

### Q: 误检太多？
A: 提高置信度阈值：`--conf-threshold 0.5`

### Q: 需要检测其他类别？
修改代码中 `classes=[0]` 为其他 COCO 类别 ID。

### Q: 模型下载失败？
手动下载模型文件放到项目目录：
```bash
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
```
