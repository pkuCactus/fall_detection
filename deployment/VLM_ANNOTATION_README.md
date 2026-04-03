# VLM图像标注工具

使用视觉语言模型(VLM)自动分析图像，检测人物状态(跌倒/站立/坐下等)和位置，并保存为Pascal VOC格式。

## 功能

- **VLM检测**: 调用Claude/GPT-4V等模型分析图像
- **状态识别**: 识别人物状态 (fall, fallen, falling, stand, sit, squat, bend, kneel, crawl, half_up)
- **位置检测**: 输出边界框坐标 [x1, y1, x2, y2]
- **VOC格式保存**: 生成标准的Pascal VOC XML标注文件
- **可视化**: 绘制检测框和标签，直观查看结果
- **数据集划分**: 自动划分为train/val/test集

## 安装依赖

```bash
pip install anthropic openai pillow requests opencv-python numpy

# 或安装项目全部依赖
pip install -r requirements.txt
```

## 使用步骤

### 1. 准备API密钥

```bash
# 使用Claude (推荐)
export ANTHROPIC_API_KEY="your-api-key"

# 或使用 GPT-4V
export OPENAI_API_KEY="your-api-key"
```

### 2. 使用VLM标注图像

#### 单张图像
```bash
python deployment/vlm_annotate.py \
    --input data/my_image.jpg \
    --output-dir outputs/Annotations \
    --vis-dir outputs/Visualizations \
    --model claude
```

#### 批量处理目录
```bash
python deployment/vlm_annotate.py \
    --input data/images/ \
    --output-dir outputs/Annotations \
    --vis-dir outputs/Visualizations \
    --model claude
```

**输出结构**:
```
outputs/
├── Annotations/           # VOC格式XML标注
│   ├── image001.xml
│   └── image002.xml
└── Visualizations/        # 可视化结果
    ├── image001.jpg
    └── image002.jpg
```

### 3. 划分数据集

将标注好的数据划分为 train/val/test:

```bash
python deployment/split_dataset.py \
    --image-dir outputs/JPEGImages \
    --anno-dir outputs/Annotations \
    --output data/VOC_fall \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15
```

**输出VOC标准结构**:
```
data/VOC_fall/
├── JPEGImages/              # 图像文件
├── Annotations/             # XML标注
└── ImageSets/
    └── Main/
        ├── train.txt        # 训练集图像列表
        ├── val.txt          # 验证集图像列表
        └── test.txt         # 测试集图像列表
```

### 4. 训练分类器

使用生成的VOC格式数据训练:

```bash
# 修改配置文件
vim configs/training/simple_classifier_voc.yaml

# 设置数据路径
voc:
  train_dirs:
    - "data/VOC_fall"
  val_dirs:
    - "data/VOC_fall"

# 运行训练
bash scripts/shell/run_train_simple_classifier.sh \
    --config configs/training/simple_classifier_voc.yaml
```

## 完整工作流示例

```bash
# 1. 创建输出目录
mkdir -p outputs/raw_images

# 2. 放入待标注图像 (手动拷贝或使用已有图像)
cp /path/to/your/images/*.jpg outputs/raw_images/

# 3. VLM自动标注
export ANTHROPIC_API_KEY="sk-xxx"
python deployment/vlm_annotate.py \
    -i outputs/raw_images \
    -o outputs/vlm_annotations/Annotations \
    -v outputs/vlm_annotations/JPEGImages \
    --model claude

# 4. 整理图像到VOC结构
mkdir -p outputs/vlm_annotations/JPEGImages
cp outputs/raw_images/*.jpg outputs/vlm_annotations/JPEGImages/

# 5. 划分数据集
python deployment/split_dataset.py \
    -i outputs/vlm_annotations/JPEGImages \
    -a outputs/vlm_annotations/Annotations \
    -o data/VLM_fall \
    --train-ratio 0.8 \
    --val-ratio 0.2

# 6. 修改训练配置
sed -i 's|train_dirs:.*|train_dirs:\n    - "data/VLM_fall"|' \
    configs/training/simple_classifier_voc.yaml

# 7. 开始训练
bash scripts/shell/run_train_simple_classifier.sh \
    --config configs/training/simple_classifier_voc.yaml
```

## 标注类别说明

VLM会检测以下人物状态:

### 跌倒/异常类别 (label=1)

| 类别 | 说明 | 颜色 |
|------|------|------|
| fall_down | 跌倒/摔倒 | 红色 |
| kneel | 跪下 | 红色 |
| half_up | 半起身 | 红色 |
| crawl | 爬行 | 红色 |

### 正常类别 (label=0)

| 类别 | 说明 | 颜色 |
|------|------|------|
| stand | 站立 | 绿色 |
| sit | 坐下 | 绿色 |
| squat | 蹲下 | 绿色 |
| bend | 弯腰 | 绿色 |

## 配置文件示例

**simple_classifier_voc.yaml**:
```yaml
data:
  format: "voc"

voc:
  train_dirs:
    - "data/VLM_fall"
  val_dirs:
    - "data/VLM_fall"
  
  fall_classes:
    - "fall_down"
    - "kneel"
    - "half_up"
    - "crawl"
  
  normal_classes:
    - "stand"
    - "sit"
    - "squat"
    - "bend"

# 其他配置...
```

## 常见问题

### Q: API调用费用?
A: Claude 3.5 Sonnet 约 $0.003/图像 (取决于图像大小)。批量处理前建议小样本测试。

### Q: 标注准确率低怎么办?
A: 
1. 使用更精确的prompt（修改 `VLMDetector._call_claude` 中的 system_prompt）
2. 人工校验后筛选高质量样本
3. 混合真实标注+VLM标注训练

### Q: 如何处理大图像?
A: 工具会自动处理，但API费用会增加。建议先resize到1920x1080以内。

### Q: 标注结果如何人工修正?
A: 使用 [LabelImg](https://github.com/tzutalin/labelImg) 或 [CVAT](https://cvat.org/) 打开生成的XML进行修正。

## 高级用法

### 自定义VLM提示词

编辑 `vlm_annotate.py` 中的 `system_prompt`:

```python
system_prompt = """You are a security camera analysis expert.

Detect all persons and classify their activity:
- normal: standing, walking, sitting normally
- warning: bending, squatting (might be preparing to fall)
- danger: falling, fallen, lying on ground

Return JSON with bounding boxes.
"""
```

### 批量处理时过滤低置信度

修改代码，只保留 confidence > 0.8 的检测结果。

### 与其他数据源合并

```bash
# VLM标注 + 人工标注合并
cp outputs/vlm_annotations/Annotations/*.xml data/combined/Annotations/
cp manual_annotations/*.xml data/combined/Annotations/

# 重新划分
python deployment/split_dataset.py -i ... -a data/combined/Annotations -o data/combined
```
