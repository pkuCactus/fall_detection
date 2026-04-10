# DDP训练Batch Size机制更新

## 重要变更

### 配置文件中的batch_size现在表示全局batch size

**之前的行为**（容易混淆）:
```yaml
batch_size: 32  # 这是per-GPU batch size

# 单卡训练
实际batch size = 32

# DDP 2卡训练
每个GPU batch size = 32
全局batch size = 32 × 2 = 64  # 自动变成2倍
```

**现在的行为**（更直观）:
```yaml
batch_size: 64  # 这是全局batch size（所有GPU的总和）

# 单卡训练
per_gpu_batch_size = 64 / 1 = 64

# DDP 2卡训练
per_gpu_batch_size = 64 / 2 = 32
全局batch size = 64  # 保持不变
```

## 代码修改详情

### 1. create_data_loaders 函数

```python
def create_data_loaders(train_dataset, val_dataset, cfg, world_size, rank, ddp):
    """
    Args:
        cfg['batch_size']: 全局batch size（所有GPU的总和）
    """
    global_batch_size = cfg.get("batch_size", 64)
    
    # 自动计算per-GPU batch size
    if ddp:
        if global_batch_size % world_size != 0:
            raise ValueError(
                f"Global batch_size ({global_batch_size}) must be divisible by "
                f"number of GPUs ({world_size})"
            )
        per_gpu_batch_size = global_batch_size // world_size
    else:
        per_gpu_batch_size = global_batch_size
    
    # 详细日志输出
    if rank == 0:
        print(f"Global batch size: {global_batch_size}")
        print(f"Per-GPU batch size: {per_gpu_batch_size}")
        if ddp:
            print(f"Number of GPUs: {world_size}")
```

### 2. 配置文件更新

```yaml
# configs/training/simple_classifier_voc.yaml

# 训练配置
epochs: 100
batch_size: 64  # 全局batch size（所有GPU的总和）
                  # DDP训练时，每个GPU的batch_size = batch_size / num_gpus
                  # 例如：batch_size=64，2卡DDP时每个GPU实际使用32
```

## 使用示例

### 单卡训练

```bash
bash scripts/shell/run_train_simple_classifier.sh \
  --config configs/training/simple_classifier_voc.yaml

# 输出:
# Global batch size: 64
# Per-GPU batch size: 64
# Iterations per epoch: 691 (假设数据集44,224样本)
```

### DDP 2卡训练

```bash
bash scripts/shell/run_train_simple_classifier.sh \
  --config configs/training/simple_classifier_voc.yaml \
  --ngpus 2

# 输出:
# Global batch size: 64
# Per-GPU batch size: 32
# Number of GPUs: 2
# Iterations per epoch: 691 (和单卡一样！)
```

### DDP 4卡训练

```bash
bash scripts/shell/run_train_simple_classifier.sh \
  --config configs/training/simple_classifier_voc.yaml \
  --ngpus 4

# 输出:
# Global batch size: 64
# Per-GPU batch size: 16
# Number of GPUs: 4
# Iterations per epoch: 691 (保持不变)
```

## 优势

### ✅ 优点

1. **配置直观**: 用户配置的就是实际使用的batch size
2. **训练行为一致**: 无论使用多少GPU，全局batch size保持不变
3. **学习率不用调整**: 因为batch size没变，学习率也无需调整
4. **收敛性一致**: 训练曲线在不同GPU数量下保持一致

### ⚠️ 注意事项

1. **batch_size必须能被GPU数量整除**
   ```python
   # 正确
   batch_size: 64, num_gpus: 2  # 64 % 2 = 0 ✓
   batch_size: 64, num_gpus: 4  # 64 % 4 = 0 ✓
   
   # 错误（会报错）
   batch_size: 64, num_gpus: 3  # 64 % 3 ≠ 0 ✗
   ```

2. **显存限制**: GPU数量增多时，per-GPU batch size减小
   ```yaml
   # 如果显存不够，可以增大batch_size
   batch_size: 128  # 4卡DDP时，每GPU用32
   ```

## 错误处理

### 检查batch_size是否能被GPU数量整除

```python
if ddp:
    if global_batch_size % world_size != 0:
        raise ValueError(
            f"Global batch_size ({global_batch_size}) must be divisible by "
            f"number of GPUs ({world_size})"
        )
```

**错误示例**:
```bash
# 配置: batch_size=64
# 运行: --ngpus 3

# 输出:
# ValueError: Global batch_size (64) must be divisible by number of GPUs (3)
```

**解决方案**:
```yaml
# 方案1: 调整batch_size
batch_size: 60  # 60 % 3 = 0 ✓

# 方案2: 使用能整除的GPU数量
# --ngpus 2 或 --ngpus 4
```

## 迁移指南

### 之前使用per-GPU batch size的配置

```yaml
# 旧配置（per-GPU）
batch_size: 32  # 每个GPU用32

# 单卡: 全局batch = 32
# 2卡DDP: 全局batch = 64
```

### 迁移到新的全局batch size

```yaml
# 新配置（全局）
# 如果希望和之前的行为一致:

# 单卡训练
batch_size: 32  # 保持不变

# 2卡DDP训练
batch_size: 64  # 改为 32 × 2 = 64

# 4卡DDP训练  
batch_size: 128  # 改为 32 × 4 = 128
```

### 推荐配置

```yaml
# 根据任务和数据集选择合适的全局batch size
batch_size: 64  # 推荐值（适用于大多数情况）

# 如果显存充足，可以增大
batch_size: 128  # 更大的batch size，训练更稳定

# 如果显存不足，可以减小
batch_size: 32   # 更小的batch size，节省显存
```

## 验证方法

### 检查实际使用的batch size

训练开始时会输出：

```
============================================================
Batch Size Configuration:
  Global batch size: 64
  Per-GPU batch size: 32
  Number of GPUs: 2
============================================================
```

### 检查迭代次数

```python
# 所有GPU配置下的迭代次数应该相同
iterations = dataset_size / global_batch_size

# 例如：数据集44,224样本，global_batch_size=64
iterations = 44224 / 64 = 691  # 无论使用1卡、2卡还是4卡
```

## 性能对比

### 单卡训练

```
配置: batch_size=64, lr=0.001
迭代次数: 691
全局batch size: 64
训练时间: T
```

### DDP 2卡训练（新实现）

```
配置: batch_size=64, lr=0.001
迭代次数: 691 (和单卡一样)
全局batch size: 64 (和单卡一样)
per_gpu_batch_size: 32
训练时间: ≈ T/2 (理想加速比)
收敛行为: 和单卡一致
```

### DDP 2卡训练（旧实现）

```
配置: batch_size=32(per-GPU), lr=0.001
迭代次数: 691
全局batch size: 64 (是单卡的2倍)
per_gpu_batch_size: 32
训练时间: ≈ T/2
收敛行为: 可能不同（batch size变大）
```

## 总结

| 方面 | 旧实现 | 新实现 |
|------|--------|--------|
| 配置含义 | per-GPU batch size | 全局batch size |
| 单卡配置 | batch_size=32 | batch_size=64 |
| 2卡DDP配置 | batch_size=32 → 全局64 | batch_size=64 → 全局64 |
| 迭代次数 | 随GPU数量变化 | 保持不变 |
| 学习率调整 | 需要手动缩放 | 无需调整 |
| 收敛一致性 | 可能不一致 | 保持一致 |

**推荐**: 使用新的全局batch size配置方式，更直观且训练行为更一致。