# DDP训练Batch Size机制详解

## 问题现象

```
单卡训练：1383 batches
DDP 2卡训练：691 batches (≈1383/2)
```

**为什么DDP训练的迭代次数减半了？**

## DDP的Batch Size机制

### 核心概念

在PyTorch DDP训练中，有三种batch size概念：

1. **Per-GPU Batch Size**：每个GPU处理的样本数
2. **Global Batch Size**：所有GPU一次迭代处理的总样本数
3. **Effective Batch Size**：考虑梯度累积后的等效batch size

### 当前实现的机制

```python
# 配置文件中的batch_size
batch_size: 32  # 这是 per-GPU batch size

# DDP训练时
train_loader = DataLoader(
    train_dataset,
    batch_size=32,        # 每个GPU的batch_size
    sampler=DistributedSampler(
        train_dataset,
        num_replicas=world_size,  # GPU数量
        rank=rank,
    ),
)
```

### 实际执行流程

#### 单卡训练
```
数据集总样本数: N = 44,224
batch_size: 32
迭代次数: N / 32 = 1,382 batches
全局batch size: 32
```

#### DDP 2卡训练
```
数据集总样本数: N = 44,224
GPU 0: 分到 N/2 = 22,112 个样本
GPU 1: 分到 N/2 = 22,112 个样本
每个GPU的batch_size: 32
每个GPU的迭代次数: 22,112 / 32 = 691 batches

全局batch size: 32 × 2 = 64
等效于: 每次迭代处理 64 个样本（GPU0和GPU1各处理32个）
```

### 数学公式

```
单卡：iterations = N / batch_size
DDP：iterations = N / (batch_size × num_gpus)

全局batch_size = batch_size_per_gpu × num_gpus
```

## 为什么这样设计？

### 1. **数据并行，不是模型并行**

DDP是数据并行策略：
- 每个GPU上有一个**完整的模型副本**
- 数据被均匀分割到各个GPU上
- 每个GPU独立计算梯度，然后AllReduce同步

### 2. **加速训练，不是增加batch size**

DDP的目标是**加速训练**，而不是增加batch size：

```
单卡训练：
- 耗时: 1382 iterations × iteration_time
- 全局batch size: 32

DDP 2卡训练：
- 耗时: 691 iterations × iteration_time (理论上是单卡的1/2)
- 全局batch size: 64 (2倍于单卡)
- 吞吐量: 2倍于单卡（如果完美并行）
```

## 如何配置Batch Size？

### 方案1：固定Per-GPU Batch Size（推荐）

```yaml
# 配置文件
batch_size: 32  # 每个GPU的batch size

# 单卡训练
全局batch size = 32

# 2卡DDP训练
全局batch size = 32 × 2 = 64

# 4卡DDP训练
全局batch size = 32 × 4 = 128
```

**优点**：
- ✅ 配置简单，无需修改
- ✅ 显存占用固定
- ✅ 训练加速明显

**缺点**：
- ⚠️ 全局batch size随GPU数量线性增长
- ⚠️ 可能影响收敛（batch size太大）

### 方案2：固定Global Batch Size（推荐）

```python
# 动态调整per-GPU batch size
def get_batch_size(cfg, world_size):
    global_batch_size = cfg.get("global_batch_size", 64)
    per_gpu_batch_size = global_batch_size // world_size
    return max(1, per_gpu_batch_size)

# 使用
batch_size = get_batch_size(cfg, world_size)
```

```yaml
# 配置文件
global_batch_size: 64  # 固定全局batch size

# 单卡训练
per_gpu_batch_size = 64 / 1 = 64

# 2卡DDP训练
per_gpu_batch_size = 64 / 2 = 32

# 4卡DDP训练
per_gpu_batch_size = 64 / 4 = 16
```

**优点**：
- ✅ 全局batch size固定，不影响收敛
- ✅ 训练行为一致

**缺点**：
- ⚠️ GPU增多时，per-GPU batch size可能太小
- ⚠️ 显存利用率可能降低

### 方案3：混合策略（最佳实践）

```python
def get_batch_size(cfg, world_size):
    # 设置per-GPU batch size上限（显存限制）
    max_per_gpu = cfg.get("max_per_gpu_batch_size", 32)
    
    # 设置global batch size目标（收敛优化）
    target_global = cfg.get("target_global_batch_size", 64)
    
    # 计算per-GPU batch size
    per_gpu = min(max_per_gpu, target_global // world_size)
    
    return per_gpu

# 实际全局batch size
global_batch_size = per_gpu * world_size
```

```yaml
# 配置文件
max_per_gpu_batch_size: 32  # 每个GPU最多32（显存限制）
target_global_batch_size: 64  # 目标全局batch size（收敛优化）

# 单卡训练
per_gpu = min(32, 64/1) = 32
全局batch size = 32

# 2卡DDP训练
per_gpu = min(32, 64/2) = 32
全局batch size = 32 × 2 = 64

# 4卡DDP训练
per_gpu = min(32, 64/4) = 16
全局batch size = 16 × 4 = 64
```

## 学习率调整

### 线性缩放规则（Linear Scaling Rule）

当全局batch size增大时，学习率应该相应增大：

```
lr_new = lr_base × (global_batch_size / base_batch_size)
```

**示例**：
```python
# 基础配置
base_batch_size = 32
base_lr = 0.001

# DDP 2卡训练
global_batch_size = 32 × 2 = 64
lr_new = 0.001 × (64 / 32) = 0.002

# DDP 4卡训练
global_batch_size = 32 × 4 = 128
lr_new = 0.001 × (128 / 32) = 0.004
```

**注意**：
- ⚠️ 这是经验规则，可能需要微调
- ⚠️ Warmup很重要，避免训练初期不稳定

## 当前项目的问题与建议

### 当前问题

```python
# 当前代码（train_simple_classifier.py:629-643）
batch_size = cfg.get("batch_size", 64)  # per-GPU batch size

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,  # 每个GPU都是64
    sampler=DistributedSampler(...),
)
```

**问题**：
- ❌ 没有学习率缩放
- ❌ 全局batch size随GPU数量增长
- ❌ 可能影响收敛

### 建议的改进

```python
def create_data_loaders(train_dataset, val_dataset, cfg, world_size, rank, ddp):
    """Create training and validation data loaders with proper batch size scaling."""
    
    # 1. 获取per-GPU batch size
    batch_size = cfg.get("batch_size", 32)
    
    # 2. 计算全局batch size
    global_batch_size = batch_size * world_size if ddp else batch_size
    
    # 3. 学习率缩放
    base_lr = cfg.get("lr", 0.001)
    base_batch_size = cfg.get("base_batch_size", 32)  # 基准batch size
    scaled_lr = base_lr * (global_batch_size / base_batch_size)
    
    if rank == 0:
        print(f"Per-GPU batch size: {batch_size}")
        print(f"Global batch size: {global_batch_size}")
        print(f"Base LR: {base_lr} -> Scaled LR: {scaled_lr}")
    
    # 4. 创建DataLoader
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if ddp else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    
    return train_loader, val_loader, scaled_lr
```

### 配置文件建议

```yaml
# configs/training/simple_classifier_voc.yaml

# Batch size配置
batch_size: 32          # Per-GPU batch size
base_batch_size: 32     # 基准batch size（用于学习率缩放）

# 学习率配置
lr: 0.001               # 基准学习率
lr_scaling: true        # 启用学习率缩放
warmup_epochs: 5        # Warmup轮数

# 说明：
# - 单卡训练：global_batch_size=32, lr=0.001
# - 2卡DDP：global_batch_size=64, lr=0.002
# - 4卡DDP：global_batch_size=128, lr=0.004
```

## 常见误区

### ❌ 误区1：DDP训练batch size应该和单卡一样

**错误理解**：认为DDP只是为了加快训练，batch size不应该变。

**正确理解**：DDP通过数据并行加速训练，全局batch size确实变大了，这是正常的。如果需要保持相同的训练行为，应该调整学习率。

### ❌ 误区2：迭代次数减少说明训练更快

**错误理解**：看到迭代次数从1383减到691，以为训练变快了。

**正确理解**：
- 每次迭代处理的数据量增加了（64 vs 32）
- 总数据量不变
- 训练时长取决于吞吐量（samples/sec），不是迭代次数

### ❌ 误区3：DDP训练效果应该和单卡完全一样

**错误理解**：认为DDP训练的loss曲线应该和单卡完全一致。

**正确理解**：由于全局batch size变大，梯度噪声降低，训练曲线会有差异。这是正常的，可以通过调整学习率来优化。

## 性能对比示例

### 单卡训练
```
配置：batch_size=32, lr=0.001
数据集：44,224 samples
迭代次数：1,382 iterations
全局batch size：32
预计训练时间：T
吞吐量：S samples/sec
```

### DDP 2卡训练（当前实现）
```
配置：batch_size=32, lr=0.001 (未缩放)
数据集：44,224 samples
迭代次数：691 iterations
全局batch size：64
预计训练时间：≈ T/2 (理想情况)
吞吐量：≈ 2S samples/sec
实际加速比：1.5x - 1.9x (考虑通信开销)
```

### DDP 2卡训练（优化后）
```
配置：batch_size=32, lr=0.002 (线性缩放)
数据集：44,224 samples
迭代次数：691 iterations
全局batch size：64
warmup: 5 epochs
预计训练时间：≈ T/2
收敛行为：接近单卡训练
```

## 验证方法

### 检查实际batch size

```python
# 在训练脚本中添加
print(f"Rank {rank}: Dataset size = {len(train_dataset)}")
print(f"Rank {rank}: Samples per GPU = {len(train_sampler)}")
print(f"Rank {rank}: Iterations per epoch = {len(train_loader)}")
print(f"Rank {rank}: Per-GPU batch size = {train_loader.batch_size}")
print(f"Global batch size = {train_loader.batch_size * world_size}")
```

### 单卡训练输出
```
Dataset size = 44224
Iterations per epoch = 1382
Per-GPU batch size = 32
Global batch size = 32
```

### DDP 2卡训练输出
```
Rank 0: Dataset size = 44224
Rank 0: Samples per GPU = 22112
Rank 0: Iterations per epoch = 691
Rank 0: Per-GPU batch size = 32
Global batch size = 64
```

## 总结

### 当前现象解释

✅ **这是正常行为！**

- 单卡：1383 batches × 32 samples = 44,256 samples
- DDP 2卡：691 batches × 64 samples = 44,224 samples
- 数据量一致，只是分配方式不同

### 建议的改进方向

1. ✅ **添加学习率缩放**：`lr_new = lr × (global_batch_size / base_batch_size)`
2. ✅ **添加训练日志**：打印per-GPU和全局batch size
3. ✅ **添加warmup**：避免训练初期不稳定
4. ✅ **监控收敛曲线**：对比单卡和DDP的训练效果

### 快速验证

```bash
# 在训练脚本中添加日志输出
PYTHONPATH=src python scripts/train/train_simple_classifier.py \
  --config configs/training/simple_classifier_voc.yaml \
  --ngpus 2

# 查看输出
# 应该看到：
# Per-GPU batch size: 32
# Global batch size: 64
# Iterations per epoch: 691
```

---

**参考资料**：
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)
- [Large Batch Training of Convolutional Networks](https://arxiv.org/abs/1708.03888)