"""Learning rate schedulers with warmup support."""

from typing import Optional


class WarmupScheduler:
    """支持 warmup 的学习率调度器包装器.

    warmup 基于总 batch step 计算，每个 step 更新学习率。
    支持两种 warmup 策略:
    - linear: 从 warmup_init_lr 线性增加到初始学习率
    - constant: 保持 warmup_init_lr 不变，warmup 结束后跳到初始学习率
    """

    def __init__(self, optimizer, scheduler, warmup_steps: int = 0,
                 warmup_strategy: str = 'linear', warmup_init_lr: float = 1e-5):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.warmup_strategy = warmup_strategy
        self.warmup_init_lr = warmup_init_lr
        self.current_step = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.warmup_finished = False

    def step(self, metrics=None):
        """Step the scheduler (每个 epoch 结束时调用)."""
        # Warmup 在 step_batch 中处理，这里只处理主 scheduler
        if self.warmup_finished and self.scheduler is not None:
            if metrics is not None and hasattr(self.scheduler, 'step'):
                self.scheduler.step(metrics)
            elif hasattr(self.scheduler, 'step'):
                self.scheduler.step()

    def step_batch(self):
        """每个 batch/step 调用，用于 warmup."""
        if self.current_step < self.warmup_steps:
            # Warmup 阶段
            if self.warmup_strategy == 'linear':
                # 线性增加学习率: lr = init_lr + (base_lr - init_lr) * (step / total_steps)
                alpha = self.current_step / self.warmup_steps
                for i, group in enumerate(self.optimizer.param_groups):
                    group['lr'] = self.warmup_init_lr + alpha * (self.base_lrs[i] - self.warmup_init_lr)
            elif self.warmup_strategy == 'constant':
                # 保持 warmup_init_lr 不变
                for group in self.optimizer.param_groups:
                    group['lr'] = self.warmup_init_lr
            self.current_step += 1
        elif not self.warmup_finished:
            # Warmup 刚结束，设置为基础学习率
            for i, group in enumerate(self.optimizer.param_groups):
                group['lr'] = self.base_lrs[i]
            self.warmup_finished = True
            self.current_step += 1
        else:
            self.current_step += 1

    def get_last_lr(self):
        """获取当前学习率."""
        return [group['lr'] for group in self.optimizer.param_groups]

    @property
    def is_warmup(self):
        """是否还在 warmup 阶段."""
        return self.current_step < self.warmup_steps


