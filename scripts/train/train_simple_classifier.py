"""Train simple image classifier for fall detection.

Refactored version with clean separation of concerns.
"""

import os
import time
import traceback
from datetime import datetime
from typing import Optional, Tuple, Dict, Any


def _timestamp() -> str:
    """获取当前时间字符串."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from fall_detection.data import CocoFallDataset, VOCFallDataset, TrainingAugmentation
from fall_detection.models.simple_classifier import SimpleFallClassifier
from fall_detection.utils import (
    WarmupScheduler,
    format_time_remaining,
    load_config,
    parse_args,
    setup_ddp,
    setup_seed,
)


def _get_dataset_config(cfg: Dict[str, Any]) -> Tuple[Dict, Dict, Any, int]:
    """Extract common dataset configuration."""
    input_cfg = cfg.get("input", {})
    crop_cfg = cfg.get("crop", {})
    aug_cfg = cfg.get("data_augmentation", {})
    train_transform = (
        None if not aug_cfg.get("enabled", True) else TrainingAugmentation(aug_cfg)
    )
    cache_cfg = cfg.get("cache", {})
    cache_size = cache_cfg.get("image_cache_size", 1000)
    return input_cfg, crop_cfg, train_transform, cache_size


def _create_voc_datasets(cfg: Dict[str, Any], rank: int) -> Tuple:
    """Create VOC format datasets."""
    voc_cfg = cfg.get("voc", {})
    input_cfg, crop_cfg, train_transform, cache_size = _get_dataset_config(cfg)
    cache_dir = cfg.get("cache", {}).get("dir", "outputs/cache")

    train_dirs = voc_cfg.get("train_dirs", voc_cfg.get("data_dir"))
    if train_dirs is None:
        raise ValueError(
            "voc.train_dirs or voc.data_dir must be specified for VOC format"
        )
    if isinstance(train_dirs, str):
        train_dirs = [train_dirs]

    val_dirs = voc_cfg.get("val_dirs")
    if isinstance(val_dirs, str):
        val_dirs = [val_dirs]

    fall_classes = voc_cfg.get("fall_classes", ["fall"])
    normal_classes = voc_cfg.get("normal_classes")

    if rank == 0:
        print(f"Loading training data from VOC directories: {train_dirs}")
        print(f"  Fall classes: {fall_classes}")
        print(f"  Normal classes: {normal_classes or '<all others>'}")

    common_args = {
        "target_size": input_cfg.get("target_size", 96),
        "use_letterbox": input_cfg.get("use_letterbox", True),
        "fill_value": input_cfg.get("fill_value", 114),
        "fall_classes": fall_classes,
        "normal_classes": normal_classes,
        "shrink_max": crop_cfg.get("shrink_max", 3),
        "expand_max": crop_cfg.get("expand_max", 25),
        "cache_dir": cache_dir,
        "cache_size": cache_size,
    }

    train_dataset = VOCFallDataset(
        data_dirs=train_dirs, split="train", transform=train_transform, **common_args
    )

    val_dataset = None
    if val_dirs:
        if rank == 0:
            print(f"Loading validation data from VOC directories: {val_dirs}")
        val_dataset = VOCFallDataset(
            data_dirs=val_dirs, split="val", transform=None, **common_args
        )

    return train_dataset, val_dataset


def _create_coco_datasets(cfg: Dict[str, Any], rank: int) -> Tuple:
    """Create COCO format datasets."""
    data_cfg = cfg.get("data", {})
    input_cfg, crop_cfg, train_transform, cache_size = _get_dataset_config(cfg)

    train_coco_json = data_cfg.get("train_coco_json")
    val_coco_json = data_cfg.get("val_coco_json")
    image_dir = data_cfg.get("image_dir")

    if not train_coco_json or not image_dir:
        raise ValueError(
            "train_coco_json and image_dir must be specified for COCO format"
        )

    if rank == 0:
        print(f"Loading training data from: {train_coco_json}")

    common_args = {
        "image_dir": image_dir,
        "target_size": input_cfg.get("target_size", 96),
        "use_letterbox": input_cfg.get("use_letterbox", True),
        "fill_value": input_cfg.get("fill_value", 114),
        "person_category_id": cfg.get("coco", {}).get("person_category_id", 1),
        "fall_category_id": cfg.get("coco", {}).get("fall_category_id", 1),
        "shrink_max": crop_cfg.get("shrink_max", 3),
        "expand_max": crop_cfg.get("expand_max", 25),
        "cache_size": cache_size,
    }

    train_dataset = CocoFallDataset(
        coco_json=train_coco_json, transform=train_transform, **common_args
    )

    val_dataset = None
    if val_coco_json and os.path.exists(val_coco_json):
        if rank == 0:
            print(f"Loading validation data from: {val_coco_json}")
        val_dataset = CocoFallDataset(
            coco_json=val_coco_json, transform=None, **common_args
        )

    return train_dataset, val_dataset


def _print_dataset_stats(dataset, name: str, rank: int):
    """Print dataset statistics."""
    if rank != 0:
        return
    print(f"{name} samples: {len(dataset)}")
    labels = [s[2] for s in dataset.samples]
    n_fall = sum(labels)
    n_normal = len(labels) - n_fall
    print(f"  Fall: {n_fall}, Normal: {n_normal}")


def create_datasets(cfg: Dict[str, Any], rank: int):
    """Create training and validation datasets."""
    data_format = cfg.get("data", {}).get("format", "coco").lower()

    if data_format == "voc":
        train_dataset, val_dataset = _create_voc_datasets(cfg, rank)
    else:
        train_dataset, val_dataset = _create_coco_datasets(cfg, rank)

    _print_dataset_stats(train_dataset, "Train", rank)
    if val_dataset:
        _print_dataset_stats(val_dataset, "Val", rank)

    return train_dataset, val_dataset


def create_model(
    cfg: Dict[str, Any], device: torch.device, ddp: bool, local_rank: int, rank: int = 0
) -> nn.Module:
    """Create and setup model."""
    model_cfg = cfg.get("model", {})
    dropout = model_cfg.get("dropout", 0.3)
    fall_class_idx = model_cfg.get("fall_class_idx", 1)
    model = SimpleFallClassifier(dropout=dropout, fall_class_idx=fall_class_idx).to(device)

    # Apply torch.compile for PyTorch 2.0+ (before DDP wrapping)
    compile_cfg = cfg.get("compile", {})
    compile_enabled = compile_cfg.get("enabled", False) and not ddp

    if compile_enabled and hasattr(torch, "compile"):
        compile_mode = compile_cfg.get("mode", "default")
        if rank == 0:
            print(f"[{_timestamp()}] Compiling model with torch.compile (mode={compile_mode})...")
        try:
            model = torch.compile(model, mode=compile_mode)
            if rank == 0:
                print(f"[{_timestamp()}] Model compiled successfully")
        except Exception as e:
            if rank == 0:
                print(f"[{_timestamp()}] Warning: torch.compile failed ({e}), using uncompiled model")

    if ddp:
        ddp_cfg = cfg.get("ddp", {})
        find_unused = ddp_cfg.get("find_unused_parameters", False)
        # gradient_as_bucket_view=True helps avoid "Grad strides do not match bucket view strides" warning
        # and can improve DDP performance by reducing memory copies
        gradient_as_bucket_view = ddp_cfg.get(
            "gradient_as_bucket_view", True
        )  # Changed default to True
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )
        if rank == 0:
            print(
                f"[{_timestamp()}] DDP enabled (find_unused={find_unused}, gradient_as_bucket_view={gradient_as_bucket_view})"
            )

    return model


def _create_base_scheduler(optimizer, cfg: Dict[str, Any], train_loader: DataLoader):
    """Create base learning rate scheduler."""
    lr_cfg = cfg.get("lr_scheduler", {})
    scheduler_type = lr_cfg.get("type", "plateau")

    if scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=lr_cfg.get("factor", 0.5),
            patience=lr_cfg.get("patience", 10),
            min_lr=lr_cfg.get("min_lr", 1e-5),
        )
    elif scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=lr_cfg.get("T_max", cfg.get("epochs", 100)),
            eta_min=lr_cfg.get("min_lr", 1e-5),
        )
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=lr_cfg.get("step_size", 30),
            gamma=lr_cfg.get("gamma", 0.1),
        )
    return None


def _apply_warmup(
    scheduler, optimizer, cfg: Dict[str, Any], train_loader: DataLoader, rank: int
):
    """Wrap scheduler with warmup if enabled."""
    lr_cfg = cfg.get("lr_scheduler", {})
    warmup_cfg = lr_cfg.get("warmup", {})

    if not warmup_cfg.get("enabled", False):
        return scheduler

    warmup_epochs = warmup_cfg.get("epochs", 5)
    warmup_steps = warmup_epochs * len(train_loader)

    if rank == 0:
        print(
            f"Warmup enabled: {warmup_epochs} epochs = {warmup_steps} steps, "
            f"strategy={warmup_cfg.get('strategy', 'linear')}, "
            f"init_lr={warmup_cfg.get('init_lr', 1e-5)}"
        )

    return WarmupScheduler(
        optimizer,
        scheduler,
        warmup_steps=warmup_steps,
        warmup_strategy=warmup_cfg.get("strategy", "linear"),
        warmup_init_lr=warmup_cfg.get("init_lr", 1e-5),
    )


def create_optimizer_scheduler(
    cfg: Dict[str, Any], model: nn.Module, train_loader: DataLoader, rank: int
):
    """Create optimizer and learning rate scheduler."""
    lr = cfg.get("lr", 0.001)
    weight_decay = cfg.get("weight_decay", 0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = _create_base_scheduler(optimizer, cfg, train_loader)
    scheduler = _apply_warmup(scheduler, optimizer, cfg, train_loader, rank)

    return optimizer, scheduler


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int = 0,
    rank: int = 0,
    log_interval: int = 50,
    scheduler: Optional[WarmupScheduler] = None,
) -> Tuple[float, int, int]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    num_batches = len(loader)

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Update warmup scheduler
        if scheduler is not None and isinstance(scheduler, WarmupScheduler):
            scheduler.step_batch()

        _, predicted = torch.max(outputs, 1)
        batch_correct = (predicted == labels).sum().item()
        batch_size = labels.size(0)

        total_correct += batch_correct
        total_samples += batch_size
        total_loss += loss.item() * batch_size

        # Log progress
        if rank == 0 and log_interval > 0 and (batch_idx + 1) % log_interval == 0:
            batch_acc = batch_correct / batch_size
            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"[{_timestamp()}] Epoch[{epoch}] Batch[{batch_idx + 1}/{num_batches}]  "
                f"loss={loss.item():.4f} acc={batch_acc:.4f}  "
                f"avg_loss={avg_loss:.4f} avg_acc={avg_acc:.4f}  lr={current_lr:.6f}",
                flush=True,
            )

    return total_loss, total_correct, total_samples


@torch.no_grad()
def eval_epoch(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, int, int]:
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        _, predicted = torch.max(outputs, 1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    return total_loss, total_correct, total_samples


def _aggregate_metrics(
    t_loss: float,
    t_correct: int,
    t_samples: int,
    v_loss: float,
    v_correct: int,
    v_samples: int,
    device: torch.device,
    ddp: bool,
) -> tuple:
    """Aggregate metrics across DDP processes."""
    if not ddp:
        return t_loss, t_correct, t_samples, v_loss, v_correct, v_samples

    metrics = torch.tensor(
        [t_loss, t_correct, t_samples, v_loss, v_correct, v_samples],
        device=device,
        dtype=torch.float64,
    )
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    return tuple(metrics.tolist())


def _update_scheduler_step(scheduler, scheduler_type: str, val_loader, v_acc: float):
    """Update learning rate scheduler for one step."""
    if scheduler is None:
        return

    if isinstance(scheduler, WarmupScheduler):
        if val_loader and scheduler.scheduler is not None:
            scheduler.step(v_acc)
        else:
            scheduler.step()
    elif scheduler_type == "plateau" and val_loader:
        scheduler.step(v_acc)
    else:
        scheduler.step()


def _format_epoch_log(
    epoch: int,
    epochs: int,
    t_loss: float,
    t_acc: float,
    v_loss: float,
    v_acc: float,
    lr: float,
    remaining: str,
    val_loader,
) -> str:
    """Format epoch log message."""
    msg = f"[{_timestamp()}] Epoch {epoch}/{epochs}  train_loss={t_loss:.4f} train_acc={t_acc:.4f}"
    if val_loader:
        msg += f"  val_loss={v_loss:.4f} val_acc={v_acc:.4f}"
    msg += f"  lr={lr:.6f}  remain={remaining}"
    return msg


def _save_model_checkpoint(model: nn.Module, save_path: str, ddp: bool):
    """Save model checkpoint."""
    state = model.module.state_dict() if ddp else model.state_dict()
    torch.save(state, save_path)


def _handle_early_stopping(
    v_acc: float,
    best_acc: float,
    patience_counter: int,
    cfg: Dict[str, Any],
    model: nn.Module,
    ddp: bool,
    epoch: int,
    epoch_log_interval: int,
    rank: int = 0,
) -> tuple:
    """Handle early stopping and model saving.

    Returns:
        (new_best_acc, new_patience_counter, should_stop)
    """
    early_cfg = cfg.get("early_stopping", {})
    output_cfg = cfg.get("output", {})
    min_delta = early_cfg.get("min_delta", 0.001)
    patience = early_cfg.get("patience", 20)

    if v_acc > best_acc + min_delta:
        save_path = os.path.join(
            output_cfg.get("dir", "outputs/simple_classifier"), "best.pt"
        )
        if rank == 0:
            _save_model_checkpoint(model, save_path, ddp)
            if epoch % epoch_log_interval == 0:
                print(f"[{_timestamp()}] Saved best model (val_acc={v_acc:.4f})")
        return v_acc, 0, False
    patience_counter += 1
    if patience_counter >= patience:
        print(f"[{_timestamp()}] Early stopping at epoch {epoch}")
        return best_acc, patience_counter, True
    return best_acc, patience_counter, False


def _handle_no_validation_save(
    cfg: Dict[str, Any], model: nn.Module, ddp: bool, epoch: int, rank: int = 0
):
    """Save checkpoint when no validation is used."""
    if rank or epoch % cfg.get("log", {}).get("epoch_log_interval", 1) != 0:
        return
    output_cfg = cfg.get("output", {})
    save_path = os.path.join(
        output_cfg.get("dir", "outputs/simple_classifier"), f"epoch_{epoch}.pt"
    )
    _save_model_checkpoint(model, save_path, ddp)
    print(f"[{_timestamp()}] Saved checkpoint (epoch={epoch})")


def train_loop(
    cfg: Dict[str, Any],
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[WarmupScheduler],
    device: torch.device,
    ddp: bool,
    rank: int,
) -> float:
    """Main training loop."""
    label_smooth = cfg.get("label_smooth", 0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    if rank == 0 and label_smooth > 0:
        print(f"[{_timestamp()}] Label smoothing enabled: {label_smooth}")
    epochs = cfg.get("epochs", 100)
    log_cfg = cfg.get("log", {})
    lr_cfg = cfg.get("lr_scheduler", {})
    scheduler_type = lr_cfg.get("type", "plateau")

    best_acc = 0.0
    patience_counter = 0
    epoch_times = []
    epoch_log_interval = log_cfg.get("epoch_log_interval", 1)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        # Train
        t_loss, t_correct, t_samples = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            epoch=epoch,
            rank=rank,
            log_interval=log_cfg.get("batch_log_interval", 50),
            scheduler=scheduler,
        )

        # Validate
        if val_loader:
            v_loss, v_correct, v_samples = eval_epoch(
                model, val_loader, criterion, device
            )
        else:
            v_loss, v_correct, v_samples = 0.0, 0, 0

        # Aggregate and calculate metrics
        metrics = _aggregate_metrics(
            t_loss, t_correct, t_samples, v_loss, v_correct, v_samples, device, ddp
        )
        t_loss, t_correct, t_samples, v_loss, v_correct, v_samples = metrics

        t_loss_avg = t_loss / max(1, t_samples)
        t_acc = t_correct / max(1, t_samples)
        v_loss_avg = v_loss / max(1, v_samples) if val_loader else 0.0
        v_acc = v_correct / max(1, v_samples) if val_loader else t_acc

        # Time tracking
        epoch_duration = time.time() - epoch_start
        epoch_times.append(epoch_duration)
        avg_epoch_time = sum(epoch_times[-10:]) / min(len(epoch_times), 10)
        remaining_str = format_time_remaining(int(avg_epoch_time * (epochs - epoch)))

        # Update scheduler (all ranks)
        _update_scheduler_step(scheduler, scheduler_type, val_loader, v_acc)

        if rank == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            if epoch % epoch_log_interval == 0 or epoch == epochs:
                print(
                    _format_epoch_log(
                        epoch,
                        epochs,
                        t_loss_avg,
                        t_acc,
                        v_loss_avg,
                        v_acc,
                        current_lr,
                        remaining_str,
                        val_loader,
                    ),
                    flush=True,
                )

        if val_loader:
            best_acc, patience_counter, should_stop = _handle_early_stopping(
                v_acc,
                best_acc,
                patience_counter,
                cfg,
                model,
                ddp,
                epoch,
                epoch_log_interval,
                rank,
            )
            if should_stop:
                break
        else:
            _handle_no_validation_save(cfg, model, ddp, epoch, rank)

    return best_acc if val_loader else t_acc


def create_data_loaders(
    train_dataset,
    val_dataset,
    cfg: Dict[str, Any],
    world_size: int,
    rank: int,
    ddp: bool,
):
    """Create training and validation data loaders.

    Note: The batch_size in config represents GLOBAL batch size across all GPUs.
    For DDP training, per-GPU batch size is automatically calculated as:
        per_gpu_batch_size = global_batch_size / world_size
    """
    global_batch_size = cfg.get("batch_size", 64)
    num_workers = cfg.get("num_workers", 4)
    prefetch_factor = cfg.get("prefetch_factor", 4) if num_workers > 0 else None

    # Calculate per-GPU batch size
    if ddp:
        if global_batch_size % world_size != 0:
            raise ValueError(
                f"Global batch_size ({global_batch_size}) must be divisible by "
                f"number of GPUs ({world_size})"
            )
        per_gpu_batch_size = global_batch_size // world_size
    else:
        per_gpu_batch_size = global_batch_size

    # Log batch size configuration
    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"Batch Size Configuration:")
        print(f"  Global batch size: {global_batch_size}")
        print(f"  Per-GPU batch size: {per_gpu_batch_size}")
        if ddp:
            print(f"  Number of GPUs: {world_size}")
        print(f"{'=' * 60}\n")

    train_sampler = (
        DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        if ddp
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_gpu_batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = None
    if val_dataset:
        val_sampler = (
            DistributedSampler(
                val_dataset, num_replicas=world_size, rank=rank, shuffle=False
            )
            if ddp
            else None
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=per_gpu_batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    return train_loader, val_loader


def _test_sample_loading(dataset, rank: int):
    """Test loading a single sample."""
    if rank != 0:
        return
    print("  Testing dataset __getitem__...")
    try:
        sample_img, sample_label = dataset[0]
        print(f"  Sample loaded: shape={sample_img.shape}, label={sample_label}")
    except Exception as e:
        print(f"  ERROR: {e}")
        traceback.print_exc()
        raise


def _test_first_batch(loader, rank: int):
    """Test loading first batch."""
    if rank != 0:
        return
    print("Testing first batch loading...")
    try:
        first_batch = next(iter(loader))
        print(
            f"First batch loaded: images={first_batch[0].shape}, labels={first_batch[1].shape}"
        )
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        raise


def _setup_output_dir(cfg: Dict[str, Any], rank: int, ddp: bool, local_rank: int):
    """Setup output directory and save config."""
    if rank == 0:
        output_dir = cfg.get("output", {}).get("dir", "outputs/simple_classifier")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config_used.yaml"), "w") as f:
            yaml.dump(cfg, f)

    if ddp:
        dist.barrier(device_ids=[local_rank])


def main():
    """Main entry point."""
    args = parse_args(
        "Train simple image classifier from COCO or Pascal VOC annotations"
    )
    cfg = load_config(args)

    # Setup DDP
    ddp, device, world_size, rank, local_rank = setup_ddp(cfg)

    # Setup seed
    setup_seed(cfg.get("seed"), rank)

    # Setup output directory
    _setup_output_dir(cfg, rank, ddp, local_rank)

    # Create datasets
    if rank == 0:
        print("Creating datasets...")
    train_dataset, val_dataset = create_datasets(cfg, rank)
    if rank == 0:
        print(
            f"Datasets created: train={len(train_dataset)}, val={len(val_dataset) if val_dataset else 0}"
        )

    # Create data loaders
    if rank == 0:
        print(f"[{_timestamp()}] Creating data loaders...")
        print(f"[{_timestamp()}]   Global batch size: {cfg.get('batch_size', 64)}")
        print(f"[{_timestamp()}]   Workers: {cfg.get('num_workers', 4)}")
        if ddp:
            print(f"[{_timestamp()}]   World size: {world_size}")
    train_loader, val_loader = create_data_loaders(
        train_dataset, val_dataset, cfg, world_size, rank, ddp
    )

    # Test sample loading
    _test_sample_loading(train_dataset, rank)

    # Create model
    if rank == 0:
        print("Creating model...")
    model = create_model(cfg, device, ddp, local_rank, rank)

    # Create optimizer and scheduler
    if rank == 0:
        print("Creating optimizer and scheduler...")
    optimizer, scheduler = create_optimizer_scheduler(cfg, model, train_loader, rank)
    if rank == 0:
        print(f"Train loader has {len(train_loader)} batches")
        print(f"\n[{_timestamp()}] Starting training: {cfg.get('epochs', 100)} epochs")
        print("=" * 50)

    # Test first batch
    _test_first_batch(train_loader, rank)
    if rank == 0:
        print("Entering training loop...")
    final_acc = train_loop(
        cfg, model, train_loader, val_loader, optimizer, scheduler, device, ddp, rank
    )

    if rank == 0:
        print(f"\n[{_timestamp()}] Training done. Final accuracy: {final_acc:.4f}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
