"""Train simple image classifier for fall detection.

Refactored version with clean separation of concerns.
"""

import argparse
import ast
import os
import random
import time
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from fall_detection.data import CocoFallDataset, VOCFallDataset, TrainingAugmentation
from fall_detection.models.simple_classifier import SimpleFallClassifier
from fall_detection.utils import WarmupScheduler


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train simple image classifier from COCO or Pascal VOC annotations"
    )
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--local-rank", type=int, default=-1)
    parser.add_argument(
        "--override",
        type=str,
        default=None,
        help="Override config values, e.g., 'lr=0.01,batch_size=128'",
    )
    return parser.parse_args()


def load_config(args) -> Dict[str, Any]:
    """Load and parse configuration."""
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Process command line overrides
    if args.override:
        for override in args.override.split(","):
            key, value = override.split("=")
            keys = key.split(".")
            d = cfg
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
            d[keys[-1]] = value

    return cfg


def setup_ddp(cfg: Dict[str, Any]) -> Tuple[bool, torch.device, int, int, int]:
    """Setup Distributed Data Parallel.

    Returns:
        (ddp_enabled, device, world_size, rank, local_rank)
    """
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1

    if ddp:
        backend = cfg.get("ddp", {}).get("backend", "nccl")
        dist.init_process_group(backend=backend if torch.cuda.is_available() else "gloo")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        rank = 0
        local_rank = 0

    return ddp, device, world_size, rank, local_rank


def setup_seed(seed: Optional[int], rank: int) -> None:
    """Set random seed for reproducibility."""
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if rank == 0:
        print(f"Random seed set to {seed}")


def create_datasets(cfg: Dict[str, Any], rank: int):
    """Create training and validation datasets."""
    data_cfg = cfg.get("data", {})
    input_cfg = cfg.get("input", {})
    crop_cfg = cfg.get("crop", {})
    aug_cfg = cfg.get("data_augmentation", {})
    data_format = data_cfg.get("format", "coco").lower()

    # Create training augmentation
    train_transform = None if not aug_cfg.get("enabled", True) else TrainingAugmentation(aug_cfg)

    if data_format == "voc":
        voc_cfg = cfg.get("voc", {})
        train_dirs = voc_cfg.get("train_dirs", voc_cfg.get("data_dir"))
        if train_dirs is None:
            raise ValueError("voc.train_dirs or voc.data_dir must be specified for VOC format")
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

        # Cache directory for samples
        cache_dir = cfg.get("cache", {}).get("dir", "outputs/cache")

        train_dataset = VOCFallDataset(
            data_dirs=train_dirs,
            split="train",
            transform=train_transform,
            target_size=input_cfg.get("target_size", 96),
            use_letterbox=input_cfg.get("use_letterbox", True),
            fill_value=input_cfg.get("fill_value", 114),
            fall_classes=fall_classes,
            normal_classes=normal_classes,
            shrink_max=crop_cfg.get("shrink_max", 3),
            expand_max=crop_cfg.get("expand_max", 25),
            cache_dir=cache_dir,
        )

        val_dataset = None
        if val_dirs:
            if rank == 0:
                print(f"Loading validation data from VOC directories: {val_dirs}")
            val_dataset = VOCFallDataset(
                data_dirs=val_dirs,
                split="val",
                transform=None,  # No augmentation for validation
                target_size=input_cfg.get("target_size", 96),
                use_letterbox=input_cfg.get("use_letterbox", True),
                fill_value=input_cfg.get("fill_value", 114),
                fall_classes=fall_classes,
                normal_classes=normal_classes,
                shrink_max=crop_cfg.get("shrink_max", 3),
                expand_max=crop_cfg.get("expand_max", 25),
                cache_dir=cache_dir,
            )
    else:
        # COCO format
        train_coco_json = data_cfg.get("train_coco_json")
        val_coco_json = data_cfg.get("val_coco_json")
        image_dir = data_cfg.get("image_dir")

        if not train_coco_json or not image_dir:
            raise ValueError("train_coco_json and image_dir must be specified for COCO format")

        if rank == 0:
            print(f"Loading training data from: {train_coco_json}")

        train_dataset = CocoFallDataset(
            image_dir=image_dir,
            coco_json=train_coco_json,
            transform=train_transform,
            target_size=input_cfg.get("target_size", 96),
            use_letterbox=input_cfg.get("use_letterbox", True),
            fill_value=input_cfg.get("fill_value", 114),
            person_category_id=cfg.get("coco", {}).get("person_category_id", 1),
            fall_category_id=cfg.get("coco", {}).get("fall_category_id", 1),
            shrink_max=crop_cfg.get("shrink_max", 3),
            expand_max=crop_cfg.get("expand_max", 25),
        )

        val_dataset = None
        if val_coco_json and os.path.exists(val_coco_json):
            if rank == 0:
                print(f"Loading validation data from: {val_coco_json}")
            val_dataset = CocoFallDataset(
                image_dir=image_dir,
                coco_json=val_coco_json,
                transform=None,
                target_size=input_cfg.get("target_size", 96),
                use_letterbox=input_cfg.get("use_letterbox", True),
                fill_value=input_cfg.get("fill_value", 114),
                person_category_id=cfg.get("coco", {}).get("person_category_id", 1),
                fall_category_id=cfg.get("coco", {}).get("fall_category_id", 1),
                shrink_max=crop_cfg.get("shrink_max", 3),
                expand_max=crop_cfg.get("expand_max", 25),
            )

    if rank == 0:
        print(f"Train samples: {len(train_dataset)}")
        labels = [s[2] for s in train_dataset.samples]
        n_fall = sum(labels)
        n_normal = len(labels) - n_fall
        print(f"  Fall: {n_fall}, Normal: {n_normal}")

        if val_dataset:
            print(f"Val samples: {len(val_dataset)}")
            labels = [s[2] for s in val_dataset.samples]
            n_fall = sum(labels)
            n_normal = len(labels) - n_fall
            print(f"  Fall: {n_fall}, Normal: {n_normal}")

    return train_dataset, val_dataset


def create_model(cfg: Dict[str, Any], device: torch.device, ddp: bool, local_rank: int) -> nn.Module:
    """Create and setup model."""
    model_cfg = cfg.get("model", {})
    ddp_cfg = cfg.get("ddp", {})

    dropout = model_cfg.get("dropout", 0.3)
    fall_class_idx = model_cfg.get("fall_class_idx", 1)
    model = SimpleFallClassifier(dropout=dropout, fall_class_idx=fall_class_idx).to(device)

    if ddp:
        find_unused = ddp_cfg.get("find_unused_parameters", False)
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused,
        )

    return model


def create_optimizer_scheduler(
    cfg: Dict[str, Any], model: nn.Module, train_loader: DataLoader, rank: int
):
    """Create optimizer and learning rate scheduler."""
    lr_cfg = cfg.get("lr_scheduler", {})

    lr = cfg.get("lr", 0.001)
    weight_decay = cfg.get("weight_decay", 0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Base scheduler
    scheduler_type = lr_cfg.get("type", "plateau")
    if scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=lr_cfg.get("factor", 0.5),
            patience=lr_cfg.get("patience", 10),
            min_lr=lr_cfg.get("min_lr", 1e-5),
        )
    elif scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=lr_cfg.get("T_max", cfg.get("epochs", 100)),
            eta_min=lr_cfg.get("min_lr", 1e-5),
        )
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=lr_cfg.get("step_size", 30),
            gamma=lr_cfg.get("gamma", 0.1),
        )
    else:
        scheduler = None

    # Wrap with warmup
    warmup_cfg = lr_cfg.get("warmup", {})
    if warmup_cfg.get("enabled", False):
        warmup_epochs = warmup_cfg.get("epochs", 5)
        warmup_steps = warmup_epochs * len(train_loader)
        scheduler = WarmupScheduler(
            optimizer,
            scheduler,
            warmup_steps=warmup_steps,
            warmup_strategy=warmup_cfg.get("strategy", "linear"),
            warmup_init_lr=warmup_cfg.get("init_lr", 1e-5),
        )
        if rank == 0:
            print(
                f"Warmup enabled: {warmup_epochs} epochs = {warmup_steps} steps, "
                f"strategy={warmup_cfg.get('strategy', 'linear')}, "
                f"init_lr={warmup_cfg.get('init_lr', 1e-5)}"
            )

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
                f"  Epoch[{epoch}] Batch[{batch_idx + 1}/{num_batches}]  "
                f"loss={loss.item():.4f} acc={batch_acc:.4f}  "
                f"avg_loss={avg_loss:.4f} avg_acc={avg_acc:.4f}  lr={current_lr:.6f}",
                flush=True,
            )

    return total_loss, total_correct, total_samples


def eval_epoch(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, int, int]:
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    return total_loss, total_correct, total_samples


def format_time_remaining(remaining_secs: int) -> str:
    """Format remaining time as human-readable string."""
    remaining_mins = int(remaining_secs / 60)
    if remaining_mins >= 60:
        remaining_hours = remaining_mins // 60
        remaining_mins_remainder = remaining_mins % 60
        return f"{remaining_hours}h{remaining_mins_remainder}m"
    return f"{remaining_mins}m{int(remaining_secs % 60)}s"


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
    criterion = nn.CrossEntropyLoss()
    epochs = cfg.get("epochs", 100)
    log_cfg = cfg.get("log", {})
    epoch_log_interval = log_cfg.get("epoch_log_interval", 1)
    early_cfg = cfg.get("early_stopping", {})
    output_cfg = cfg.get("output", {})
    lr_cfg = cfg.get("lr_scheduler", {})
    scheduler_type = lr_cfg.get("type", "plateau")

    best_acc = 0.0
    patience_counter = 0
    epoch_times = []

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
            v_loss, v_correct, v_samples = eval_epoch(model, val_loader, criterion, device)
        else:
            v_loss, v_correct, v_samples = 0.0, 0, 0

        # Aggregate metrics
        if ddp:
            metrics = torch.tensor(
                [t_loss, t_correct, t_samples, v_loss, v_correct, v_samples],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            t_loss, t_correct, t_samples, v_loss, v_correct, v_samples = metrics.tolist()

        t_loss_avg = t_loss / max(1, t_samples)
        t_acc = t_correct / max(1, t_samples)
        v_loss_avg = v_loss / max(1, v_samples) if val_loader else 0.0
        v_acc = v_correct / max(1, v_samples) if val_loader else t_acc

        # Time tracking
        epoch_duration = time.time() - epoch_start
        epoch_times.append(epoch_duration)
        avg_epoch_time = sum(epoch_times[-10:]) / min(len(epoch_times), 10)
        remaining_time = avg_epoch_time * (epochs - epoch)
        remaining_str = format_time_remaining(int(remaining_time))

        if rank == 0:
            # Update learning rate
            if scheduler is not None:
                if isinstance(scheduler, WarmupScheduler):
                    if val_loader and scheduler.scheduler is not None:
                        scheduler.step(v_acc)
                    else:
                        scheduler.step()
                elif scheduler_type == "plateau" and val_loader:
                    scheduler.step(v_acc)
                else:
                    scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]

            if epoch % epoch_log_interval == 0 or epoch == epochs:
                if val_loader:
                    print(
                        f"Epoch {epoch}/{epochs}  "
                        f"train_loss={t_loss_avg:.4f} train_acc={t_acc:.4f}  "
                        f"val_loss={v_loss_avg:.4f} val_acc={v_acc:.4f}  "
                        f"lr={current_lr:.6f}  remain={remaining_str}"
                    )
                else:
                    print(
                        f"Epoch {epoch}/{epochs}  "
                        f"train_loss={t_loss_avg:.4f} train_acc={t_acc:.4f}  "
                        f"lr={current_lr:.6f}  remain={remaining_str}"
                    )

            # Save model
            if val_loader and v_acc > best_acc:
                best_acc = v_acc
                patience_counter = 0
                save_path = os.path.join(
                    output_cfg.get("dir", "outputs/simple_classifier"), "best.pt"
                )
                torch.save(
                    model.module.state_dict() if ddp else model.state_dict(), save_path
                )
                if epoch % epoch_log_interval == 0:
                    print(f"  -> Saved best model (val_acc={v_acc:.4f})")
            elif not val_loader and epoch % output_cfg.get("save_every", 10) == 0:
                save_path = os.path.join(
                    output_cfg.get("dir", "outputs/simple_classifier"), f"epoch_{epoch}.pt"
                )
                torch.save(
                    model.module.state_dict() if ddp else model.state_dict(), save_path
                )
                print(f"  -> Saved checkpoint (epoch={epoch})")

            # Early stopping
            if early_cfg.get("enabled", True) and val_loader:
                if v_acc <= best_acc + early_cfg.get("min_delta", 0.001):
                    patience_counter += 1
                    if patience_counter >= early_cfg.get("patience", 20):
                        print(f"Early stopping at epoch {epoch}")
                        break
                else:
                    patience_counter = 0

    return best_acc if val_loader else t_acc


def main():
    """Main entry point."""
    args = parse_args()
    cfg = load_config(args)

    # Setup DDP
    ddp, device, world_size, rank, local_rank = setup_ddp(cfg)

    # Setup seed
    setup_seed(cfg.get("seed"), rank)

    # Create output directory
    if rank == 0:
        output_dir = cfg.get("output", {}).get("dir", "outputs/simple_classifier")
        os.makedirs(output_dir, exist_ok=True)
        import yaml

        with open(os.path.join(output_dir, "config_used.yaml"), "w") as f:
            yaml.dump(cfg, f)

    if ddp:
        dist.barrier()

    # Create datasets
    if rank == 0:
        print("Creating datasets...")
    train_dataset, val_dataset = create_datasets(cfg, rank)
    if rank == 0:
        print(f"Datasets created: train={len(train_dataset)}, val={len(val_dataset) if val_dataset else 0}")

    # Create data loaders
    if rank == 0:
        print("Creating data loaders...")
        print(f"  batch_size={cfg.get('batch_size', 64)}, num_workers={cfg.get('num_workers', 4)}")
    train_sampler = (
        DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if ddp
        else None
    )

    # Test single sample loading before creating DataLoader
    if rank == 0:
        print("  Testing dataset __getitem__...")
        try:
            sample_img, sample_label = train_dataset[0]
            print(f"  Sample loaded: shape={sample_img.shape}, label={sample_label}")
        except Exception as e:
            print(f"  ERROR loading sample: {e}")
            import traceback
            traceback.print_exc()
            raise

    if rank == 0:
        print("  Creating train DataLoader...")

    num_workers = cfg.get("num_workers", 4)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.get("batch_size", 64),
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = None
    if val_dataset:
        val_sampler = (
            DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
            if ddp
            else None
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.get("batch_size", 64),
            sampler=val_sampler,
            shuffle=False,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=True,
        )

    # Create model
    if rank == 0:
        print("Creating model...")
    model = create_model(cfg, device, ddp, local_rank)
    if rank == 0:
        print("Model created.")

    # Create optimizer and scheduler
    if rank == 0:
        print("Creating optimizer and scheduler...")
    optimizer, scheduler = create_optimizer_scheduler(cfg, model, train_loader, rank)
    if rank == 0:
        print("Optimizer and scheduler created.")
        print(f"Train loader has {len(train_loader)} batches")

    # Training loop
    if rank == 0:
        print(f"\nStarting training: {cfg.get('epochs', 100)} epochs")
        print("="*50)
        print("Testing first batch loading...")
        try:
            first_batch = next(iter(train_loader))
            print(f"First batch loaded: images={first_batch[0].shape}, labels={first_batch[1].shape}")
        except Exception as e:
            print(f"ERROR loading first batch: {e}")
            import traceback
            traceback.print_exc()
            raise
        print("Entering training loop...")
    final_acc = train_loop(
        cfg, model, train_loader, val_loader, optimizer, scheduler, device, ddp, rank
    )

    if rank == 0:
        print(f"\nTraining done. Final accuracy: {final_acc:.4f}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
