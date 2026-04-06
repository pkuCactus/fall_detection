"""Train fall classifier from cached features (config-based).

Refactored to use config file like train_simple_classifier.
"""

import argparse
import ast
import glob
import os
import sys
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, "src")
from fall_detection.models.classifier import FallClassifier


class FeatureDataset(Dataset):
    """Dataset for cached features."""

    def __init__(self, cache_dir):
        self.files = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        roi = torch.from_numpy(data["roi"]).permute(2, 0, 1).float() / 255.0
        kpts = torch.from_numpy(data["kpts"]).float()
        motion = torch.from_numpy(data["motion"]).float()
        label = torch.tensor(float(data["label"]), dtype=torch.float32)
        return roi, kpts, motion, label


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train fall classifier from cached features"
    )
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument(
        "--override",
        type=str,
        default=None,
        help="Override config values, e.g., 'epochs=100,batch_size=64'",
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


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0
    for roi, kpts, motion, label in loader:
        roi, kpts, motion, label = (
            roi.to(device),
            kpts.to(device),
            motion.to(device),
            label.to(device),
        )
        optimizer.zero_grad()
        out = model(roi, kpts, motion).squeeze()
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * label.size(0)
        total_samples += label.size(0)
    return total_loss, total_samples


def eval_epoch(model, loader, criterion, device):
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for roi, kpts, motion, label in loader:
            roi, kpts, motion, label = (
                roi.to(device),
                kpts.to(device),
                motion.to(device),
                label.to(device),
            )
            out = model(roi, kpts, motion).squeeze()
            loss = criterion(out, label)
            total_loss += loss.item() * label.size(0)
            pred = (out >= 0.5).float()
            correct += (pred == label).sum().item()
            total += label.size(0)
    return total_loss, correct, total


def main():
    """Main entry point."""
    args = parse_args()
    cfg = load_config(args)

    # Get configuration values
    cache_dir = cfg.get("cache_dir", "outputs/cache")
    epochs = cfg.get("epochs", 100)
    batch_size = cfg.get("batch_size", 32)
    lr = cfg.get("lr", 0.001)
    weight_decay = cfg.get("weight_decay", 0.0001)
    val_ratio = cfg.get("val_ratio", 0.2)
    output_dir = cfg.get("output_dir", "outputs/classifier")
    dropout = cfg.get("dropout", 0.3)
    num_workers = cfg.get("num_workers", 0)

    # Early stopping
    early_cfg = cfg.get("early_stopping", {})
    early_enabled = early_cfg.get("enabled", True)
    early_patience = early_cfg.get("patience", 20)
    early_min_delta = early_cfg.get("min_delta", 0.001)

    # Setup DDP
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        backend = cfg.get("ddp", {}).get("backend", "nccl")
        dist.init_process_group(
            backend=backend if torch.cuda.is_available() else "gloo"
        )
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

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        # Save config
        with open(os.path.join(output_dir, "config_used.yaml"), "w") as f:
            yaml.dump(cfg, f)

    if ddp:
        dist.barrier()

    if rank == 0:
        print(f"Starting fusion classifier training...")
        print(f"  Config: {args.config}")
        print(f"  Cache dir: {cache_dir}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {lr}")
        print(f"  Val ratio: {val_ratio}")
        print(f"  Output dir: {output_dir}")

    # Load dataset
    dataset = FeatureDataset(cache_dir)
    n_val = int(len(dataset) * val_ratio)
    n_train = len(dataset) - n_val

    # Deterministic split across ranks
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        if ddp
        else None
    )
    val_sampler = (
        DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if ddp
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    if rank == 0:
        print(f"  Train samples: {len(train_ds)}")
        print(f"  Val samples: {len(val_ds)}")

    # Create model
    model = FallClassifier(dropout=dropout).to(device)
    if ddp:
        find_unused = cfg.get("ddp", {}).get("find_unused_parameters", False)
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused,
        )

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Learning rate scheduler
    lr_cfg = cfg.get("lr_scheduler", {})
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
            T_max=lr_cfg.get("T_max", epochs),
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

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t_loss, t_samples = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_correct, v_total = eval_epoch(model, val_loader, criterion, device)

        # Aggregate metrics across ranks
        metrics = torch.tensor(
            [t_loss, t_samples, v_loss, v_correct, v_total],
            device=device,
            dtype=torch.float64,
        )

        if ddp:
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

        t_loss, t_samples, v_loss, v_correct, v_total = metrics.tolist()

        t_loss_avg = t_loss / max(1, t_samples)
        v_loss_avg = v_loss / max(1, v_total)
        v_acc = v_correct / max(1, v_total)

        if rank == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch}/{epochs}  "
                f"train_loss={t_loss_avg:.4f}  "
                f"val_loss={v_loss_avg:.4f}  "
                f"val_acc={v_acc:.4f}  "
                f"lr={current_lr:.6f}"
            )

            if v_acc > best_acc + early_min_delta:
                best_acc = v_acc
                patience_counter = 0
                torch.save(
                    model.module.state_dict() if ddp else model.state_dict(),
                    os.path.join(output_dir, "best.pt"),
                )
                print(f"  -> Saved best model (val_acc={v_acc:.4f})")
            else:
                patience_counter += 1

            # Update learning rate
            if scheduler is not None:
                if scheduler_type == "plateau":
                    scheduler.step(v_acc)
                else:
                    scheduler.step()

            # Early stopping
            if early_enabled and patience_counter >= early_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if rank == 0:
        print(f"\nTraining done. Best val_acc={best_acc:.4f}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
