import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, "src")
from fall_detection.models.classifier import FallClassifier


class FeatureDataset(Dataset):
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


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for roi, kpts, motion, label in loader:
        roi, kpts, motion, label = roi.to(device), kpts.to(device), motion.to(device), label.to(device)
        optimizer.zero_grad()
        out = model(roi, kpts, motion).squeeze()
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * label.size(0)
        total_samples += label.size(0)
    return total_loss, total_samples


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for roi, kpts, motion, label in loader:
            roi, kpts, motion, label = roi.to(device), kpts.to(device), motion.to(device), label.to(device)
            out = model(roi, kpts, motion).squeeze()
            loss = criterion(out, label)
            total_loss += loss.item() * label.size(0)
            pred = (out >= 0.5).float()
            correct += (pred == label).sum().item()
            total += label.size(0)
    return total_loss, correct, total


def main():
    parser = argparse.ArgumentParser(description="Train fall classifier from cached features")
    parser.add_argument("--cache-dir", default="train/cache")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--output-dir", default="train/classifier")
    parser.add_argument("--local-rank", type=int, default=-1, help="Local rank for DDP")
    args = parser.parse_args()

    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
        rank = 0

    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)

    if ddp:
        dist.barrier()

    dataset = FeatureDataset(args.cache_dir)
    n_val = int(len(dataset) * args.val_ratio)
    n_train = len(dataset) - n_val
    # Deterministic split across ranks
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False) if ddp else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler,
        shuffle=False, num_workers=0, pin_memory=True
    )

    model = FallClassifier().to(device)
    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t_loss, t_samples = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_correct, v_total = eval_epoch(model, val_loader, criterion, device)

        # Aggregate metrics across ranks
        t_loss_tensor = torch.tensor(t_loss, device=device, dtype=torch.float64)
        t_samples_tensor = torch.tensor(t_samples, device=device, dtype=torch.float64)
        v_loss_tensor = torch.tensor(v_loss, device=device, dtype=torch.float64)
        v_correct_tensor = torch.tensor(v_correct, device=device, dtype=torch.float64)
        v_total_tensor = torch.tensor(v_total, device=device, dtype=torch.float64)

        if ddp:
            dist.all_reduce(t_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(t_samples_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(v_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(v_correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(v_total_tensor, op=dist.ReduceOp.SUM)

        t_loss_avg = t_loss_tensor.item() / max(1, t_samples_tensor.item())
        v_loss_avg = v_loss_tensor.item() / max(1, v_total_tensor.item())
        v_acc = v_correct_tensor.item() / max(1, v_total_tensor.item())

        if rank == 0:
            print(f"Epoch {epoch}/{args.epochs}  train_loss={t_loss_avg:.4f}  val_loss={v_loss_avg:.4f}  val_acc={v_acc:.4f}")
            if v_acc > best_acc:
                best_acc = v_acc
                torch.save(model.module.state_dict() if ddp else model.state_dict(),
                           os.path.join(args.output_dir, "best.pt"))

    if rank == 0:
        print(f"Training done. Best val_acc={best_acc:.4f}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
