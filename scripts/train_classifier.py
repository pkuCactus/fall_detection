import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, "src")
from fall_detection.classifier import FallClassifier


class FeatureDataset(Dataset):
    def __init__(self, cache_dir):
        self.files = glob.glob(os.path.join(cache_dir, "*.npz"))

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
    for roi, kpts, motion, label in loader:
        roi, kpts, motion, label = roi.to(device), kpts.to(device), motion.to(device), label.to(device)
        optimizer.zero_grad()
        out = model(roi, kpts, motion).squeeze()
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


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
            total_loss += loss.item()
            pred = (out >= 0.5).float()
            correct += (pred == label).sum().item()
            total += label.size(0)
    return total_loss / len(loader), correct / max(1, total)


def main():
    parser = argparse.ArgumentParser(description="Train fall classifier from cached features")
    parser.add_argument("--cache-dir", default="train/cache")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--output-dir", default="train/classifier")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FeatureDataset(args.cache_dir)
    n_val = int(len(dataset) * args.val_ratio)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = FallClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        t_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_acc = eval_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch}/{args.epochs}  train_loss={t_loss:.4f}  val_loss={v_loss:.4f}  val_acc={v_acc:.4f}")
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best.pt"))

    print(f"Training done. Best val_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
