import argparse
import json
import os
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import SquareFolderDataset, DEFAULT_CLASSES, save_classes_json
from .model import build_model


def parse_args():
    p = argparse.ArgumentParser(description="Train square classifier for chess image->FEN")
    p.add_argument("--data-root", default="data", help="Root containing train/ and val/")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--out-dir", default="weights")
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def make_loaders(data_root: str, img_size: int, batch_size: int, num_workers: int):
    train_ds = SquareFolderDataset(os.path.join(data_root, "train"), classes=DEFAULT_CLASSES, img_size=img_size)
    val_ds = SquareFolderDataset(os.path.join(data_root, "val"), classes=DEFAULT_CLASSES, img_size=img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, train_ds.classes


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(1, total)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, classes = make_loaders(args.data_root, args.img_size, args.batch_size, args.num_workers)
    model = build_model(num_classes=len(classes)).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()
            running += loss.item() * x.size(0)
        sched.step()

        val_acc = evaluate(model, val_loader, device)
        train_loss = running / max(1, len(train_loader.dataset))
        print(f"Epoch {epoch:03d} | loss {train_loss:.4f} | val_acc {val_acc:.4f}")

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": model.state_dict(), "classes": classes}, os.path.join(args.out_dir, "best.pt"))
            save_classes_json(classes, os.path.join(args.out_dir, "classes.json"))
            print(f"Saved best model to {os.path.join(args.out_dir, 'best.pt')} (acc={best_acc:.4f})")


if __name__ == "__main__":
    main()
