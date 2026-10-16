"""Train the WikiArt CNN using the TP05 train/validation workflow."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import create_dataloaders
from .model import WikiArtCNN


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        batch_size = labels.size(0)
        total_loss += loss_fn(logits, labels).item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size
    return total_loss / total, correct / total


def fit(
    model,
    train_loader,
    val_loader,
    epochs,
    learning_rate,
    device,
    checkpoint_path,
    classes,
    seed,
    image_size,
):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "num_classes": len(classes),
                    "image_size": image_size,
                    "dropout": 0.3,
                    "seed": seed,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": val_acc,
                },
                checkpoint_path,
            )

        print(
            f"epoch {epoch + 1:2d}/{epochs} | train loss {train_loss:.4f} "
            f"| train acc {train_acc:.3f} | val loss {val_loss:.4f} | val acc {val_acc:.3f}"
        )

    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/wikiart_styles")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--history", default="outputs/history.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, _, classes = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
    )
    model = WikiArtCNN(num_classes=len(classes)).to(device)
    print(f"device: {device} | classes: {classes}")

    history = fit(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
        checkpoint_path=args.checkpoint,
        classes=classes,
        seed=args.seed,
        image_size=args.image_size,
    )
    history_path = Path(args.history)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"best checkpoint: {Path(args.checkpoint).resolve()}")
    print(f"training history: {history_path.resolve()}")


if __name__ == "__main__":
    main()
