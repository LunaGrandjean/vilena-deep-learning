"""Train a WikiArt image classifier with validation tracking."""

import argparse
import json
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn

from .data import DEFAULT_DATA_DIR, create_dataloaders, dataset_signature
from .model import build_model
from .frozen_features import cache_feature_loaders
from .reporting import write_json


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    # Keep frozen pretrained BatchNorm layers' ImageNet running statistics fixed.
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) and not any(
            parameter.requires_grad for parameter in module.parameters()
        ):
            module.eval()
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
    optimizer_name,
    device,
    checkpoint_path,
    classes,
    seed,
    image_size,
    normalization,
    architecture,
    width,
    pretrained,
    hidden_dim=256,
    normalization_stats=None,
    data_signature=None,
    hidden1=None,
    hidden2=None,
    dropout=0.3,
    weight_decay=0.0,
    cached_features=False,
):
    loss_fn = nn.CrossEntropyLoss()
    if architecture == "resnet18":
        optimizer_parameters = [
            {"params": model.layer4.parameters(), "lr": learning_rate * 0.1},
            {"params": model.fc.parameters(), "lr": learning_rate},
        ]
    else:
        optimizer_parameters = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(optimizer_parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(optimizer_parameters, lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.Adam(optimizer_parameters, lr=learning_rate, weight_decay=weight_decay)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_head = None

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model.head if cached_features else model, train_loader, loss_fn, optimizer, device
        )
        val_loss, val_acc = evaluate(model.head if cached_features else model, val_loader, loss_fn, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "classes": classes,
                "num_classes": len(classes),
                "architecture": architecture,
                "width": width,
                "pretrained": pretrained,
                "pretrained_weights": "VGG16_Weights.IMAGENET1K_V1" if architecture == "vgg16" else None,
                "hidden_dim": hidden_dim,
                "hidden1": hidden1,
                "hidden2": hidden2,
                "mlp_implementation": "viviam_v1" if architecture == "mlp" else None,
                "normalization_stats": normalization_stats,
                "dataset_signature": data_signature,
                "image_size": image_size,
                "normalization": normalization,
                "dropout": dropout,
                "weight_decay": weight_decay,
                "seed": seed,
                "best_val_loss": best_val_loss,
                "best_val_acc": val_acc,
                "best_epoch": epoch + 1,
                "optimizer": optimizer_name,
                "learning_rate": learning_rate,
                "transfer_mode": "frozen_features_linear_head" if cached_features else None,
            }
            if cached_features:
                # Retain just the changing head until training finishes. Avoid
                # rewriting the same ~537 MB of frozen weights every epoch.
                best_head = {key: value.detach().cpu().clone()
                             for key, value in model.head.state_dict().items()}
            else:
                torch.save(checkpoint, checkpoint_path)

        print(
            f"epoch {epoch + 1:2d}/{epochs} | train loss {train_loss:.4f} "
            f"| train acc {train_acc:.3f} | val loss {val_loss:.4f} | val acc {val_acc:.3f}"
        )

    if cached_features:
        if best_head is None:
            raise ValueError("VGG16 training produced no finite validation checkpoint")
        model.head.load_state_dict(best_head)
        checkpoint["model_state_dict"] = model.state_dict()
        torch.save(checkpoint, checkpoint_path)
    return history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--history", default="outputs/history.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--model", choices=("mlp", "cnn", "resnet18", "vgg16"), default="cnn")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden1", type=int, help="MLP first hidden layer; overrides --hidden-dim")
    parser.add_argument("--hidden2", type=int, help="MLP second hidden layer; defaults to hidden1 // 2")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--width", type=int, choices=(1, 2), default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=("sgd", "adam", "adamw"), default="sgd")
    parser.add_argument(
        "--normalization", choices=("train", "half", "imagenet"), default="train"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args(argv)
    args.hidden1 = args.hidden_dim if args.hidden1 is None else args.hidden1
    args.hidden2 = args.hidden1 // 2 if args.hidden2 is None else args.hidden2
    if args.model == "vgg16":
        # Pretrained dropout layers are frozen in eval mode; the new head is linear.
        args.dropout = 0.0
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if (args.batch_size < 1 or args.image_size < 1 or args.hidden1 < 1
            or args.hidden2 < 1 or args.threads < 1):
        parser.error("Invalid batch size, image size, hidden dimension or thread count")
    if not 0 <= args.dropout < 1 or args.weight_decay < 0:
        parser.error("Dropout must be in [0, 1) and weight decay must be nonnegative")
    if args.model in ("resnet18", "vgg16") and args.normalization != "imagenet":
        parser.error("Pretrained ImageNet models require --normalization imagenet")
    if args.model == "vgg16" and args.image_size != 224:
        parser.error("The VGG16 experiment requires --image-size 224")

    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    preparation_start = perf_counter()
    train_loader, val_loader, test_loader, classes = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
        normalization=args.normalization,
        augment=args.model != "vgg16",
    )
    pretrained = args.model in ("resnet18", "vgg16")
    model = build_model(
        architecture=args.model,
        num_classes=len(classes),
        width=args.width,
        pretrained=pretrained,
        image_size=args.image_size,
        hidden_dim=args.hidden_dim,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        dropout=args.dropout,
    ).to(device)
    normalizer = train_loader.dataset.dataset.transform.transforms[-1]
    normalization_stats = {"mean": list(normalizer.mean), "std": list(normalizer.std)}
    data_signature = dataset_signature(train_loader.dataset.dataset)
    preparation_seconds = perf_counter() - preparation_start
    print(
        f"device: {device} | classes: {classes} | optimizer: {args.optimizer} "
        f"| normalization: {args.normalization} | model: {args.model} "
        f"| pretrained: {pretrained}"
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    training_start = perf_counter()
    feature_extraction_seconds = 0.0
    split_counts = {"train": len(train_loader.dataset),
                    "validation": len(val_loader.dataset), "test": len(test_loader.dataset)}
    if args.model == "vgg16":
        train_loader, val_loader = cache_feature_loaders(model, train_loader, val_loader, device, args.seed)
        if device.type == "cuda":
            torch.cuda.synchronize()
        feature_extraction_seconds = perf_counter() - training_start
    history = fit(
        model,
        train_loader,
        val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        optimizer_name=args.optimizer,
        device=device,
        checkpoint_path=args.checkpoint,
        classes=classes,
        seed=args.seed,
        image_size=args.image_size,
        normalization=args.normalization,
        architecture=args.model,
        width=args.width,
        pretrained=pretrained,
        hidden_dim=args.hidden_dim,
        normalization_stats=normalization_stats,
        data_signature=data_signature,
        hidden1=args.hidden1,
        hidden2=args.hidden2,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        cached_features=args.model == "vgg16",
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    training_seconds = perf_counter() - training_start
    history_path = Path(args.history)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    write_json(history_path.with_name("run.json"), {
        **vars(args), "data_dir": str(Path(args.data_dir).resolve()),
        "device": str(device), "torch": torch.__version__,
        "preparation_seconds": preparation_seconds,
        "training_seconds": training_seconds,
        "training_timing_scope": "Frozen feature extraction (when used), all epochs, validation and checkpoint writes; excludes preparation",
        "feature_extraction_seconds": feature_extraction_seconds,
        "transfer_mode": "frozen_features_linear_head" if args.model == "vgg16" else None,
        "pretrained_weights": "VGG16_Weights.IMAGENET1K_V1" if args.model == "vgg16" else None,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "dataset_signature": data_signature,
        "split_counts": split_counts,
    })
    print(f"best checkpoint: {Path(args.checkpoint).resolve()}")
    print(f"training history: {history_path.resolve()}")
    return history


if __name__ == "__main__":
    main()
