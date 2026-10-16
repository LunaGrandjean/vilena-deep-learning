"""Evaluate a saved checkpoint on the fixed held-out test split."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

from .data import create_dataloaders
from .model import WikiArtCNN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/wikiart_styles")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    seed = checkpoint.get("seed", 0) if args.seed is None else args.seed
    _, _, test_loader, classes = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=checkpoint["image_size"],
        seed=seed,
    )
    if classes != checkpoint["classes"]:
        raise ValueError(
            f"Dataset classes {classes} do not match checkpoint classes {checkpoint['classes']}"
        )

    model = WikiArtCNN(
        num_classes=checkpoint["num_classes"], dropout=checkpoint["dropout"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    true_labels, predictions = [], []
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0.0, 0, 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, device_labels = images.to(device), labels.to(device)
            logits = model(images)
            total_loss += loss_fn(logits, device_labels).item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == device_labels).sum().item()
            total += labels.size(0)
            true_labels.extend(labels.tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    loss = total_loss / total
    accuracy = total_correct / total

    matrix = confusion_matrix(true_labels, predictions, labels=range(len(classes)))
    report = classification_report(
        true_labels,
        predictions,
        labels=range(len(classes)),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "test_metrics.json"
    metrics_path.write_text(
        json.dumps({"test_loss": loss, "test_accuracy": accuracy, "classification_report": report}, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true / predicted", *classes])
        for class_name, row in zip(classes, matrix):
            writer.writerow([class_name, *row.tolist()])

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(classes)), labels=classes, rotation=35, ha="right")
    ax.set_yticks(range(len(classes)), labels=classes)
    ax.set_xlabel("Predicted style")
    ax.set_ylabel("True style")
    ax.set_title("WikiArt style classification")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    print(f"test loss: {loss:.4f} | test accuracy: {accuracy:.3f}")
    print(f"metrics and confusion matrix saved under {output_dir.resolve()}")


if __name__ == "__main__":
    main()
