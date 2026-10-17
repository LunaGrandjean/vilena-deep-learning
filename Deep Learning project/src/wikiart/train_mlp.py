##Train the feed-forward network (MLP) on exactly the same split as the CNN.

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from .data import create_dataloaders
from .model import WikiArtMLP
from .train import set_seed, train_one_epoch, evaluate

#SETTINGS
EPOCHS = 20            # how many full passes through the training images
LEARNING_RATE = 1e-3   # step size of the optimizer (same as the CNN, for a fair comparison)
IMAGE_SIZE = 64       # 64x64 instead of 128x128: 12,288 inputs instead of 49,152
SEED = 0               # same seed as the CNN -> same train/val/test split


def main():
    # 1. Fix the randomness so the run is reproducible
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. DATA: reuse data.py (stratified 70/15/15 split, standardized pixels)
    train_loader, val_loader, test_loader, classes = create_dataloaders(
        data_dir="data/wikiart_styles",
        image_size=IMAGE_SIZE,
        seed=SEED,
    )

    # 3. MODEL MLP
    model = WikiArtMLP(num_classes=len(classes), image_size=IMAGE_SIZE).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {device} | classes: {classes}")
    print(f"MLP trainable parameters: {n_params:,}")

    # 4. COST FUNCTION: cross-entropy on the raw logits (it applies softmax internally)
    loss_fn = nn.CrossEntropyLoss()

    # 5. OPTIMIZER: Adam >> a version of gradient descent
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. TRAINING LOOP: one epoch = one pass over all training images
    #    (train_one_epoch does zero_grad -> forward -> loss -> backward -> step for each batch)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"epoch {epoch + 1:2d}/{EPOCHS} | train loss {train_loss:.4f} | train acc {train_acc:.3f} "
            f"| val loss {val_loss:.4f} | val acc {val_acc:.3f}"
        )

    # 7. numbers (evidence for the report)
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/history_mlp.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # 8. learning curves (train vs validation)
    epochs = list(range(1, EPOCHS + 1))
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15, 5))

    def plot_with_labels(ax, train_values, val_values, title, as_percent):
        # One dot per epoch on each line
        ax.plot(epochs, train_values, marker="o", label="train")
        ax.plot(epochs, val_values, marker="o", label="validation")

        # values for each dot
        for x, y_train, y_val in zip(epochs, train_values, val_values):
            if as_percent:
                text_train, text_val = f"{y_train * 100:.0f}", f"{y_val * 100:.0f}"   # 0.372 -> "37"
            else:
                text_train, text_val = f"{y_train:.2f}", f"{y_val:.2f}"               # 2.8670 -> "2.87"
            ax.annotate(text_train, (x, y_train), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7, color="tab:blue")
            ax.annotate(text_val, (x, y_val), textcoords="offset points", xytext=(0, -13),
                        ha="center", fontsize=7, color="tab:orange")

        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_xticks(epochs)    # epoch number on x 
        ax.grid(alpha=0.3)      
        ax.legend()

    plot_with_labels(ax_loss, history["train_loss"], history["val_loss"],
                     f"MLP ({IMAGE_SIZE}px) - loss", as_percent=False)
    plot_with_labels(ax_acc, history["train_acc"], history["val_acc"],
                     f"MLP ({IMAGE_SIZE}px) - accuracy (%)", as_percent=True)

    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(f"reports/figures/mlp_curves_{IMAGE_SIZE}px.png", dpi=150)
    print(f"Curves saved to reports/figures/mlp_curves_{IMAGE_SIZE}px.png")


if __name__ == "__main__":
    main()
    