##Train the feed-forward network (MLP) on exactly the same split as the CNN.

import json
import copy
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from .data import create_dataloaders
from .model import WikiArtMLP
from .train import set_seed, train_one_epoch, evaluate

#SETTINGS (identical for every run, so the runs can be compared)
EPOCHS = 50            # how many full passes through the training images
LEARNING_RATE = 1e-3   # step size of the optimizer (same as the CNN, for a fair comparison)
IMAGE_SIZE = 128        # 64x64 images: 12,288 inputs per image
SEED = 0               # same seed as the CNN -> same train/val/test split
OUTPUT_DIR = Path("reports/mlp")


def main():
    # 0. SETTINGS FOR THIS RUN: change ONE thing at a time, everything else stays identical   # NEW
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout", type=float, default=0.0)        # probability of switching a neuron off
    parser.add_argument("--weight-decay", type=float, default=0.0)   # L2 penalty (weight decay)-- ridge
    parser.add_argument("--hidden1", type=int, default=256)          # size of hidden layer 1
    parser.add_argument("--hidden2", type=int, default=128)          # size of hidden layer 2
    parser.add_argument("--test", action="store_true")               # evaluate on test: FINAL model only
    args = parser.parse_args()
    run_name = f"{IMAGE_SIZE}px_h{args.hidden1}-{args.hidden2}_dropout{args.dropout}_wd{args.weight_decay}"
    print(f"run: {run_name}")

    # 1. Fix the randomness so the run is reproducible
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. DATA: reuse data.py (stratified 70/15/15 split, standardized pixels)
    train_loader, val_loader, test_loader, classes = create_dataloaders(
        data_dir="data/wikiart_styles",
        image_size=IMAGE_SIZE,
        seed=SEED,
    )

    # 3. MODEL MLP (sizes and dropout come from the command line)                              
    model = WikiArtMLP(num_classes=len(classes), image_size=IMAGE_SIZE,
                       hidden1=args.hidden1, hidden2=args.hidden2, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {device} | classes: {classes}")
    print(f"MLP trainable parameters: {n_params:,}")

    # 4. COST FUNCTION: cross-entropy on the raw logits (it applies softmax internally)
    loss_fn = nn.CrossEntropyLoss()

    # 5. OPTIMIZER: AdamW = Adam + weight decay applied correctly.                          
    #    With weight_decay=0 it is exactly plain Adam.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=args.weight_decay)

    # 6. TRAINING LOOP: one epoch = one pass over all training images
    #    (train_one_epoch does zero_grad -> forward -> loss -> backward -> step for each batch)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    best_epoch = 0
    best_weights = None

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # EARLY STOPPING: remember the weights of the epoch with the lowest validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_weights = copy.deepcopy(model.state_dict())

        print(
            f"epoch {epoch + 1:2d}/{EPOCHS} | train loss {train_loss:.4f} | train acc {train_acc:.3f} "
            f"| val loss {val_loss:.4f} | val acc {val_acc:.3f}"
        )

    # 7. numbers (evidence for the report)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"history_{run_name}.json").write_text(json.dumps(history, indent=2), encoding="utf-8")   

    # 7b. SUMMARY of this run, to compare the runs in one table                                 
    b = best_epoch - 1                      # list index of the best epoch
    summary = {
        "run": run_name,
        "parameters": n_params,
        "best_epoch": best_epoch,
        "val_loss_at_best": history["val_loss"][b],
        "val_acc_at_best": history["val_acc"][b],
        "train_acc_at_best": history["train_acc"][b],
        "gap_at_best": history["train_acc"][b] - history["val_acc"][b],   # train - validation accuracy
        "final_train_acc": history["train_acc"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_val_acc": history["val_acc"][-1],
    }
    (OUTPUT_DIR / f"summary_{run_name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"SUMMARY {run_name} | best epoch {best_epoch} | val loss {summary['val_loss_at_best']:.3f} "
        f"| val acc {summary['val_acc_at_best']:.3f} | train acc {summary['train_acc_at_best']:.3f} "
        f"| gap {summary['gap_at_best']:.3f} | final val loss {summary['final_val_loss']:.3f}"
    )

    # 8. learning curves (train vs validation)
    epochs = list(range(1, EPOCHS + 1))
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15, 5))

    def plot_with_labels(ax, train_values, val_values, title, as_percent):
        # One dot per epoch on each line
        ax.plot(epochs, train_values, marker="o", label="train")
        ax.plot(epochs, val_values, marker="o", label="validation")
        ax.axvline(best_epoch, color="gray", linestyle="--", label=f"best epoch ({best_epoch})")

        # values for each dot
        for x, y_train, y_val in zip(epochs, train_values, val_values):
            if EPOCHS > 20 and x % 5 != 0:
                continue
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

        if EPOCHS > 20:
            ax.set_xticks([1] + list(range(5, EPOCHS + 1, 5)))   # 1, 5, 10, 15
        else:
            ax.set_xticks(epochs) # every epoch

        ax.grid(alpha=0.3)
        ax.legend()

    plot_with_labels(ax_loss, history["train_loss"], history["val_loss"],
                     f"MLP ({run_name}) - loss", as_percent=False)                
    plot_with_labels(ax_acc, history["train_acc"], history["val_acc"],
                     f"MLP ({run_name}) - accuracy (%)", as_percent=True)         

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"curves_{run_name}.png", dpi=150)                   
    print(f"Curves saved to {OUTPUT_DIR}/curves_{run_name}.png")                  

    # 9. TEST: only for the FINAL chosen model (--test), using the weights of the best epoch   
    if args.test:
        model.load_state_dict(best_weights)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
        print(f"best epoch {best_epoch} | test loss {test_loss:.4f} | test acc {test_acc:.3f}")

        # confusion matrix: rows = true style, columns = predicted style
        y_true, y_pred = [], []
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                logits = model(images.to(device))
                y_pred.extend(logits.argmax(dim=1).cpu().tolist())
                y_true.extend(labels.tolist())
        matrix = confusion_matrix(y_true, y_pred)

        results = {
            "best_epoch": best_epoch,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "confusion_matrix": matrix.tolist(),
        }
        (OUTPUT_DIR / f"test_metrics_{run_name}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")   

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(matrix, cmap="Blues")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, matrix[i, j], ha="center", va="center")
        ax.set_xticks(range(len(classes)), labels=classes, rotation=35)
        ax.set_yticks(range(len(classes)), labels=classes)
        ax.set_xlabel("predicted style")
        ax.set_ylabel("true style")
        ax.set_title(f"MLP ({run_name}) - test confusion matrix")                 
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"confusion_matrix_{run_name}.png", dpi=150)     


if __name__ == "__main__":
    main()