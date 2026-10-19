"""Viviam's MLP experiments, integrated with Luna's checkpoints and reports.

Retains independent hidden sizes, AdamW/weight decay, annotated curves and
validation summaries from vivi@d395106. Test evaluation is opt-in (--test).
Historical results imported from that branch remain unchanged in reports/mlp/.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from . import evaluate, train
from .analyze import analyze_run
from .data import DEFAULT_DATA_DIR, PROJECT_DIR
from .reporting import write_json


def plot_curves(history, best_epoch, run_name, destination):
    """Viviam's annotated train/validation curves with a best-epoch marker."""
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, suffix, title, factor in zip(axes, ("loss", "acc"),
                                        ("Loss", "Accuracy (%)"), (1, 100)):
        train_values = [factor * value for value in history[f"train_{suffix}"]]
        val_values = [factor * value for value in history[f"val_{suffix}"]]
        ax.plot(epochs, train_values, marker="o", label="train")
        ax.plot(epochs, val_values, marker="o", label="validation")
        ax.axvline(best_epoch, color="gray", linestyle="--", label=f"best epoch ({best_epoch})")
        for epoch, train_value, val_value in zip(epochs, train_values, val_values):
            if len(epochs) > 20 and epoch % 5:
                continue
            for value, offset, color in ((train_value, 7, "tab:blue"),
                                         (val_value, -13, "tab:orange")):
                label = f"{value:.0f}" if suffix == "acc" else f"{value:.2f}"
                ax.annotate(label, (epoch, value), textcoords="offset points",
                            xytext=(0, offset), ha="center", fontsize=7, color=color)
        ax.set(title=f"MLP ({run_name}) - {title}", xlabel="Epoch", ylabel=title)
        ax.set_xticks([1] + list(range(5, len(epochs) + 1, 5)) if len(epochs) > 20 else epochs)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden1", type=int, default=256)
    parser.add_argument("--hidden2", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--normalization", choices=("half", "train"), default="half")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", help="Explicit report directory; defaults to reports/mlp/runs/<settings>")
    parser.add_argument("--checkpoint")
    parser.add_argument("--history")
    parser.add_argument("--benchmark-repeats", type=int, default=50)
    parser.add_argument("--test", action="store_true", help="Evaluate held-out test and CPU for the chosen model")
    args = parser.parse_args(argv)
    if (min(args.hidden1, args.hidden2, args.image_size, args.epochs, args.batch_size,
            args.threads, args.benchmark_repeats) < 1 or args.learning_rate <= 0
            or not 0 <= args.dropout < 1 or args.weight_decay < 0):
        parser.error("Invalid dimensions, training settings or regularization")
    run_name = (f"{args.image_size}px_h{args.hidden1}-{args.hidden2}"
                f"_dropout{args.dropout}_wd{args.weight_decay}")
    if args.seed:
        run_name += f"_seed{args.seed}"
    report = Path(args.output_dir) if args.output_dir else PROJECT_DIR / "reports/mlp/runs" / run_name
    checkpoint = Path(args.checkpoint) if args.checkpoint else PROJECT_DIR / "checkpoints/mlp_runs" / run_name / "best_model.pt"
    history_path = Path(args.history) if args.history else PROJECT_DIR / "outputs/mlp_runs" / run_name / "history.json"
    if (report.exists() or checkpoint.exists() or history_path.exists()
            or history_path.with_name("run.json").exists()):
        parser.error("Run output already exists; choose new output/history/checkpoint paths")
    # Use the shared split, identity checks, normalization records and checkpoint
    # writer. Keep AdamW and Viviam's baseline normalization of RGB around 0.5.
    arguments = [
        "--model", "mlp", "--optimizer", "adamw", "--data-dir", args.data_dir,
        "--image-size", args.image_size, "--hidden1", args.hidden1, "--hidden2", args.hidden2,
        "--dropout", args.dropout, "--weight-decay", args.weight_decay,
        "--learning-rate", args.learning_rate, "--epochs", args.epochs,
        "--batch-size", args.batch_size, "--normalization", args.normalization,
        "--seed", args.seed, "--device", args.device, "--threads", args.threads,
        "--checkpoint", checkpoint, "--history", history_path,
    ]
    history = train.main(list(map(str, arguments)))
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    run = json.loads(history_path.with_name("run.json").read_text(encoding="utf-8"))
    report.mkdir(parents=True, exist_ok=False)
    best = saved["best_epoch"] - 1
    # Keep raw histories and Viviam-style summaries versionable with the plots.
    write_json(report / f"history_{run_name}.json", history)
    write_json(report / f"summary_{run_name}.json", {
        "run": run_name, "parameters": sum(value.numel() for value in saved["model_state_dict"].values()),
        "best_epoch": best + 1, "val_loss_at_best": history["val_loss"][best],
        "val_acc_at_best": history["val_acc"][best], "train_acc_at_best": history["train_acc"][best],
        "gap_at_best": history["train_acc"][best] - history["val_acc"][best],
        "final_train_acc": history["train_acc"][-1], "final_val_loss": history["val_loss"][-1],
        "final_val_acc": history["val_acc"][-1], "training": run,
        "source": "Viviam MLP (vivi@d395106), adapted to the common Luna protocol",
    })
    if args.test:
        metrics = evaluate.main(list(map(str, [
            "--data-dir", args.data_dir, "--checkpoint", checkpoint, "--history", history_path,
            "--output-dir", report, "--batch-size", args.batch_size,
            "--threads", args.threads, "--benchmark-repeats", args.benchmark_repeats,
        ])))
        analyze_run(report, args.data_dir, history_path)
        print(f"Test accuracy: {metrics['test_accuracy']:.3f}")
    plot_curves(history, best + 1, run_name, report / f"curves_{run_name}.png")
    print(f"Summary and annotated curves saved under {report.resolve()}")


if __name__ == "__main__":
    main()
