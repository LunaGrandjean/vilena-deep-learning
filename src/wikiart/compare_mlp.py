##Compare the MLP runs: read every summary_*.json and history_*.json, one table (terminal + markdown + image) and one figure with all the curves.

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import PROJECT_DIR

REPORT_DIR = PROJECT_DIR / "reports/mlp"
HEADERS = ["fix tested", "run (file name)", "parameters", "best epoch", "val loss (best)",
           "val acc (best)", "final train acc", "final val acc", "final gap", "final val loss"]


def describe(run):
    ##Readable label from the run name, e.g. '64px_h256-128_dropout0.5_wd0.0' -> 'dropout 0.5'
    number = r"\d+(?:\.\d*)?(?:[eE][-+]?\d+)?"
    match = re.fullmatch(
        rf"(\d+)px_h(\d+)-(\d+)_dropout({number})_wd({number})(?:_seed(-?\d+))?", run
    )
    if match is None:
        raise ValueError(f"Unrecognized MLP run name: {run}")
    size, h1, h2, dropout, wd, seed = match.groups()
    changes = []
    if (h1, h2) != ("256", "128"):
        changes.append(f"smaller model ({h1}/{h2})")
    if float(dropout) > 0:
        changes.append(f"dropout {dropout}")
    if float(wd) > 0:
        changes.append(f"weight decay {wd} (AdamW)")
    label = " + ".join(changes) if changes else "baseline"
    if size != "64":
        label += f" [{size}px]"
    if seed is not None:
        label += f" [seed {seed}]"
    return label


def load_runs(report_dirs=None):
    ##One dict per run: its summary, its readable label and its full history
    runs = []
    identities, names = set(), set()
    for directory in report_dirs if report_dirs is not None else [REPORT_DIR]:
        for path in sorted(Path(directory).glob("summary_*.json")):
            s = json.loads(path.read_text(encoding="utf-8"))
            if s["run"] in names:
                raise ValueError(f"Duplicate run: {s['run']}")
            names.add(s["run"])
            s["label"] = describe(s["run"])
            s["history"] = json.loads((path.parent / f"history_{s['run']}.json").read_text(encoding="utf-8"))
            lengths = {len(s["history"][key]) for key in ("train_loss", "train_acc", "val_loss", "val_acc")}
            if len(lengths) != 1 or not next(iter(lengths)) or not 1 <= s["best_epoch"] <= next(iter(lengths)):
                raise ValueError(f"Invalid history: {s['run']}")
            training = s.get("training")
            identity = (training["dataset_signature"], training["seed"],
                        tuple(sorted(training["split_counts"].items())), training["normalization"]) if training else ("historical",)
            identities.add(identity)
            runs.append(s)
    if not runs:
        raise ValueError("No summary_*.json files found in the requested directories")
    if len(identities) != 1:
        raise ValueError("MLP reports have different data/splits/normalization or mix historical and integrated runs")
    runs.sort(key=lambda s: (s["label"] != "baseline", s["label"]))   # baseline first, then alphabetical
    return runs


def table_rows(runs):
    rows = []
    for s in runs:
        rows.append([
            s["label"],
            s["run"].replace("_dropout", "\n_dropout"),                   # file name on 2 lines
            f"{s['parameters']:,}",
            str(s["best_epoch"]),
            f"{s['val_loss_at_best']:.3f}",
            f"{s['val_acc_at_best']:.1%}",
            f"{s['final_train_acc']:.1%}",
            f"{s['final_val_acc']:.1%}",
            f"{s['final_train_acc'] - s['final_val_acc']:.1%}",           # final gap = the overfitting measure
            f"{s['final_val_loss']:.3f}",
        ])
    return rows


def save_table(rows, output_dir=REPORT_DIR):
    # 1. Terminal
    flat = [[v.replace("\n", "") for v in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in flat)) for i, h in enumerate(HEADERS)]
    print(" | ".join(h.ljust(w) for h, w in zip(HEADERS, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in flat:
        print(" | ".join(v.ljust(w) for v, w in zip(r, widths)))

    # 2. Markdown
    lines = ["| " + " | ".join(HEADERS) + " |", "|" + "---|" * len(HEADERS)]
    lines += ["| " + " | ".join(v.replace("\n", "<br>") for v in r) + " |" for r in rows]
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 3. Image for the slides
    fig, ax = plt.subplots(figsize=(16, 0.8 + 0.7 * len(rows)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=HEADERS, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.6)                                                   # taller rows for the 2-line names
    table.auto_set_column_width(col=list(range(len(HEADERS))))            # columns fit their text
    for col in range(len(HEADERS)):
        table[0, col].set_facecolor("#dddddd")                            # grey header row
        table[0, col].set_text_props(weight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


LABEL_EVERY = 10          # write the value every N epochs
ANNOTATE_TRAIN = False    # True = also write the train accuracy (dashed lines); busier chart


def annotate(ax, epochs, values, color, as_percent, above=True):
    ##Dot + value every LABEL_EVERY epochs, in the colour of the line
    for x, y in zip(epochs, values):
        if x % LABEL_EVERY != 0:
            continue
        text = f"{y * 100:.0f}%" if as_percent else f"{y:.2f}"
        ax.plot(x, y, marker="o", color=color, markersize=4)
        ax.annotate(text, (x, y), textcoords="offset points", xytext=(0, 6 if above else -12),
                    ha="center", fontsize=7, color=color,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))


def save_curves(runs, output_dir=REPORT_DIR):
    ##One colour per run. Left: validation loss. Right: train (dashed) vs validation (solid) accuracy.
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(16, 5.5))
    for s in runs:
        h = s["history"]
        epochs = list(range(1, len(h["val_loss"]) + 1))
        line, = ax_loss.plot(epochs, h["val_loss"], label=s["label"])
        color = line.get_color()                                          # same colour in both panels
        ax_acc.plot(epochs, h["val_acc"], color=color, label=s["label"])
        ax_acc.plot(epochs, h["train_acc"], color=color, linestyle="--", alpha=0.6)

        annotate(ax_loss, epochs, h["val_loss"], color, as_percent=False)
        annotate(ax_acc, epochs, h["val_acc"], color, as_percent=True, above=False)
        if ANNOTATE_TRAIN:
            annotate(ax_acc, epochs, h["train_acc"], color, as_percent=True)

    ax_loss.set(title="Validation loss per epoch (lower is better)", xlabel="epoch", ylabel="validation loss")
    ax_acc.set(title="Accuracy per epoch: train (dashed) vs validation (solid)", xlabel="epoch", ylabel="accuracy")
    for ax in (ax_loss, ax_acc):
        final_epoch = max(len(s["history"]["val_loss"]) for s in runs)
        ax.set_xticks([1] + list(range(LABEL_EVERY, final_epoch + 1, LABEL_EVERY)))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("MLP: overfitting and regularization experiments")
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_curves.png", dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare Viviam-style MLP validation summaries and curves")
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--report-dir", type=Path, help="One directory of summaries (default: original Viviam reports)")
    sources.add_argument("--run-dirs", nargs="+", type=Path, help="Explicit integrated run directories under reports/mlp/runs/")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    directories = args.run_dirs or [args.report_dir or REPORT_DIR]
    output_dir = args.output_dir or (directories[0] if len(directories) == 1 else REPORT_DIR / "comparisons")
    try:
        runs = load_runs(directories)
    except ValueError as error:
        parser.error(str(error))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_table(table_rows(runs), output_dir)
    save_curves(runs, output_dir)
    print(f"\nSaved {output_dir}/comparison.md, comparison.png and comparison_curves.png")


if __name__ == "__main__":
    main()
