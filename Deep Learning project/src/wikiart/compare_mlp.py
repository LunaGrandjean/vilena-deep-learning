##Compare the MLP runs: read every summary_*.json and history_*.json, one table (terminal + markdown + image) and one figure with all the curves.

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

REPORT_DIR = Path("reports/mlp")
HEADERS = ["fix tested", "run (file name)", "parameters", "best epoch", "val loss (best)",
           "val acc (best)", "final train acc", "final val acc", "final gap", "final val loss"]


def describe(run):
    ##Readable label from the run name, e.g. '64px_h256-128_dropout0.5_wd0.0' -> 'dropout 0.5'
    size, h1, h2, dropout, wd = re.match(r"(\d+)px_h(\d+)-(\d+)_dropout([\d.]+)_wd([\d.]+)$", run).groups()
    changes = []
    if (h1, h2) != ("256", "128"):
        changes.append(f"smaller model ({h1}/{h2})")
    if float(dropout) > 0:
        changes.append(f"dropout {dropout}")
    if float(wd) > 0:
        changes.append(f"weight decay {wd} (ridge)")
    return " + ".join(changes) if changes else "baseline"


def load_runs():
    ##One dict per run: its summary, its readable label and its full history
    runs = []
    for path in REPORT_DIR.glob("summary_*.json"):
        s = json.loads(path.read_text(encoding="utf-8"))
        s["label"] = describe(s["run"])
        s["history"] = json.loads((REPORT_DIR / f"history_{s['run']}.json").read_text(encoding="utf-8"))
        runs.append(s)
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


def save_table(rows):
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
    (REPORT_DIR / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

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
    fig.savefig(REPORT_DIR / "comparison.png", dpi=200, bbox_inches="tight")


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


def save_curves(runs):
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
        ax.set_xticks([1] + list(range(LABEL_EVERY, max(epochs) + 1, LABEL_EVERY)))  # ticks 1, 10, 20...
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("MLP 64px: overfitting and the fixes tested (one change per run)")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "comparison_curves.png", dpi=150)


def main():
    runs = load_runs()
    save_table(table_rows(runs))
    save_curves(runs)
    print(f"\nSaved {REPORT_DIR}/comparison.md, comparison.png and comparison_curves.png")


if __name__ == "__main__":
    main()