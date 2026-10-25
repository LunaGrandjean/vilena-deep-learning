##Demo figure: random test images, hand-picked hard test images and new images, with ResNet18's predictions

import argparse
import csv
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image

from .data import DEFAULT_DATA_DIR
from .predict import load_model, preprocess


def read_test_csv(csv_path):
    with open(csv_path, encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_item(row, kind):
    ##(path, true style, kind, CSV prediction, CSV confidence) for one row of the test CSV
    return (DEFAULT_DATA_DIR / row["image_path"], row["true_style"], kind,
            row["predicted_style"], float(row["confidence"]))


def pick_test_images(rows, n, seed):
    ##Random test images (never seen in training), chosen by a seed, not by hand
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    return [test_item(r, "test image (random)") for r in rows[:n]]


def pick_hard_test_images(rows, names):                                         
    ##Test images chosen ON PURPOSE by their image_path, to show the model's limits
    by_path = {r["image_path"]: r for r in rows}
    missing = [n for n in names if n not in by_path]
    if missing:
        raise ValueError(f"Not in the test CSV: {missing}")
    return [test_item(by_path[n], "hard case (picked)") for n in names]


def save_table(results, classes, table_dir):
    ##Save the results as CSV (for analysis) and Markdown
    table_dir.mkdir(parents=True, exist_ok=True)
    fields = ["kind", "image", "true_style", "predicted_style", "confidence",
              "csv_predicted_style", "csv_confidence", "check"] + [f"probability_{c}" for c in classes]
    with open(table_dir / "demo_predictions.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    lines = ["| Kind | Image | True style | Predicted | Confidence | Team evaluation (CSV) | Check |",
             "|---|---|---|---|---:|---|---|"]
    for r in results:
        csv_part = f"{r['csv_predicted_style']} ({r['csv_confidence']:.2%})" if r["csv_predicted_style"] else "—"
        lines.append(f"| {r['kind']} | {r['image']} | {r['true_style']} | {r['predicted_style']} "
                     f"| {r['confidence']:.2%} | {csv_part} | {r['check'] or '—'} |")
    (table_dir / "demo_predictions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Trained ResNet18 .pt file")
    parser.add_argument("--test-csv", default="reports/comparaison_v1/resnet18/test_predictions.csv")
    parser.add_argument("--n-test", type=int, default=3, help="How many random test images")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the random test images")
    parser.add_argument("--hard-test", nargs="*", default=[],                    # NEW
                        help="image_path values from the test CSV to add on purpose (e.g. Baroque/s54_r0751.jpg)")
    parser.add_argument("--new", nargs="*", default=[], help="Paths of new images (internet, photos)")
    parser.add_argument("--new-labels", nargs="*", default=[], help="Expected style of each new image, or ?")
    parser.add_argument("--output", default=str(Path.home() / "Desktop" / "demo_predictions.png"),
                        help="The figure (contains images: keep it OUT of the repo)")
    parser.add_argument("--table-dir", default="reports/demo",
                        help="Where to save the results table (numbers only: can go in the repo)")
    args = parser.parse_args()
    if args.new_labels and len(args.new_labels) != len(args.new):
        parser.error("Give one --new-labels value per --new image")

    model, checkpoint = load_model(args.checkpoint)
    classes = checkpoint["classes"]
    labels = args.new_labels or ["?"] * len(args.new)

    # Each item: (path, true style, kind, CSV prediction, CSV confidence). New images have no CSV values
    rows = read_test_csv(args.test_csv)
    items = pick_test_images(rows, args.n_test, args.seed)
    items += pick_hard_test_images(rows, args.hard_test)                         
    items += [(Path(p), label, "new image", None, None) for p, label in zip(args.new, labels)]

    results = []
    fig, axes = plt.subplots(2, len(items), figsize=(3.4 * len(items), 6.8),
                             gridspec_kw={"height_ratios": [3, 2]}, squeeze=False)
    for col, (path, true, kind, csv_pred, csv_conf) in enumerate(items):
        with torch.no_grad():
            probs = torch.softmax(model(preprocess(path, checkpoint)), dim=1)[0]
        pred = classes[int(probs.argmax())]
        confidence = probs.max().item()
        mark = "" if true == "?" else ("  ✓" if pred == true else "  ✗")

        status = ""
        if csv_pred is not None:
            status = "MATCH" if csv_pred == pred else "DIFFERENT"
        check = f" | CSV: {csv_pred} ({csv_conf:.2%})  {status}" if csv_pred else ""
        print(f"{kind:<20} {path.name:<45} true: {true:<14} demo: {pred} ({confidence:.2%}){check}")

        row = {"kind": kind, "image": path.name, "true_style": true, "predicted_style": pred,
               "confidence": confidence, "csv_predicted_style": csv_pred or "",
               "csv_confidence": csv_conf if csv_conf is not None else "", "check": status}
        row.update({f"probability_{c}": p for c, p in zip(classes, probs.tolist())})
        results.append(row)

        ax_img, ax_bar = axes[0, col], axes[1, col]
        ax_img.imshow(Image.open(path).convert("RGB"))
        ax_img.axis("off")
        ax_img.set_title(f"{kind}\n{path.name}\ntrue: {true}\npredicted: {pred} ({confidence:.2%}){mark}",
                         fontsize=8)

        colors = ["tab:green" if c == pred else "lightgray" for c in classes]
        ax_bar.barh(classes, probs.numpy(), color=colors)
        ax_bar.set_xlim(0, 1)
        ax_bar.invert_yaxis()
        ax_bar.tick_params(labelsize=8)
        for y, p in enumerate(probs.tolist()):
            ax_bar.text(min(p + 0.02, 0.72), y, f"{p:.2%}", va="center", fontsize=7)

    fig.suptitle("ResNet18 demo: predicted style probabilities", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    save_table(results, classes, Path(args.table_dir))
    print(f"Saved {args.output}")
    print(f"Saved {args.table_dir}/demo_predictions.csv and demo_predictions.md")


if __name__ == "__main__":
    main()