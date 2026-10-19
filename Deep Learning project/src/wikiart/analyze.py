"""Plot learning curves and produce measured error diagnostics for one run."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .data import DEFAULT_DATA_DIR


def learning_diagnostics(history, output_dir):
    keys = ("train_loss", "val_loss", "train_acc", "val_acc")
    arrays = {key: np.asarray(history[key], dtype=float) for key in keys}
    lengths = {len(values) for values in arrays.values()}
    if len(lengths) != 1 or not next(iter(lengths)) or not all(
            np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("History must contain four equally sized, nonempty, finite series")
    epochs = np.arange(1, len(arrays["train_loss"]) + 1)
    best = int(np.argmin(arrays["val_loss"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, suffix, label in zip(axes, ("loss", "acc"), ("Cross-entropy loss", "Accuracy")):
        ax.plot(epochs, arrays[f"train_{suffix}"], label="Train (augmentation + dropout)")
        ax.plot(epochs, arrays[f"val_{suffix}"], label="Validation")
        ax.axvline(best + 1, color="gray", linestyle="--", label="Selected checkpoint")
        ax.set(xlabel="Epoch", ylabel=label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "learning_curves.png", dpi=160)
    plt.close(fig)
    gap = arrays["train_acc"][best] - arrays["val_acc"][best]
    lines = [
        "## Apprentissage",
        f"- Époques enregistrées : {len(epochs)}.",
        f"- Modèle sélectionné : époque {best + 1}, perte de validation minimale "
        f"{arrays['val_loss'][best]:.4f}, accuracy de validation {arrays['val_acc'][best]:.1%}.",
        f"- À cette époque : accuracy train {arrays['train_acc'][best]:.1%}, "
        f"écart train − validation {gap:+.1%}.",
        f"- Perte de validation finale : {arrays['val_loss'][-1]:.4f}.",
    ]
    if (best < len(epochs) - 1 and arrays["val_loss"][-1] > arrays["val_loss"][best] * 1.1
            and arrays["train_loss"][-1] < arrays["train_loss"][best]):
        lines.append("- Signal compatible avec du surapprentissage : après le minimum de "
                     "validation, la perte train baisse mais la perte de validation finit "
                     "plus de 10 % au-dessus de ce minimum. À vérifier sur la courbe.")
    else:
        lines.append("- Le critère automatique de divergence (>10 % de hausse de perte "
                     "validation et baisse de perte train après le minimum) n'est pas rempli. "
                     "Cela ne prouve pas l'absence de surapprentissage.")
    if best == len(epochs) - 1:
        lines.append("- Le meilleur point est la dernière époque : la convergence reste à examiner.")
    lines.append("Les mesures train utilisent augmentation/dropout, celles de validation les "
                 "désactivent : leur écart ne mesure pas à lui seul le surapprentissage.")
    return lines


def analyze_run(report_dir, data_dir=DEFAULT_DATA_DIR, history_path=None):
    report_dir, data_dir = Path(report_dir), Path(data_dir)
    metrics = json.loads((report_dir / "test_metrics.json").read_text(encoding="utf-8"))
    with (report_dir / "test_predictions.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("No test predictions to analyze")
    lines = ["# Diagnostic de l'expérience", "", "Généré à partir des mesures de cette exécution.", ""]
    if history_path:
        history_path = Path(history_path)
        run_path = history_path.with_name("run.json")
        if not run_path.is_file():
            raise FileNotFoundError("A matching run.json must accompany the history")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        training = metrics.get("metadata", {}).get("training")
        if training != run:
            raise ValueError("History metadata does not match the evaluated training run")
        history = json.loads(history_path.read_text(encoding="utf-8"))
        lines.extend(learning_diagnostics(history, report_dir))
    else:
        lines.append("Pas d'historique fourni : aucun constat sur les courbes d'apprentissage.")
    wrong = sorted((row for row in rows if row["true_style"] != row["predicted_style"]),
                   key=lambda row: float(row["confidence"]), reverse=True)
    with (report_dir / "misclassified.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(wrong)
    lines.extend(["", "## Résultats et erreurs",
                  f"- Images de test : {len(rows)} ; erreurs : {len(wrong)}.",
                  f"- Accuracy test : {metrics['test_accuracy']:.1%}.",
                  f"- F1 macro : {metrics['classification_report']['macro avg']['f1-score']:.3f}.", "",
                  "| Style | Précision | Rappel | F1 |", "|---|---:|---:|---:|"])
    styles = sorted({row["true_style"] for row in rows})
    for style in styles:
        values = metrics["classification_report"][style]
        lines.append(f"| {style} | {values['precision']:.3f} | {values['recall']:.3f} | {values['f1-score']:.3f} |")
    if wrong:
        pairs = Counter((row["true_style"], row["predicted_style"]) for row in wrong)
        lines.extend(["", "Confusions les plus fréquentes (vrai → prédit) :"])
        lines.extend(f"- {true} → {predicted} : {count}." for (true, predicted), count in pairs.most_common(5))
        examples = wrong[:12]
        fig, axes = plt.subplots((len(examples) + 3) // 4, 4,
                                 figsize=(12, 3 * ((len(examples) + 3) // 4)), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, row in zip(axes.ravel(), examples):
            path = (data_dir / row["image_path"]).resolve()
            if not path.is_relative_to(data_dir.resolve()):
                raise ValueError("Image path outside the dataset")
            with Image.open(path) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_title(f"True: {row['true_style']}\nPred: {row['predicted_style']} "
                         f"({float(row['confidence']):.0%})", fontsize=9)
        fig.tight_layout()
        fig.savefig(report_dir / "misclassified_examples.png", dpi=140)
        plt.close(fig)
        lines.extend(["", "La galerie montre les erreurs ayant les probabilités prédites les plus élevées. "
                      "Ces probabilités ne sont pas nécessairement calibrées. L'examen visuel "
                      "doit compléter les comptages avant d'attribuer une cause aux erreurs."])
    lines.extend(["", "## Limites", "La séparation est stratifiée par image, pas par artiste. "
                  "Des tableaux d'un même peintre peuvent apparaître dans train et test. "
                  "Ces scores ne démontrent donc pas la généralisation à des artistes inconnus. "
                  "Les erreurs test servent au bilan final ; les réglages doivent rester fondés "
                  "sur la validation, sans réutiliser le test pour choisir un modèle."])
    (report_dir / "diagnostic.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--history")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    args = parser.parse_args()
    analyze_run(args.report_dir, args.data_dir, args.history)
    print(f"Diagnostic saved under {args.report_dir}")


if __name__ == "__main__":
    main()
