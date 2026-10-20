"""Compare measured accuracy, CPU latency, training time and artifact size."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compare_reports(report_dirs, output_dir):
    rows, identities, benchmark_settings = [], set(), set()
    for directory in map(Path, report_dirs):
        metrics = json.loads((directory / "test_metrics.json").read_text(encoding="utf-8"))
        cpu = json.loads((directory / "cpu_benchmark.json").read_text(encoding="utf-8"))
        meta = metrics.get("metadata", {})
        required = ("dataset_signature", "test_signature", "seed", "model")
        if any(key not in meta for key in required):
            raise ValueError(f"{directory}: legacy report without comparison metadata; rerun first")
        if (cpu["model"] != meta["model"]
                or cpu["dataset_signature"] != meta["dataset_signature"]
                or cpu["test_signature"] != meta["test_signature"]):
            raise ValueError(f"{directory}: CPU benchmark and accuracy report are from different runs")
        identities.add((meta["dataset_signature"], meta["test_signature"], meta["seed"]))
        benchmark_settings.add(tuple(cpu[key] for key in
                                     ("platform", "processor", "python", "torch", "threads", "batch_size", "repeats", "warmup")))
        training = meta.get("training") or {}
        preparation = training.get("preparation_seconds",
                                   training.get("feature_extraction_train_val_seconds"))
        rows.append({
            "model": meta["model"], "accuracy": metrics["test_accuracy"],
            "macro_f1": metrics["classification_report"]["macro avg"]["f1-score"],
            "cpu_model_median_ms": cpu["model_only"]["median_ms"],
            "cpu_total_median_ms": cpu["end_to_end"]["median_ms"],
            "cpu_total_p95_ms": cpu["end_to_end"]["p95_ms"],
            "preparation_seconds": preparation, "training_seconds": cpu["training_seconds"],
            "training_device": cpu["training_device"],
            "parameters": cpu["parameter_count"], "artifact_mb": cpu["artifact_bytes"] / 1e6,
        })
    if len(rows) < 2 or len(identities) != 1 or len(benchmark_settings) != 1:
        raise ValueError("Need >=2 reports with identical data, test split and CPU benchmark settings")
    if len({row["model"] for row in rows}) != len(rows):
        raise ValueError("Duplicate model names in comparison")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Comparaison des modèles", "",
             "Même jeu d'images, même séparation stratifiée, même graine et même protocole CPU.", "",
             "| Modèle | Accuracy test | F1 macro | CPU modèle (ms) | CPU total médian (ms) | CPU total p95 (ms) | Entraînement (s) | Taille (Mo) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        training_seconds = row["training_seconds"]
        duration = f"{training_seconds:.2f} ({row['training_device']})" if training_seconds is not None else "non mesuré"
        lines.append(f"| {row['model']} | {row['accuracy']:.1%} | {row['macro_f1']:.3f} | "
                     f"{row['cpu_model_median_ms']:.3f} | {row['cpu_total_median_ms']:.3f} | "
                     f"{row['cpu_total_p95_ms']:.3f} | {duration} | {row['artifact_mb']:.2f} |")
    frontier = [row["model"] for row in rows if not any(
        other["accuracy"] >= row["accuracy"]
        and other["cpu_total_median_ms"] <= row["cpu_total_median_ms"]
        and (other["accuracy"] > row["accuracy"] or other["cpu_total_median_ms"] < row["cpu_total_median_ms"])
        for other in rows)]
    best, fastest = max(rows, key=lambda row: row["accuracy"]), min(rows, key=lambda row: row["cpu_total_median_ms"])
    lines.extend(["", f"- Meilleure accuracy observée : {best['model']} ({best['accuracy']:.1%}).",
                  f"- Inférence complète la plus rapide : {fastest['model']} "
                  f"({fastest['cpu_total_median_ms']:.3f} ms en médiane).",
                  f"- Compromis non dominés en accuracy/latence médiane : {', '.join(frontier)}.", "",
                  "Le temps total inclut ouverture/décodage JPEG, prétraitement et prédiction "
                  "pour une seule image. Le temps modèle seul utilise une entrée déjà préparée. "
                  "Le processus et les caches sont chauds, le chargement initial des poids est exclu. "
                  "Les tailles mesurent les fichiers sérialisés, pas le pic de RAM.", "",
                  "Les temps d'entraînement couvrent toutes les époques (et leurs validations) pour "
                  "les réseaux, et tous les candidats C pour la régression logistique. La préparation "
                  "des données/caractéristiques est séparée dans le CSV. Les architectures utilisent "
                  "des prétraitements adaptés (tailles et normalisation enregistrées dans leurs métadonnées) ; les écarts "
                  "reflètent donc les pipelines complets, pas seulement les architectures.", "",
                  "Pour recommander un modèle au musée, fixer d'abord la qualité minimale et le "
                  "budget de latence, puis confirmer ces mesures sur le CPU des bornes. "
                  "Le meilleur score test seul ne suffit pas à trancher. Aucun choix de réglages "
                  "ne doit être fait à partir des erreurs test. Ces résultats portent sur une graine "
                  "et une séparation par image ; ils ne prouvent ni la stabilité sur plusieurs "
                  "séparations ni la généralisation à de nouveaux artistes."])
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in rows:
        ax.scatter(row["cpu_total_median_ms"], 100 * row["accuracy"], s=70)
        ax.annotate(row["model"], (row["cpu_total_median_ms"], 100 * row["accuracy"]),
                    xytext=(5, 5), textcoords="offset points")
    ax.set(xscale="log", xlabel="CPU single-image end-to-end median (ms, log scale)",
           ylabel="Test accuracy (%)", title="Accuracy / CPU latency trade-off")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_cpu.png", dpi=160)
    plt.close(fig)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", default="reports/experiments")
    args = parser.parse_args()
    compare_reports(args.report_dirs, args.output_dir)
    print(f"Comparison saved under {args.output_dir}")


if __name__ == "__main__":
    main()
