"""Common evaluation artifacts and comparable CPU latency measurements."""

import csv
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from threadpoolctl import threadpool_limits


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sample_signature(paths, labels):
    payload = json.dumps(list(zip(paths, map(int, labels))), separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def write_evaluation(output_dir, classes, paths, labels, probabilities, metadata):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if (probabilities.shape != (len(labels), len(classes)) or len(paths) != len(labels)
            or not len(labels) or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5, rtol=0)):
        raise ValueError("Invalid or mismatched class probabilities")
    # Float32 softmax rows can differ from one by ~1e-7; sklearn uses float64 here.
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    metrics = {
        "test_loss": float(log_loss(labels, probabilities, labels=range(len(classes)))),
        "test_accuracy": float(np.mean(labels == predictions)),
        "classification_report": classification_report(
            labels, predictions, labels=range(len(classes)), target_names=classes,
            output_dict=True, zero_division=0,
        ),
        "metadata": {
            **metadata, "classes": classes, "test_count": len(labels),
            "test_signature": sample_signature(paths, labels),
        },
    }
    write_json(output_dir / "test_metrics.json", metrics)
    with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "image_path", "true_style", "predicted_style", "confidence",
            *(f"probability_{name}" for name in classes),
        ])
        for path, label, prediction, row in zip(paths, labels, predictions, probabilities):
            writer.writerow([
                path, classes[label], classes[prediction], f"{max(row):.8f}",
                *(f"{value:.8f}" for value in row),
            ])
    matrix = confusion_matrix(labels, predictions, labels=range(len(classes)))
    with (output_dir / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true / predicted", *classes])
        writer.writerows([name, *row] for name, row in zip(classes, matrix.tolist()))
    fig, ax = plt.subplots(figsize=(8, 7))
    plot = ax.imshow(matrix, cmap="Blues")
    for (i, j), count in np.ndenumerate(matrix):
        ax.text(j, i, str(count), ha="center", va="center",
                color="white" if count > matrix.max() / 2 else "black")
    ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set(xlabel="Predicted style", ylabel="True style", title="WikiArt test set")
    fig.colorbar(plot, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    return metrics


def benchmark_cpu(model_predict, end_to_end_predict, *, repeats=50, warmup=5, threads=1):
    """Single-image warm latency; end-to-end includes file decoding and features."""
    if repeats < 1 or warmup < 0 or threads < 1:
        raise ValueError("Invalid benchmark settings")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    measurements = {}
    try:
        with threadpool_limits(limits=threads), torch.inference_mode():
            for name, predict in (("model_only", model_predict),
                                  ("end_to_end", end_to_end_predict)):
                for i in range(warmup):
                    predict(i)
                samples = []
                for i in range(repeats):
                    start = perf_counter()
                    predict(i)
                    samples.append((perf_counter() - start) * 1000)
                measurements[name] = {
                    "median_ms": float(np.median(samples)),
                    "p95_ms": float(np.percentile(samples, 95)),
                    "mean_ms": float(np.mean(samples)),
                }
    finally:
        torch.set_num_threads(previous_threads)
    return {
        **measurements, "repeats": repeats, "warmup": warmup,
        "threads": threads, "batch_size": 1,
        "platform": platform.platform(), "processor": platform.processor(),
        "python": platform.python_version(), "torch": torch.__version__,
        "note": "Warm process and OS file cache; excludes model loading. "
                "End-to-end includes JPEG decode, preprocessing and prediction. "
                "Desktop measurements must be repeated on the target kiosk CPU.",
    }
