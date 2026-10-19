"""Train a classical RGB-feature/logistic-regression baseline on the same splits."""

import argparse
from pathlib import Path
from time import perf_counter
import warnings

import joblib
import numpy as np
from PIL import Image
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torchvision.datasets import ImageFolder

from .data import DEFAULT_DATA_DIR, dataset_signature, split_indices
from .reporting import benchmark_cpu, write_evaluation, write_json


def image_features(image):
    """768 low-resolution RGB pixels + 96 normalized color histogram bins."""
    image = image.convert("RGB")
    small = np.asarray(image.resize((16, 16), Image.Resampling.BILINEAR), dtype=np.float32) / 255
    rgb = np.asarray(image, dtype=np.float32) / 255
    histograms = [np.histogram(rgb[..., channel], bins=32, range=(0, 1))[0]
                  for channel in range(3)]
    histograms = np.concatenate(histograms).astype(np.float32) / (rgb.shape[0] * rgb.shape[1])
    return np.concatenate([small.ravel(), histograms])


def features_from_file(path):
    with Image.open(path) as image:
        return image_features(image)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default="reports/classical")
    parser.add_argument("--checkpoint", default="checkpoints/classical.joblib")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--c-values", nargs="+", type=float, default=[0.01, 0.1, 1, 10])
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--benchmark-repeats", type=int, default=50)
    args = parser.parse_args()
    if args.threads < 1 or args.max_iter < 1 or args.benchmark_repeats < 1 or min(args.c_values) <= 0:
        parser.error("Threads, iterations, repeats and C values must be positive")
    dataset = ImageFolder(args.data_dir)
    train, val, test = split_indices(dataset.targets, args.seed)
    signature = dataset_signature(dataset)
    preparation_start = perf_counter()

    def extract(indices):
        return np.stack([features_from_file(dataset.samples[i][0]) for i in indices])

    train_x, val_x = extract(train), extract(val)
    train_y, val_y = np.asarray(dataset.targets)[train], np.asarray(dataset.targets)[val]
    feature_seconds = perf_counter() - preparation_start
    training_start = perf_counter()
    candidates = []
    best_model, best_loss, best_c = None, float("inf"), None
    with threadpool_limits(limits=args.threads):
        for c_value in args.c_values:
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=c_value, max_iter=args.max_iter, random_state=args.seed,
            ))
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model.fit(train_x, train_y)
            probabilities = model.predict_proba(val_x)
            loss = float(log_loss(val_y, probabilities, labels=range(len(dataset.classes))))
            candidates.append({"C": c_value, "validation_loss": loss,
                               "validation_accuracy": float(accuracy_score(val_y, probabilities.argmax(1)))})
            if loss < best_loss:
                best_model, best_loss, best_c = model, loss, c_value
        training_seconds = perf_counter() - training_start
        # Held-out test images are evaluated only after C has been selected on validation.
        test_x = extract(test)
        probabilities = best_model.predict_proba(test_x)
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": best_model, "classes": dataset.classes, "seed": args.seed,
                 "dataset_signature": signature, "features": "rgb16_hist32_v1"}, checkpoint)
    paths = [Path(dataset.samples[i][0]).relative_to(args.data_dir).as_posix() for i in test]
    run = {
        "model": "classical", "seed": args.seed, "device": "cpu", "threads": args.threads,
        "features": "16x16 RGB pixels and 32-bin histograms for each RGB channel (864 features)",
        "feature_extraction_train_val_seconds": feature_seconds,
        "training_seconds": training_seconds,
        "training_timing_scope": "Scaler fits, all C candidates and validation; excludes image features",
        "selected_C": best_c, "validation_candidates": candidates,
        "split_counts": {"train": len(train), "validation": len(val), "test": len(test)},
        "dataset_signature": signature,
    }
    metrics = write_evaluation(args.output_dir, dataset.classes, paths,
                               np.asarray(dataset.targets)[test], probabilities, {
        "model": "classical", "seed": args.seed, "dataset_signature": signature, "training": run,
    })
    write_json(Path(args.output_dir) / "selection.json", run)
    benchmark_files = [dataset.samples[i][0] for i in val[:16]]
    timings = benchmark_cpu(
        lambda i: best_model.predict_proba(val_x[i % len(benchmark_files):i % len(benchmark_files) + 1]),
        lambda i: best_model.predict_proba(features_from_file(
            benchmark_files[i % len(benchmark_files)]).reshape(1, -1)),
        repeats=args.benchmark_repeats, threads=args.threads,
    )
    classifier = best_model.named_steps["logisticregression"]
    write_json(Path(args.output_dir) / "cpu_benchmark.json", {
        **timings, "model": "classical", "dataset_signature": signature,
        "test_signature": metrics["metadata"]["test_signature"],
        "parameter_count": int(classifier.coef_.size + classifier.intercept_.size),
        "artifact_bytes": checkpoint.stat().st_size,
        "training_seconds": training_seconds, "training_device": "cpu",
    })
    print(f"classical: C={best_c}, test accuracy {metrics['test_accuracy']:.3f}")


if __name__ == "__main__":
    main()
