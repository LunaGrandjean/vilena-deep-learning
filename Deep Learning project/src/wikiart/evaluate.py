"""Evaluate a neural checkpoint and measure single-image inference on CPU."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .data import DEFAULT_DATA_DIR, create_dataloaders, dataset_signature
from .model import build_model, load_model_state
from .reporting import benchmark_cpu, write_evaluation, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--history", help="Matching training history; run.json is read beside it")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--benchmark-repeats", type=int, default=50)
    args = parser.parse_args(argv)
    if args.threads < 1 or args.batch_size < 1 or args.benchmark_repeats < 1:
        parser.error("Threads, batch size and benchmark repeats must be positive")
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    seed = checkpoint.get("seed", 0)
    if args.seed is not None and args.seed != seed:
        parser.error("Use the checkpoint seed to preserve the held-out test split")
    _, val_loader, test_loader, classes = create_dataloaders(
        data_dir=args.data_dir, batch_size=args.batch_size,
        image_size=checkpoint["image_size"], seed=seed,
        normalization=checkpoint.get("normalization", "half"),
        normalization_stats=checkpoint.get("normalization_stats"), augment=False,
    )
    if classes != checkpoint["classes"]:
        raise ValueError("Dataset classes do not match checkpoint classes")
    signature = dataset_signature(test_loader.dataset.dataset)
    if checkpoint.get("dataset_signature") not in (None, signature):
        raise ValueError("Dataset contents changed since training; retrain before evaluating")
    architecture = checkpoint.get("architecture", "cnn")
    model = build_model(
        architecture=architecture, num_classes=checkpoint["num_classes"],
        dropout=checkpoint.get("dropout", 0.3), width=checkpoint.get("width", 1),
        pretrained=False, image_size=checkpoint["image_size"],
        hidden_dim=checkpoint.get("hidden_dim", 256),
        hidden1=checkpoint.get("hidden1"), hidden2=checkpoint.get("hidden2"),
    ).eval()
    load_model_state(model, checkpoint["model_state_dict"])
    probabilities, labels = [], []
    with torch.inference_mode():
        for images, targets in test_loader:
            probabilities.extend(torch.softmax(model(images), dim=1).tolist())
            labels.extend(targets.tolist())
    subset = test_loader.dataset
    paths = [Path(subset.dataset.samples[i][0]).relative_to(args.data_dir).as_posix()
             for i in subset.indices]
    run = None
    if args.history:
        history_path = Path(args.history)
        run_path = history_path.with_name("run.json")
        if not history_path.is_file() or not run_path.is_file():
            raise FileNotFoundError("History and sibling run.json are required together")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if (run["dataset_signature"] != signature or run["seed"] != seed
                or run["model"] != architecture
                or Path(run["checkpoint"]).resolve() != Path(args.checkpoint).resolve()):
            raise ValueError("Training metadata does not match this checkpoint")
    metrics = write_evaluation(
        args.output_dir, classes, paths, labels, np.asarray(probabilities), {
            "model": architecture, "seed": seed, "dataset_signature": signature,
            "image_size": checkpoint["image_size"],
            "normalization": checkpoint.get("normalization", "half"),
            "best_epoch": checkpoint.get("best_epoch"), "training": run,
            "hidden1": checkpoint.get("hidden1"), "hidden2": checkpoint.get("hidden2"),
            "dropout": checkpoint.get("dropout", 0.3),
            "mlp_implementation": checkpoint.get("mlp_implementation"),
        },
    )
    validation = val_loader.dataset
    transform = validation.dataset.transform
    tensors = [validation[i][0].unsqueeze(0) for i in range(min(16, len(validation)))]
    files = [validation.dataset.samples[i][0] for i in validation.indices[:16]]

    def predict_file(i):
        with Image.open(files[i % len(files)]) as image:
            tensor = transform(image.convert("RGB")).unsqueeze(0)
        return torch.softmax(model(tensor), dim=1)

    timings = benchmark_cpu(
        lambda i: torch.softmax(model(tensors[i % len(tensors)]), dim=1), predict_file,
        repeats=args.benchmark_repeats, threads=args.threads,
    )
    write_json(Path(args.output_dir) / "cpu_benchmark.json", {
        **timings, "model": architecture, "dataset_signature": signature,
        "test_signature": metrics["metadata"]["test_signature"],
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "artifact_bytes": Path(args.checkpoint).stat().st_size,
        "training_seconds": run["training_seconds"] if run else None,
        "training_device": run["device"] if run else None,
    })
    print(f"{architecture}: test accuracy {metrics['test_accuracy']:.3f}")
    print(f"CPU end-to-end median: {timings['end_to_end']['median_ms']:.3f} ms")
    print(f"Saved under {Path(args.output_dir).resolve()}")
    return metrics


if __name__ == "__main__":
    main()
