"""Explicitly launch the full comparison; never run on import."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

from .data import DEFAULT_DATA_DIR, PROJECT_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--models", nargs="+", choices=("mlp", "cnn", "resnet18", "classical"),
                        default=["mlp", "cnn", "resnet18", "classical"])
    parser.add_argument("--run-name", default="experiments",
                        help="A new directory name under reports/, outputs/ and checkpoints/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--benchmark-repeats", type=int, default=50)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if (not args.run_name or args.run_name in (".", "..")
            or Path(args.run_name).name != args.run_name):
        parser.error("--run-name must be a single directory name")
    if len(set(args.models)) != len(args.models):
        parser.error("Duplicate models")
    if args.epochs < 1 or args.threads < 1 or args.benchmark_repeats < 1:
        parser.error("Epochs, threads and benchmark repeats must be positive")
    data_dir = str(Path(args.data_dir).resolve())
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_DIR / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["MPLBACKEND"] = "Agg"
    report_dirs = []
    # Refuse accidental replacement of historical or partially completed runs.
    for model in args.models:
        for parent in ("reports", "outputs", "checkpoints"):
            if parent == "reports" and model == "mlp":
                target = PROJECT_DIR / "reports/mlp/runs" / args.run_name
            else:
                target = PROJECT_DIR / parent / args.run_name / model
            if target.exists():
                parser.error(f"{target} already exists; choose a new --run-name")

    def run(module, *arguments):
        command = [sys.executable, "-m", f"wikiart.{module}", *map(str, arguments)]
        print(subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=PROJECT_DIR, env=env, check=True)

    for model in args.models:
        report = (Path("reports/mlp/runs") / args.run_name if model == "mlp"
                  else Path("reports") / args.run_name / model)
        report_dirs.append(report)
        checkpoint_dir = Path("checkpoints") / args.run_name / model
        common = ["--data-dir", data_dir, "--seed", args.seed, "--threads", args.threads]
        if model == "mlp":
            # Adopt Viviam's 64px MLP and dropout study while retaining common
            # evaluation, CPU timing and output directories for all models.
            run("train_mlp", *common, "--epochs", args.epochs, "--device", args.device,
                "--image-size", 64, "--hidden1", 256, "--hidden2", 128,
                "--dropout", 0.5, "--weight-decay", 0.0, "--normalization", "half",
                "--output-dir", report, "--checkpoint", checkpoint_dir / "best_model.pt",
                "--history", Path("outputs") / args.run_name / model / "history.json",
                "--benchmark-repeats", args.benchmark_repeats, "--test")
            continue
        if model == "classical":
            run("classical", *common, "--output-dir", report,
                "--checkpoint", checkpoint_dir / "model.joblib",
                "--benchmark-repeats", args.benchmark_repeats)
            run("analyze", "--data-dir", data_dir, "--report-dir", report)
            continue
        settings = {
            "cnn": (128, "train", 0.001),
            "resnet18": (224, "imagenet", 0.0001),
        }
        size, normalization, rate = settings[model]
        history = Path("outputs") / args.run_name / model / "history.json"
        checkpoint = checkpoint_dir / "best_model.pt"
        run("train", *common, "--model", model, "--epochs", args.epochs,
            "--image-size", size, "--normalization", normalization,
            "--optimizer", "adam", "--learning-rate", rate, "--device", args.device,
            "--checkpoint", checkpoint, "--history", history)
        run("evaluate", "--data-dir", data_dir, "--threads", args.threads,
            "--checkpoint", checkpoint, "--history", history, "--output-dir", report,
            "--benchmark-repeats", args.benchmark_repeats)
        run("analyze", "--data-dir", data_dir, "--report-dir", report, "--history", history)
    if len(report_dirs) > 1:
        run("compare", "--report-dirs", *report_dirs,
            "--output-dir", Path("reports") / args.run_name)


if __name__ == "__main__":
    main()
