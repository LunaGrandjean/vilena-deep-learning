"""Exercise the pipeline on temporary synthetic images, never on WikiArt."""

from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from PIL import Image
import torch
from torchvision.datasets import ImageFolder

# Also supports unittest before an editable package installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wikiart import analyze, classical, evaluate, run_experiments, train, train_mlp
from wikiart.compare import compare_reports
from wikiart.data import dataset_signature, split_indices
from wikiart.model import build_model, load_model_state


def run_main(module, *arguments):
    with patch.object(sys, "argv", [module.__name__, *map(str, arguments)]), redirect_stdout(io.StringIO()):
        module.main()


def make_images(data):
    rng = np.random.default_rng(7)
    for label in range(5):
        folder = data / f"style_{label}"
        folder.mkdir(parents=True)
        for i in range(20):
            pixels = rng.integers(0, 40, (16, 16, 3), dtype=np.uint8)
            pixels[:, :, label % 3] += 100 + label * 15
            Image.fromarray(pixels).save(folder / f"{i:02d}.png")


class WorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_runner_uses_central_mlp_folder_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="wikiart-routing-test-") as temp:
            root = Path(temp)
            arguments = ["--models", "mlp", "--run-name", "test_run"]
            with patch.object(run_experiments, "PROJECT_DIR", root), \
                    patch.object(run_experiments.subprocess, "run") as launch:
                run_main(run_experiments, *arguments)
                command = launch.call_args.args[0]
                output_dir = command[command.index("--output-dir") + 1]
                self.assertEqual(Path(output_dir), Path("reports/mlp/runs/test_run"))
                (root / output_dir).mkdir(parents=True)
                launch.reset_mock()
                with redirect_stdout(io.StringIO()), patch.object(sys, "stderr", io.StringIO()), \
                        self.assertRaises(SystemExit):
                    run_main(run_experiments, *arguments)
                launch.assert_not_called()

    def test_historical_luna_mlp_checkpoint_keeps_identical_predictions(self):
        legacy = torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(3 * 16 * 16, 16),
            torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(16, 8),
            torch.nn.ReLU(), torch.nn.Dropout(0.3), torch.nn.Linear(8, 5),
        ).eval()
        old_state = {f"layers.{key}": value for key, value in legacy.state_dict().items()}
        restored = build_model("mlp", 5, image_size=16, hidden_dim=16).eval()
        load_model_state(restored, old_state)
        batch = torch.rand(3, 3, 16, 16)
        with torch.inference_mode():
            torch.testing.assert_close(restored(batch), legacy(batch), rtol=0, atol=0)
        self.assertIn("layers.1.weight", old_state)

    def test_viv_mlp_exploration_and_final_evaluation(self):
        with tempfile.TemporaryDirectory(prefix="wikiart-viv-test-") as temp:
            root = Path(temp)
            data = root / "images"
            make_images(data)
            for test_enabled in (False, True):
                with self.subTest(test_enabled=test_enabled):
                    destination = root / ("final" if test_enabled else "exploration")
                    report = destination / "report"
                    checkpoint = destination / "model.pt"
                    arguments = ["--data-dir", data, "--image-size", 16,
                                 "--hidden1", 16, "--hidden2", 7, "--dropout", 0.5,
                                 "--weight-decay", 0.01, "--epochs", 1,
                                 "--output-dir", report, "--checkpoint", checkpoint,
                                 "--history", destination / "history.json", "--benchmark-repeats", 3]
                    if test_enabled:
                        run_main(train_mlp, *arguments, "--test")
                        self.assertTrue((report / "test_metrics.json").is_file())
                        self.assertTrue((report / "cpu_benchmark.json").is_file())
                        self.assertTrue((report / "diagnostic.md").is_file())
                    else:
                        with patch.object(evaluate, "main", side_effect=AssertionError("Unexpected test evaluation")):
                            run_main(train_mlp, *arguments)
                        self.assertFalse((report / "test_metrics.json").exists())
                    saved = torch.load(checkpoint, weights_only=True)
                    self.assertEqual((saved["hidden1"], saved["hidden2"]), (16, 7))
                    self.assertEqual(saved["dropout"], 0.5)
                    self.assertEqual(saved["weight_decay"], 0.01)
                    self.assertEqual(saved["optimizer"], "adamw")
                    self.assertEqual(saved["normalization"], "half")
                    self.assertEqual(len(list(report.glob("summary_*.json"))), 1)
                    self.assertEqual(len(list(report.glob("history_*.json"))), 1)
                    self.assertEqual(len(list(report.glob("curves_*.png"))), 1)

    def test_neural_architectures_have_finite_logits_and_gradients(self):
        for architecture, size in (("mlp", 16), ("cnn", 32), ("resnet18", 32)):
            with self.subTest(model=architecture):
                model = build_model(architecture, 5, image_size=size, hidden_dim=16, pretrained=False)
                logits = model(torch.rand(2, 3, size, size))
                self.assertEqual(tuple(logits.shape), (2, 5))
                loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 4]))
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                    if p.grad is not None))

    def test_full_synthetic_workflow_and_comparison_guards(self):
        with tempfile.TemporaryDirectory(prefix="wikiart-test-") as temp:
            root = Path(temp)
            data = root / "images"
            make_images(data)
            dataset = ImageFolder(data)
            train_indices, val_indices, test_indices = split_indices(dataset.targets, 0)
            self.assertEqual((len(train_indices), len(val_indices), len(test_indices)), (70, 15, 15))
            self.assertFalse(set(train_indices) & set(test_indices))
            self.assertFalse(set(train_indices) & set(val_indices))
            self.assertFalse(set(val_indices) & set(test_indices))
            self.assertEqual(split_indices(dataset.targets, 0)[2], test_indices)
            for indices, per_class in ((train_indices, 14), (val_indices, 3), (test_indices, 3)):
                self.assertEqual(np.bincount(np.asarray(dataset.targets)[indices]).tolist(), [per_class] * 5)
            checkpoint = root / "mlp.pt"
            history = root / "history.json"
            neural_report, classic_report = root / "mlp", root / "classical"
            run_main(train, "--data-dir", data, "--model", "mlp", "--image-size", 16,
                     "--hidden-dim", 16, "--epochs", 2, "--optimizer", "adam",
                     "--learning-rate", 0.001, "--device", "cpu", "--threads", 1,
                     "--checkpoint", checkpoint, "--history", history)
            saved = torch.load(checkpoint, weights_only=True)
            expected_mean = np.mean([np.asarray(dataset[i][0], dtype=float).mean((0, 1)) / 255
                                     for i in train_indices], axis=0)
            np.testing.assert_allclose(saved["normalization_stats"]["mean"], expected_mean, atol=1e-6)
            run_main(evaluate, "--data-dir", data, "--checkpoint", checkpoint,
                     "--history", history, "--output-dir", neural_report, "--benchmark-repeats", 3)
            analyze.analyze_run(neural_report, data, history)
            run_main(classical, "--data-dir", data, "--output-dir", classic_report,
                     "--checkpoint", root / "classical.joblib", "--c-values", 0.01, 0.1,
                     "--benchmark-repeats", 3)
            analyze.analyze_run(classic_report, data)
            classic = joblib.load(root / "classical.joblib")
            self.assertEqual(int(classic["pipeline"].named_steps["standardscaler"].n_samples_seen_), 70)
            rows = compare_reports([neural_report, classic_report], root / "comparison")
            self.assertEqual(len(rows), 2)
            for report in (neural_report, classic_report):
                with (report / "test_predictions.csv").open(newline="", encoding="utf-8") as file:
                    predictions = list(csv.DictReader(file))
                self.assertEqual(len(predictions), 15)
                expected_paths = [Path(dataset.samples[i][0]).relative_to(data).as_posix() for i in test_indices]
                self.assertEqual([row["image_path"] for row in predictions], expected_paths)
                for row in predictions:
                    total = sum(float(row[f"probability_{style}"]) for style in dataset.classes)
                    self.assertAlmostEqual(total, 1, places=6)
                self.assertTrue((report / "diagnostic.md").is_file())
            self.assertTrue((neural_report / "learning_curves.png").is_file())
            self.assertTrue((root / "comparison" / "accuracy_vs_cpu.png").is_file())
            metrics_file = classic_report / "test_metrics.json"
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            metrics["metadata"]["seed"] = 99
            metrics_file.write_text(json.dumps(metrics), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identical data"):
                compare_reports([neural_report, classic_report], root / "bad-comparison")
            original_signature = dataset_signature(dataset)
            changed_file = Path(dataset.samples[0][0])
            Image.new("RGB", (16, 16), color="white").save(changed_file)
            self.assertNotEqual(original_signature, dataset_signature(dataset))
            with self.assertRaisesRegex(ValueError, "Dataset contents changed"):
                run_main(evaluate, "--data-dir", data, "--checkpoint", checkpoint,
                         "--output-dir", root / "changed-data", "--benchmark-repeats", 1)


if __name__ == "__main__":
    unittest.main()
