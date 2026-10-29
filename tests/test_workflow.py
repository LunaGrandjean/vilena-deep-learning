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
from wikiart.compare_mlp import describe, load_runs
from wikiart.data import dataset_signature, split_indices
from wikiart.frozen_features import cache_feature_loaders
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

    def test_vgg16_frozen_features_checkpoint_and_full_image_evaluation(self):
        # A small stand-in exercises the same VGG wrapper and cache mechanics
        # without downloading ImageNet weights or writing a 537 MB checkpoint.
        def tiny_vgg(**kwargs):
            base = torch.nn.Module()
            base.features = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.ReLU())
            base.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
            base.classifier = torch.nn.Sequential(
                torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Dropout(0.5),
                torch.nn.Linear(8, 7), torch.nn.ReLU(), torch.nn.Dropout(0.5),
                torch.nn.Linear(7, 1000))
            return base

        with tempfile.TemporaryDirectory(prefix="wikiart-vgg-test-") as temp, \
                patch("wikiart.model.vgg16", side_effect=tiny_vgg):
            root = Path(temp)
            data = root / "images"
            make_images(data)
            from wikiart.data import create_dataloaders
            loaders = create_dataloaders(data, batch_size=8, image_size=224,
                                         normalization="imagenet", augment=False)
            model = build_model("vgg16", 5, pretrained=True)
            model.train()
            self.assertFalse(any(layer.training for layer in model.modules()
                                 if isinstance(layer, torch.nn.Dropout)))
            self.assertEqual([name for name, p in model.named_parameters() if p.requires_grad],
                             ["classifier.6.weight", "classifier.6.bias"])
            with redirect_stdout(io.StringIO()):
                cached_train, cached_val = cache_feature_loaders(model, *loaders[:2], "cpu")
            images, _ = next(iter(torch.utils.data.DataLoader(loaders[0].dataset, batch_size=8)))
            with torch.no_grad():
                torch.testing.assert_close(cached_train.dataset.original[:8], model.forward_features(images))
                torch.testing.assert_close(cached_train.dataset.flipped[:8], model.forward_features(images.flip(-1)))
                torch.testing.assert_close(model.head(model.forward_features(images)), model(images))
            self.assertEqual(len(cached_train.dataset), 70)
            self.assertEqual(len(cached_val.dataset), 15)
            checkpoint, history = root / "best.pt", root / "history.json"
            created = []
            def track_model(*args, **kwargs):
                network = build_model(*args, **kwargs)
                created.append((network, {name: p.detach().clone()
                                         for name, p in network.named_parameters()}))
                return network
            with patch.object(train, "build_model", side_effect=track_model):
                run_main(train, "--model", "vgg16", "--data-dir", data,
                         "--image-size", 224, "--normalization", "imagenet", "--device", "cpu",
                         "--epochs", 2, "--batch-size", 8, "--optimizer", "adam",
                         "--learning-rate", 0.001, "--checkpoint", checkpoint, "--history", history)
            saved = torch.load(checkpoint, weights_only=True)
            restored = build_model("vgg16", 5).eval()
            load_model_state(restored, saved["model_state_dict"])
            self.assertEqual(saved["transfer_mode"], "frozen_features_linear_head")
            trace = json.loads(history.read_text())
            self.assertEqual(saved["best_epoch"], int(np.argmin(trace["val_loss"])) + 1)
            val_loss, _ = train.evaluate(restored, loaders[1], torch.nn.CrossEntropyLoss(), "cpu")
            self.assertAlmostEqual(val_loss, saved["best_val_loss"], places=5)
            trained, before = created[0]
            for name, parameter in trained.named_parameters():
                if not parameter.requires_grad:
                    torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
            self.assertFalse(torch.equal(trained.head.weight, before["classifier.6.weight"]))
            report = root / "report"
            run_main(evaluate, "--data-dir", data, "--checkpoint", checkpoint,
                     "--history", history, "--output-dir", report, "--benchmark-repeats", 3)
            result = json.loads((report / "test_metrics.json").read_text())
            self.assertEqual(result["metadata"]["model"], "vgg16")
            self.assertEqual(result["metadata"]["test_count"], 15)
            analyze.analyze_run(report, data, history)
            self.assertIn("VGG16", (report / "diagnostic.md").read_text(encoding="utf-8"))
            from wikiart.predict import load_model as load_demo_model
            demo_model, _ = load_demo_model(checkpoint)
            with torch.no_grad():
                torch.testing.assert_close(demo_model(images), restored(images), rtol=0, atol=0)

    def test_mlp_study_refuses_mixed_historical_and_common_protocols(self):
        with tempfile.TemporaryDirectory(prefix="wikiart-study-test-") as temp:
            root = Path(temp)
            folders = [root / "historical", root / "integrated"]
            history = {"train_loss": [2.0, 1.0], "val_loss": [2.0, 1.5],
                       "train_acc": [0.2, 0.7], "val_acc": [0.2, 0.4]}
            for folder, dropout in zip(folders, ("0.0", "0.5")):
                folder.mkdir()
                name = f"64px_h256-128_dropout{dropout}_wd0.0"
                summary = {"run": name, "best_epoch": 2}
                if folder.name == "integrated":
                    summary["training"] = {"dataset_signature": "synthetic-data", "seed": 0,
                                           "split_counts": {"train": 70, "validation": 15, "test": 15},
                                           "normalization": "half"}
                (folder / f"summary_{name}.json").write_text(json.dumps(summary), encoding="utf-8")
                (folder / f"history_{name}.json").write_text(json.dumps(history), encoding="utf-8")
            self.assertEqual(len(load_runs([folders[0]])), 1)
            self.assertEqual(len(load_runs([folders[1]])), 1)
            with self.assertRaisesRegex(ValueError, "mix historical"):
                load_runs(folders)
            self.assertIn("seed 1", describe("64px_h256-128_dropout0.5_wd1e-05_seed1"))

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
