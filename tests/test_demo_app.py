"""Exercise real image decoding, preprocessing and inference through the local API."""

import csv
from io import BytesIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wikiart import app as demo
from wikiart.predict import load_model, prepare_painting, preprocess, preprocess_image
from wikiart.classical import image_features
from wikiart.model import build_model


def image_bytes(format="PNG"):
    stream = BytesIO()
    Image.new("RGB", (20, 12), (160, 60, 30)).save(stream, format=format)
    return stream.getvalue()


class DemoAppTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="wikiart-demo-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoints = {key: self.root / f"{key}.pt" for key in demo.MODEL_NAMES}
        for path in self.checkpoints.values():
            path.write_bytes(b"local test checkpoint")
        data = self.root / "images"
        for label in ("Baroque", "Cubism"):
            folder = data / label
            folder.mkdir(parents=True)
            (folder / "test.jpg").write_bytes(image_bytes("JPEG"))
        self.csv = self.root / "predictions.csv"
        with self.csv.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["image_path", "true_style", "predicted_style"])
            writer.writerow(["Baroque/test.jpg", "Baroque", "Cubism"])
            writer.writerow(["Cubism/test.jpg", "Cubism", "Baroque"])
            writer.writerow(["../../outside.jpg", "Pop_Art", "Pop_Art"])
        self.app = demo.create_app(self.checkpoints, data, self.csv)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        self.meta = {"normalization": "imagenet", "image_size": 224,
                     "classes": list(demo.STYLE_LABELS)}

    def test_page_assets_and_unfiltered_test_examples(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        for asset in ("/static/demo.css", "/static/demo.js"):
            with self.client.get(asset) as response:
                self.assertEqual(response.status_code, 200)
        config = self.client.get("/api/config").json
        self.assertEqual([item["label"] for item in config["examples"]], ["Baroque", "Cubism"])
        self.assertTrue(all(item["available"] for item in config["models"]))
        with self.client.get(config["examples"][0]["url"]) as response:
            self.assertEqual(response.data, image_bytes("JPEG"))
        self.assertEqual(self.client.get("/api/examples/99").status_code, 404)

    def test_upload_uses_shared_preprocessing_and_caches_each_model(self):
        torch.manual_seed(0)
        model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 5)).eval()
        payload = image_bytes()
        with torch.inference_mode():
            expected = torch.softmax(model(preprocess_image(Image.open(BytesIO(payload)), self.meta)), dim=1)[0]
        before = set(self.root.rglob("*"))
        with patch.object(demo, "load_model", return_value=(model, self.meta)) as loader:
            for model_id in ("resnet18", "resnet18", "vgg16"):
                response = self.client.post(f"/api/predict?model={model_id}", data=payload, content_type="image/png")
                self.assertEqual(response.status_code, 200, response.json)
                result = response.json
                self.assertEqual(result["model_id"], model_id)
                probabilities = result["probabilities"]
                self.assertEqual(len(probabilities), 5)
                self.assertAlmostEqual(sum(item["probability"] for item in probabilities), 1, places=6)
                self.assertEqual(result["predicted_style"], probabilities[0]["label"])
                self.assertEqual([item["probability"] for item in probabilities], sorted(expected.tolist(), reverse=True))
                self.assertEqual(response.headers["Cache-Control"], "no-store")
            self.assertEqual(loader.call_count, 2)
        self.assertEqual(set(self.root.rglob("*")), before)

    def test_invalid_uploads_and_models_return_readable_errors(self):
        for payload in (b"", b"this is not an image", image_bytes("GIF")):
            response = self.client.post("/api/predict", data=payload)
            self.assertEqual(response.status_code, 400)
            self.assertIsInstance(response.json["error"], str)
        response = self.client.post("/api/predict?model=unknown", data=image_bytes())
        self.assertEqual(response.status_code, 400)
        with patch.object(demo, "MAX_IMAGE_PIXELS", 10):
            self.assertEqual(self.client.post("/api/predict", data=image_bytes()).status_code, 400)
        self.app.config["MAX_CONTENT_LENGTH"] = 16
        response = self.client.post("/api/predict", data=b"x" * 17)
        self.assertEqual(response.status_code, 413)
        self.assertIn("10 MB", response.json["error"])

    def test_missing_weights_disable_only_the_missing_model(self):
        self.checkpoints["vgg16"].unlink()
        config = self.client.get("/api/config").json
        available = {item["id"]: item["available"] for item in config["models"]}
        self.assertEqual(available, {key: key != "vgg16" for key in demo.MODEL_NAMES})
        response = self.client.post("/api/predict?model=vgg16", data=image_bytes())
        self.assertEqual(response.status_code, 503)
        self.assertIn("VGG16", response.json["error"])

    def test_classical_route_uses_saved_scaler_features_and_class_mapping(self):
        rng = np.random.default_rng(4)
        pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        pipeline.fit(rng.normal(size=(25, 864)), np.tile(np.arange(5), 5))
        checkpoint = {"pipeline": pipeline, "classes": list(demo.STYLE_LABELS),
                      "features": "rgb16_hist32_v1"}
        joblib.dump(checkpoint, self.checkpoints["classical"])
        payload = image_bytes()
        features = image_features(prepare_painting(demo.decode_image(payload))).reshape(1, -1)
        expected = pipeline.predict_proba(features)[0]
        response = self.client.post("/api/predict?model=classical", data=payload)
        self.assertEqual(response.status_code, 200, response.json)
        actual = {item["style"]: item["probability"] for item in response.json["probabilities"]}
        np.testing.assert_allclose([actual[name] for name in checkpoint["classes"]], expected, rtol=1e-12)

    def test_neural_loader_and_preprocessing_preserve_mlp_and_cnn_settings(self):
        for architecture, size, normalization in (("mlp", 16, "half"), ("cnn", 32, "train")):
            with self.subTest(architecture=architecture):
                original = build_model(architecture, 5, image_size=size, hidden1=11, hidden2=7,
                                       dropout=0.5, width=2).eval()
                stats = {"mean": [0.3, 0.4, 0.5], "std": [0.2, 0.3, 0.4]}
                checkpoint = {"architecture": architecture, "num_classes": 5,
                              "image_size": size, "hidden1": 11, "hidden2": 7,
                              "dropout": 0.5, "width": 2, "normalization": normalization,
                              "normalization_stats": stats, "model_state_dict": original.state_dict()}
                path = self.root / f"{architecture}_real.pt"
                torch.save(checkpoint, path)
                restored, metadata = load_model(path)
                image = demo.decode_image(image_bytes())
                tensor = preprocess_image(image, metadata)
                expected = transforms.Compose([
                    transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(**stats)
                ])(prepare_painting(image)).unsqueeze(0)
                torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
                with torch.no_grad():
                    torch.testing.assert_close(restored(tensor), original(tensor), rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "normalization statistics"):
            preprocess_image(image, {"image_size": 32, "normalization": "train"})

    def test_comparison_handles_missing_and_failed_models_without_losing_results(self):
        def prediction(image, key):
            return {"model_id": key, "model": demo.MODEL_NAMES[key], "predicted_style": "Baroque",
                    "probabilities": [{"style": "Baroque", "label": "Baroque", "probability": 1.0}],
                    "elapsed_ms": 1.0}
        with patch.object(demo.DemoPredictor, "predict", side_effect=prediction) as predict:
            response = self.client.post("/api/predict?model=all", data=image_bytes())
            self.assertEqual(response.status_code, 200)
            self.assertEqual([item["model_id"] for item in response.json["results"]], list(demo.MODEL_NAMES))
            self.assertEqual(predict.call_count, 6)
            self.assertEqual(response.json["errors"], [])
            self.assertEqual(response.json["unavailable"], [])
        self.checkpoints["vgg16"].unlink()
        def partial_prediction(image, key):
            if key == "cnn":
                raise RuntimeError("Broken test checkpoint")
            return prediction(image, key)
        with patch.object(demo.DemoPredictor, "predict", side_effect=partial_prediction), \
                self.assertLogs(self.app.logger, level="ERROR"):
            response = self.client.post("/api/predict?model=all", data=image_bytes())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json["results"]), 4)
        self.assertEqual(response.json["unavailable"], ["VGG16"])
        self.assertEqual(response.json["errors"], [{"model_id": "cnn", "model": "Small CNN"}])

    def test_phone_orientation_and_path_api_remain_consistent(self):
        image = Image.new("RGB", (20, 12), (40, 60, 100))
        exif = Image.Exif()
        exif[274] = 6
        stream = BytesIO()
        image.save(stream, format="JPEG", exif=exif)
        payload = stream.getvalue()
        self.assertEqual(demo.decode_image(payload).size, (12, 20))
        path = self.root / "phone.jpg"
        path.write_bytes(payload)
        from_memory = preprocess_image(demo.decode_image(payload), self.meta)
        from_path = preprocess(path, self.meta)
        torch.testing.assert_close(from_memory, from_path, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
