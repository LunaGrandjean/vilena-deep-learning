"""Local painting-style demo: python -m wikiart.app (http://127.0.0.1:8501)."""

import argparse
import csv
from io import BytesIO
from pathlib import Path
from threading import RLock
from time import perf_counter

from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image, ImageOps, UnidentifiedImageError
import joblib
import torch
from threadpoolctl import threadpool_limits
from werkzeug.exceptions import RequestEntityTooLarge

from .data import DEFAULT_DATA_DIR, PROJECT_DIR
from .classical import image_features
from .predict import load_model, prepare_painting, preprocess_image

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
STYLE_LABELS = {"Baroque": "Baroque", "Cubism": "Cubism", "Impressionism": "Impressionism",
                "Pop_Art": "Pop Art", "Ukiyo_e": "Ukiyo-e"}
MODEL_NAMES = {"resnet18": "ResNet18", "vgg16": "VGG16", "cnn": "Small CNN",
               "mlp": "MLP (dropout 0.5)", "mlp_baseline": "MLP (no dropout)",
               "classical": "Logistic regression"}
MODEL_HINTS = {"resnet18": "Transfer learning · recommended", "vgg16": "Transfer learning · larger network",
               "cnn": "Convolutions · trained from scratch", "mlp": "Dense network · regularised",
               "mlp_baseline": "Dense network · baseline", "classical": "RGB features · classical method"}
DEFAULT_CHECKPOINTS = {
    "resnet18": PROJECT_DIR / "checkpoints/comparaison_v1/resnet18/best_model.pt",
    "vgg16": PROJECT_DIR / "checkpoints/comparaison_vgg16/vgg16/best_model.pt",
    "cnn": PROJECT_DIR / "checkpoints/comparaison_v1/cnn/best_model.pt",
    "mlp": PROJECT_DIR / "checkpoints/mlp_runs/64px_h256-128_dropout0.5_wd0.0/best_model.pt",
    "mlp_baseline": PROJECT_DIR / "checkpoints/mlp_runs/64px_h256-128_dropout0.0_wd0.0/best_model.pt",
    "classical": PROJECT_DIR / "checkpoints/comparaison_v1/classical/model.joblib",
}


def decode_image(payload):
    if not payload:
        raise ValueError("Choose a painting image first.")
    if len(payload) > MAX_IMAGE_BYTES:
        raise ValueError("This image is too large. Choose a file smaller than 10 MB.")
    try:
        with Image.open(BytesIO(payload)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Choose a JPG, PNG or WebP image.")
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError("This image is too large. Resize it to 25 megapixels or less.")
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("We couldn't read this image. Try another JPG, PNG or WebP.") from error


class DemoPredictor:
    """Reuse local checkpoints and serialize inference on the single CPU thread."""

    def __init__(self, checkpoints):
        self.checkpoints = {key: Path(value) for key, value in checkpoints.items()}
        self.models = {}
        self.lock = RLock()

    def predict(self, image, model_id):
        with self.lock:
            if model_id not in self.models:
                if model_id == "classical":
                    checkpoint = joblib.load(self.checkpoints[model_id])
                    if checkpoint.get("features") != "rgb16_hist32_v1":
                        raise ValueError("Unsupported classical feature representation")
                    self.models[model_id] = (checkpoint["pipeline"], checkpoint)
                else:
                    self.models[model_id] = load_model(self.checkpoints[model_id])
            model, checkpoint = self.models[model_id]
            if model_id == "classical":
                with threadpool_limits(limits=1):
                    start = perf_counter()
                    features = image_features(prepare_painting(image)).reshape(1, -1)
                    scores = model.predict_proba(features)[0].tolist()
                    elapsed = (perf_counter() - start) * 1000
                classes = [checkpoint["classes"][int(index)] for index in model.classes_]
            else:
                start = perf_counter()
                with torch.inference_mode():
                    scores = torch.softmax(model(preprocess_image(image, checkpoint)), dim=1)[0].tolist()
                elapsed = (perf_counter() - start) * 1000
                classes = checkpoint["classes"]
            probabilities = sorted([
                {"style": name, "label": STYLE_LABELS.get(name, name), "probability": float(score)}
                for name, score in zip(classes, scores)
            ], key=lambda item: item["probability"], reverse=True)
            return {"model": MODEL_NAMES[model_id], "model_id": model_id,
                    "predicted_style": probabilities[0]["label"],
                    "probabilities": probabilities, "elapsed_ms": round(elapsed, 1)}


def find_examples(data_dir, predictions_csv):
    """First held-out image per style, without selecting for correct predictions."""
    root = Path(data_dir).resolve()
    if not Path(predictions_csv).is_file():
        return []
    by_style = {}
    with Path(predictions_csv).open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            style = row.get("true_style")
            path = (root / row.get("image_path", "")).resolve()
            if style in STYLE_LABELS and style not in by_style and path.is_relative_to(root) and path.is_file():
                by_style[style] = {"label": STYLE_LABELS[style], "path": path}
    return [by_style[style] for style in STYLE_LABELS if style in by_style]


def create_app(checkpoints=None, data_dir=DEFAULT_DATA_DIR, predictions_csv=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES
    torch.set_num_threads(1)
    checkpoints = DEFAULT_CHECKPOINTS if checkpoints is None else checkpoints
    predictor = DemoPredictor(checkpoints)
    examples = find_examples(data_dir, predictions_csv or PROJECT_DIR / "reports/comparaison_v1/resnet18/test_predictions.csv")

    @app.get("/")
    def index():
        return render_template("demo.html")

    @app.get("/api/config")
    def config():
        return jsonify({
            "models": [{"id": key, "name": label,
                        "description": MODEL_HINTS[key],
                        "available": key in predictor.checkpoints and predictor.checkpoints[key].is_file()}
                       for key, label in MODEL_NAMES.items()],
            "examples": [{"id": index, "label": example["label"], "url": f"/api/examples/{index}"}
                         for index, example in enumerate(examples)],
        })

    @app.get("/api/examples/<int:index>")
    def example_image(index):
        if index >= len(examples):
            return jsonify(error="This example is not available."), 404
        return send_file(examples[index]["path"], max_age=3600)

    @app.post("/api/predict")
    def predict():
        model_id = request.args.get("model", "resnet18")
        if model_id not in MODEL_NAMES and model_id != "all":
            return jsonify(error="Choose one of the available models."), 400
        available = [key for key in MODEL_NAMES if key in predictor.checkpoints and predictor.checkpoints[key].is_file()]
        if model_id == "all" and not available:
            return jsonify(error="No model is available on this machine."), 503
        if model_id != "all" and model_id not in available:
            return jsonify(error=f"{MODEL_NAMES[model_id]} is not available on this machine."), 503
        try:
            # A raw binary body keeps uploads in memory rather than a multipart
            # temporary file. No uploaded image is saved or sent to a cloud API.
            image = decode_image(request.get_data(cache=False))
        except ValueError as error:
            return jsonify(error=str(error)), 400
        if model_id == "all":
            results, errors = [], []
            for key in available:
                try:
                    results.append(predictor.predict(image, key))
                except Exception:
                    app.logger.exception("Local checkpoint prediction failed: %s", key)
                    errors.append({"model_id": key, "model": MODEL_NAMES[key]})
            if not results:
                return jsonify(error="None of the available models could analyse this image."), 503
            result = {"results": results, "errors": errors,
                      "unavailable": [MODEL_NAMES[key] for key in MODEL_NAMES if key not in available]}
        else:
            try:
                result = predictor.predict(image, model_id)
            except Exception:
                app.logger.exception("Local checkpoint prediction failed: %s", model_id)
                return jsonify(error="The model couldn't analyse this image. Check the local model and try again."), 503
        response = jsonify(result)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(error):
        return jsonify(error="This image is too large. Choose a file smaller than 10 MB."), 413

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535")
    app = create_app()
    print(f"WikiArt demo: http://127.0.0.1:{args.port}", flush=True)
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
