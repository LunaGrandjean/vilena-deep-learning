# Project architecture

The current dataset contains **1,500 images (300 per style)** under
`data/wikiart_style5/`, with a 1,050/225/225 train/validation/test split.
The [README](README.md) describes the current commands and full comparison.

## Additional models and analysis

- `WikiArtMLP`: Viviam's independently configurable hidden layers, by default
  256 and 128 units. The integrated runner selects 64x64 RGB, Dropout(0.5),
  AdamW with zero weight decay and fixed 0.5 RGB normalization. Hidden-layer
  initialization follows Viviam's implementation. Historical Luna 32px
  checkpoints remain readable, and reports/comparaison_v1 remains unchanged.
- `train_mlp.py`: Viviam's experiment settings, validation summaries and annotated
  curves, using the common checkpoints, split, CPU evaluation and error reports.
  Original Viviam reports are archived unchanged in reports/mlp/.
  All MLP results share that root: new runs in reports/mlp/runs/ and historical
  Luna 32px results in reports/mlp/archives/luna_32px/.
- `compare_mlp.py`: Viviam's seven-run regularization comparison, with table and
  combined curves. Explicit run directories are supported; mismatched protocols
  and mixtures of historical/integrated reports are rejected. Her historical
  dropout-0.5 model achieves 36.0% test accuracy on the older 1049/226/225 split.
- ResNet18: ImageNet initialization, 224x224 input, fine-tuned last residual
  block and classifier; earlier weights and BatchNorm statistics remain fixed.
- Classical reference: standardized 16x16 RGB pixels and color histograms,
  logistic regression with regularization selected on validation.
- `analyze.py`: learning curves, diagnostic heuristics and error galleries.
- `compare.py`: quality, CPU latency, training duration and model file size.
- `run_experiments.py`: run each model, analyze it, then compare the reports.

All models share the same stratified split function. Dataset contents and test
samples have signatures. Normalization and scaling are fitted only on train.

## Image classifier

Input images are RGB paintings of shape `3 × 128 × 128`. Use a small CNN with ReLU
hidden activations and He/Kaiming initialization, applying the activation and
initialization lessons from TP05:

```text
RGB image (3 × 128 × 128)
  → Conv 3×3, 32 channels → BatchNorm → ReLU → MaxPool 2×2
  → Conv 3×3, 64 channels → BatchNorm → ReLU → MaxPool 2×2
  → Conv 3×3, 128 channels → BatchNorm → ReLU → MaxPool 2×2
  → Adaptive average pool (1 × 1)
  → Flatten → Dropout(0.3) → Linear(128, 5) → class logits
```

Train with cross-entropy on the five logits. Keep the final layer unactivated; as in
TP05, `CrossEntropyLoss` handles the logits internally. Start with Adam (`lr=1e-3`),
then compare validation results before tuning model depth, activation, or regularization.

## Data and evaluation flow

1. Load `data/wikiart_style5/` with `torchvision.datasets.ImageFolder`; its sorted class
   folders define the labels.
2. Create a reproducible, stratified 70/15/15 train/validation/test split. With 300
   images per class, this gives 210/45/45 per style. Apply any augmentation to training
   images only.
3. Train in mini-batches. Record train loss, validation loss, and validation accuracy
   each epoch, following TP05's `fit` workflow. Save the checkpoint with the best
   validation loss under `checkpoints/`.
4. Evaluate the selected model once on the held-out test split. Report accuracy and a
   confusion matrix, then inspect misclassified examples as in TP05.
5. Record the random seed, split method, model settings, and final metrics in the report.

The subset has a `metadata.csv` with artist information. Check artist overlap between
splits when discussing generalization: images by the same named painter can otherwise
appear in both training and test data. `Unknown Artist` is a catch-all label in the
source metadata, so it cannot be treated as one actual painter for grouping.

## Repository layout

```text
final-project/
├── .gitignore                 # excludes local data and training artifacts
├── ARCHITECTURE.md             # model and workflow plan
├── README.md
├── requirements.txt
├── pyproject.toml              # install the src/ package in editable mode
├── graphs-slides/              # presentation plots in burgundy and olive
├── data/                       # local, ignored
│   └── wikiart_style5/         # 5 classes × 300 images, metadata, subset info
├── notebooks/                  # exploration and course experiments
├── src/wikiart/                # models, training, evaluation, analysis, comparison
├── checkpoints/                # saved weights, ignored
└── reports/                    # metrics, plots, diagnostics and CPU comparison
```

The course notebook is stored at `TP05_vision_project.ipynb`; it is the reference for the training and evaluation
workflow, not a model to apply unchanged to the RGB paintings.
