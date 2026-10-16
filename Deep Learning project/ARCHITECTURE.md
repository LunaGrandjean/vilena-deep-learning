# Project architecture

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

1. Load `data/wikiart_styles/` with `torchvision.datasets.ImageFolder`; its sorted class
   folders define the labels.
2. Create a reproducible, stratified 70/15/15 train/validation/test split. With 200
   images per class, this gives 140/30/30 per style. Apply any augmentation to training
   images only.
3. Train in mini-batches. Record train loss, validation loss, and validation accuracy
   each epoch, following TP05's `fit` workflow. Save the checkpoint with the best
   validation score under `checkpoints/`.
4. Evaluate the selected model once on the held-out test split. Report accuracy and a
   confusion matrix, then inspect misclassified examples as in TP05.
5. Record the random seed, split method, model settings, and final metrics in the report.

The subset has a `metadata.csv` with artist information. Check artist overlap between
splits when discussing generalization: images by the same named painter can otherwise
appear in both training and test data. `Unknown Artist` is a catch-all label in the
source metadata, so it cannot be treated as one actual painter for grouping.

## Repository layout

```text
Deep Learning project/
├── .gitignore                 # excludes local data and training artifacts
├── ARCHITECTURE.md             # model and workflow plan
├── README.md
├── requirements.txt
├── make_wikiart_subset.py      # reproducible dataset preparation
├── data/                       # local, ignored
│   └── wikiart_styles/         # 5 classes × 200 images, metadata, subset info
├── notebooks/                  # exploration and course experiments
├── src/wikiart/                # data loading, model, training, evaluation modules
├── checkpoints/                # saved weights, ignored
└── reports/figures/            # plots and confusion matrices
```

The course notebook remains one directory above this project at
`../TP05_vision_project.ipynb`; it is the reference for the training and evaluation
workflow, not a model to apply unchanged to the RGB paintings.
