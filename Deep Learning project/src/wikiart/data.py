"""ImageFolder loading and reproducible stratified data splits."""

from pathlib import Path

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def create_dataloaders(
    data_dir: str | Path = "data/wikiart_styles",
    batch_size: int = 32,
    image_size: int = 128,
    seed: int = 0,
    num_workers: int = 0,
):
    """Return train, validation, test loaders and sorted ImageFolder class names.

    The split is stratified 70/15/15. Training images get a horizontal flip;
    validation and test images use deterministic transforms only.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"ImageFolder directory not found: {data_dir.resolve()}")

    normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor(), normalize]
    )

    # Use one transform-free dataset to get stable targets and indices.
    index_dataset = datasets.ImageFolder(data_dir)
    if len(index_dataset.classes) < 2:
        raise ValueError(f"Expected at least two class folders in {data_dir}")
    indices = list(range(len(index_dataset.samples)))
    targets = index_dataset.targets

    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=0.15,
        random_state=seed,
        stratify=targets,
    )
    train_targets = [targets[i] for i in train_val_indices]
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=0.15 / 0.85,
        random_state=seed,
        stratify=train_targets,
    )

    train_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    eval_dataset = datasets.ImageFolder(data_dir, transform=eval_transform)
    if train_dataset.class_to_idx != index_dataset.class_to_idx:
        raise RuntimeError("Class folder mapping changed while preparing the data")

    generator = torch.Generator().manual_seed(seed)
    pin_memory = torch.cuda.is_available()
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    train_loader = DataLoader(
        Subset(train_dataset, train_indices), shuffle=True, generator=generator, **common
    )
    val_loader = DataLoader(Subset(eval_dataset, val_indices), shuffle=False, **common)
    test_loader = DataLoader(Subset(eval_dataset, test_indices), shuffle=False, **common)
    return train_loader, val_loader, test_loader, index_dataset.classes
