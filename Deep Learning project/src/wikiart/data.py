"""ImageFolder loading with a reproducible stratified split and TP05-style normalization."""

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
    normalization: str = "half",
):
    """Return train, validation, test loaders and sorted ImageFolder class names.

    Splits are stratified 70/15/15. With normalization="train", RGB channel
    mean/std are computed from the training split only, like TP05, then reused
    for validation and test. Training images receive a horizontal flip.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"ImageFolder directory not found: {data_dir.resolve()}")

    index_dataset = datasets.ImageFolder(data_dir)
    if len(index_dataset.classes) < 2:
        raise ValueError(f"Expected at least two class folders in {data_dir}")

    indices = list(range(len(index_dataset.samples)))
    targets = index_dataset.targets
    train_val_indices, test_indices = train_test_split(
        indices, test_size=0.15, random_state=seed, stratify=targets
    )
    train_targets = [targets[i] for i in train_val_indices]
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=0.15 / 0.85,
        random_state=seed,
        stratify=train_targets,
    )

    if normalization == "train":
        stats_dataset = datasets.ImageFolder(
            data_dir,
            transform=transforms.Compose(
                [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
            ),
        )
        stats_loader = DataLoader(
            Subset(stats_dataset, train_indices),
            batch_size=64,
            shuffle=False,
            num_workers=num_workers,
        )
        channel_sum = torch.zeros(3, dtype=torch.float64)
        channel_square_sum = torch.zeros(3, dtype=torch.float64)
        pixel_count = 0
        for images, _ in stats_loader:
            images = images.to(torch.float64)
            channel_sum += images.sum(dim=(0, 2, 3))
            channel_square_sum += images.square().sum(dim=(0, 2, 3))
            pixel_count += images.shape[0] * images.shape[2] * images.shape[3]
        mean = channel_sum / pixel_count
        std = (channel_square_sum / pixel_count - mean.square()).clamp_min(1e-12).sqrt()
        normalize = transforms.Normalize(mean.tolist(), std.tolist())
    elif normalization == "half":
        normalize = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    elif normalization == "imagenet":
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )
    else:
        raise ValueError("normalization must be 'train', 'half', or 'imagenet'")

    if normalization == "imagenet":
        resize_crop = [
            transforms.Resize(int(round(image_size * 256 / 224))),
            transforms.CenterCrop(image_size),
        ]
    else:
        resize_crop = [transforms.Resize((image_size, image_size))]

    train_transform = transforms.Compose(
        [
            *resize_crop,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [*resize_crop, transforms.ToTensor(), normalize]
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
        Subset(train_dataset, train_indices),
        shuffle=True,
        generator=generator,
        **common,
    )
    val_loader = DataLoader(Subset(eval_dataset, val_indices), shuffle=False, **common)
    test_loader = DataLoader(Subset(eval_dataset, test_indices), shuffle=False, **common)
    return train_loader, val_loader, test_loader, index_dataset.classes
