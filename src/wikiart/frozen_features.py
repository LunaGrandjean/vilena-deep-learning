"""Cache deterministic frozen features; keep random train-only horizontal flips."""

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


class FlipFeatures(Dataset):
    def __init__(self, original, flipped, labels):
        if original.shape != flipped.shape or len(original) != len(labels):
            raise ValueError("Mismatched original, flipped features and labels")
        self.original, self.flipped, self.labels = original, flipped, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        features = self.flipped if torch.rand(()).item() < 0.5 else self.original
        return features[index], self.labels[index]


@torch.no_grad()
def cache_feature_loaders(model, train_loader, val_loader, device, seed=0):
    """Never access the test subset. Input loaders must have no augmentation.

    A horizontal flip of a normalized tensor is equivalent to flipping the RGB
    image before normalization. Both orientations are computed only for train;
    each training access randomly chooses one. Validation always uses originals.
    Small extraction batches bound activation memory independently of head batch size.
    """
    model.eval()
    cached = []
    for name, source, flip in (("train", train_loader, True), ("validation", val_loader, False)):
        original, flipped, labels = [], [], []
        loader = DataLoader(source.dataset, batch_size=min(8, source.batch_size), shuffle=False)
        for step, (images, targets) in enumerate(loader, 1):
            images = images.to(device)
            original.append(model.forward_features(images).cpu())
            if flip:
                flipped.append(model.forward_features(images.flip(-1)).cpu())
            labels.append(targets)
            if step == 1 or step % 10 == 0 or step == len(loader):
                print(f"VGG16 frozen features: {name} batch {step}/{len(loader)}", flush=True)
        features, targets = torch.cat(original), torch.cat(labels)
        cached.append(FlipFeatures(features, torch.cat(flipped), targets) if flip
                      else TensorDataset(features, targets))
    generator = torch.Generator().manual_seed(seed)
    return (DataLoader(cached[0], batch_size=train_loader.batch_size, shuffle=True, generator=generator),
            DataLoader(cached[1], batch_size=val_loader.batch_size, shuffle=False))
