"""Compact CNN for the five-class WikiArt style subset."""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


class WikiArtCNN(nn.Module):
    """Three convolutional blocks followed by global pooling and a classifier."""

    def __init__(self, num_classes: int, dropout: float = 0.3, width: int = 1):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if width < 1:
            raise ValueError("width must be at least 1")

        channels = (32 * width, 64 * width, 128 * width)
        self.width = width
        self.features = nn.Sequential(
            _conv_block(3, channels[0]),
            _conv_block(channels[0], channels[1]),
            _conv_block(channels[1], channels[2]),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(channels[2], num_classes),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, images):
        features = self.features(images)
        pooled = self.pool(features)
        return self.classifier(pooled)


def build_model(
    architecture: str,
    num_classes: int,
    dropout: float = 0.3,
    width: int = 1,
    pretrained: bool = False,
) -> nn.Module:
    """Build the project CNN or a pretrained ResNet18 for WikiArt classes."""
    if architecture == "cnn":
        return WikiArtCNN(num_classes=num_classes, dropout=dropout, width=width)
    if architecture == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if pretrained:
            # Fine-tune the final residual block and new classifier; keep earlier
            # ImageNet features fixed for this small training set.
            for name, parameter in model.named_parameters():
                parameter.requires_grad = name.startswith(("layer4.", "fc."))
        return model
    raise ValueError(f"Unknown architecture: {architecture}")
