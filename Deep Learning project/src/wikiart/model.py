"""Compact CNN for the five-class WikiArt style subset."""

import torch.nn as nn


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


class WikiArtCNN(nn.Module):
    """Three convolutional blocks followed by global pooling and a classifier."""

    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")

        self.features = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
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

##Feed-forward network (MLP): flatten the image, then fully connected layers
class WikiArtMLP(nn.Module):

    def __init__(self, num_classes: int, image_size: int = 64, hidden1: int = 256, hidden2: int = 128, dropout: float = 0.0):
        super().__init__()
        n_inputs = 3 * image_size * image_size   # 3 colors x 64 x 64 = 12,288 numbers per image

        self.flatten = nn.Flatten()                     # (3, 64, 64) -> one vector of 12,288
        self.layer1 = nn.Linear(n_inputs, hidden1)      # 12,288 -> 256
        self.layer2 = nn.Linear(hidden1, hidden2)       # 256 -> 128
        self.output = nn.Linear(hidden2, num_classes)   # 128 -> 5 logits (one per style)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)   #during training, each hidden neuron is switched off with probability p

        # He initialization for the ReLU layers (keeps signals stable from layer to layer)
        for layer in [self.layer1, self.layer2]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def forward(self, images):
        x = self.flatten(images)
        x = self.dropout(self.relu(self.layer1(x)))   # A1 = ReLU(Z1), then dropout
        x = self.dropout(self.relu(self.layer2(x)))   # A2 = ReLU(Z2), then dropout
        return self.output(x)           # raw logits 