"""Dense MLP, compact CNN and pretrained ResNet18 for WikiArt styles."""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


class WikiArtMLP(nn.Module):
    """Viviam's dense network: two independently configurable hidden layers."""

    def __init__(self, num_classes, image_size=64, hidden1=256, hidden2=128, dropout=0.0):
        super().__init__()
        if num_classes < 2 or image_size < 1 or hidden1 < 1 or hidden2 < 1:
            raise ValueError("Invalid MLP dimensions")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(3 * image_size * image_size, hidden1)
        self.layer2 = nn.Linear(hidden1, hidden2)
        self.output = nn.Linear(hidden2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # Preserve Viviam's initialization: He for hidden layers, PyTorch's
        # default initialization for the output layer.
        for layer in (self.layer1, self.layer2):
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)

    def forward(self, images):
        features = self.flatten(images)
        features = self.dropout(self.relu(self.layer1(features)))
        features = self.dropout(self.relu(self.layer2(features)))
        return self.output(features)


def load_model_state(model, state):
    """Keep Luna's historical MLP checkpoints readable after adopting Viv's names."""
    if isinstance(model, WikiArtMLP) and "layers.1.weight" in state:
        names = {"layers.1": "layer1", "layers.4": "layer2", "layers.7": "output"}
        state = {f"{names[key.rsplit('.', 1)[0]]}.{key.rsplit('.', 1)[1]}": value
                 for key, value in state.items()}
    model.load_state_dict(state)


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
    image_size: int = 128,
    hidden_dim: int = 256,
    hidden1: int | None = None,
    hidden2: int | None = None,
) -> nn.Module:
    """Output raw logits; CrossEntropyLoss applies log-softmax internally."""
    if architecture == "mlp":
        first = hidden_dim if hidden1 is None else hidden1
        second = first // 2 if hidden2 is None else hidden2
        return WikiArtMLP(num_classes, image_size, first, second, dropout)
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
