##Demo: give a painting image, get the predicted style and the confidence (top 3)

import argparse

import torch
from PIL import Image, ImageOps
from torchvision import transforms

from .model import build_model, load_model_state

DATASET_SIZE = 128   # make_wikiart_subset.py stored every training image as a 128x128 square


def load_model(checkpoint_path):
    """Rebuild the network from the checkpoint, exactly like evaluate.py (S4: load_state_dict, eval)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(architecture=checkpoint["architecture"],
                        num_classes=checkpoint["num_classes"],
                        dropout=checkpoint.get("dropout", 0.3),
                        width=checkpoint.get("width", 1),
                        pretrained=False,                       # the weights come from the file
                        image_size=checkpoint["image_size"],
                        hidden_dim=checkpoint.get("hidden_dim", 256),
                        hidden1=checkpoint.get("hidden1"),
                        hidden2=checkpoint.get("hidden2")).eval()
    load_model_state(model, checkpoint["model_state_dict"])
    return model, checkpoint


def prepare_painting(image):
    """Match the square 128px base images while respecting camera orientation."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = transforms.CenterCrop(min(image.size))(image)             # 1. square, like the dataset
    return image.resize((DATASET_SIZE, DATASET_SIZE))                  # 2. same resolution as training


def preprocess_image(image, checkpoint):
    """Reuse each neural checkpoint's input size and training normalization."""
    size = checkpoint["image_size"]
    normalization = checkpoint.get("normalization", "half")
    stats = checkpoint.get("normalization_stats")
    if stats is None:
        if normalization == "imagenet":
            stats = {"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)}
        elif normalization == "half":
            stats = {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}
        else:
            raise ValueError("The checkpoint is missing its training normalization statistics")
    resize = ([transforms.Resize(round(size * 256 / 224)), transforms.CenterCrop(size)]
              if normalization == "imagenet" else [transforms.Resize((size, size))])
    evaluation = transforms.Compose([*resize, transforms.ToTensor(), transforms.Normalize(**stats)])
    image = prepare_painting(image)
    return evaluation(image).unsqueeze(0)                             # (1, 3, 224, 224): a batch of 1


def preprocess(image_path, checkpoint):
    """Turn any photo into the same kind of tensor the model saw during evaluation."""
    with Image.open(image_path) as image:
        return preprocess_image(image, checkpoint)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Path to a painting image (jpg, png...)")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained neural .pt checkpoint")
    parser.add_argument("--top", type=int, default=3, help="How many styles to show")
    args = parser.parse_args()

    model, checkpoint = load_model(args.checkpoint)
    x = preprocess(args.image, checkpoint)
    with torch.no_grad():                                             # prediction only, no gradients
        probabilities = torch.softmax(model(x), dim=1)[0]             # logits -> probabilities 
    values, indices = probabilities.topk(args.top)                    # the most likely styles

    print(f"Image: {args.image}")
    for p, i in zip(values, indices):
        print(f"  {checkpoint['classes'][i]:<14} {p.item():.2%}")    # 2 decimals, e.g. 99.64%


if __name__ == "__main__":
    main()
