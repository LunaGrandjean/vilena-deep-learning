# WikiArt style classification

Classify a painting image into one of five WikiArt styles. The local dataset is in
`data/wikiart_styles/`: 1,000 square 128×128 JPEGs, with 200 images in each class
(`Baroque`, `Cubism`, `Impressionism`, `Pop_Art`, and `Ukiyo_e`). The `data/`
directory is ignored by Git; use `make_wikiart_subset.py` and its recorded command to
rebuild the subset when needed.

## Project plan

The workflow follows the PyTorch steps from `../TP05_vision_project.ipynb`: fixed
random seed, train/validation/test split, mini-batch training, validation tracking,
one final test evaluation, and confusion-matrix/error review. TP05 uses a dense network
on 28×28 grayscale FashionMNIST; this project uses a compact CNN for 128×128 RGB art.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the proposed network and repository layout.

## Environment

Install dependencies from the project directory:

```powershell
python -m pip install -r requirements.txt
```
