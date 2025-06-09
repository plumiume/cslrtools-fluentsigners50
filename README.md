# cslrtools-fluentsigners50

A toolkit for handling the FluentSigners50 dataset for PyTorch/Lightning-based continuous sign language recognition (CSLR) tasks.

## Features

- Efficient loading of FluentSigners50 landmark and annotation data
- PyTorch `Dataset` and Lightning `DataModule` support
- Fast data loading with parallel processing
- Supports both gloss and translation annotations
- Dataset saving and reloading

## Installation

```sh
pip install .
# For Pytorch-Lightning support
pip install .[lightning]
```

## Usage

### Command Line

```sh
python -m cslrtools_fluentsigners50 <dataset_root> <landmarks_dir> <output_pickle_file>
```

### Python API

```python
from cslrtools_fluentsigners50 import FluentSigners50

# Create dataset
dataset = FluentSigners50(
    root="FluentSigners50",
    landmarks="landmarks",
    use_translation=False  # Use gloss annotation
)

# Use as PyTorch Dataset
torch_dataset = dataset.dataset

# Save and load
dataset.save("dataset.pkl")
loaded = FluentSigners50.load("dataset.pkl")
```

## Dependencies

- Python >= 3.11
- torch >= 2.0.0
- numpy >= 2.3.0
- tqdm
- parse
- halo
- cslrtools
- lightning.pytorch (optional)

## License

[Apache License 2.0](./LICENSE)

---

This toolkit is intended for research purposes. For questions or bug reports, please contact [ikeda@lee-lab.org](mailto:ikeda@lee-lab.org).
