# Multi-Scale CNN for MNIST Classification

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

A PyTorch implementation of a multi-scale convolutional neural network (CNN) for digit classification on the MNIST dataset.

## Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Multi-scale feature extraction**: Parallel convolutional branches with 1x1, 3x3, and 5x5 kernels
- **Advanced architecture**: Includes residual connections and batch normalization
- **Data augmentation**: Random rotations and translations for improved generalization
- **Training utilities**: Learning rate scheduling, gradient clipping, and early stopping
- **Reproducibility**: Seed configuration for deterministic training
- **Logging**: Comprehensive training logs and model checkpoints

## Requirements

- Python 3.8+
- PyTorch 2.0+
- TorchVision
- Typer (for CLI)
- tqdm (for progress bars)

Install dependencies:

```bash
pip install torch torchvision typer tqdm
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/thealper2/mnist-multi-scale-cnn.git
cd mnist-multi-scale-cnn
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the training script with default parameters:

```bash
python main.py \
    --data_dir ./data \
    --batch_size 256 \
    --epochs 30 \
    --learning_rate 0.0005 \
    --dropout_rate 0.3 \
    --save_path best_model.pth
```

### Command Line Arguments

```shell
 Usage: main.py [OPTIONS]

 Train a Multi-scale CNN on the MNIST dataset.


╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --data-dir             TEXT     Directory to store MNIST data [default: ./data]                          │
│ --batch-size           INTEGER  Batch size for training [default: 128]                                   │
│ --epochs               INTEGER  Number of training epochs [default: 20]                                  │
│ --learning-rate        FLOAT    Learning rate [default: 0.001]                                           │
│ --weight-decay         FLOAT    Weight decay for regularization [default: 0.0001]                        │
│ --dropout-rate         FLOAT    Dropout rate [default: 0.5]                                              │
│ --val-split            FLOAT    Validation split ratio [default: 0.1]                                    │
│ --save-path            TEXT     Path to save the best model [default: ./best_model.pth]                  │
│ --seed                 INTEGER  Random seed for reproducibility [default: 42]                            │
│ --help                          Show this message and exit.                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (git checkout -b feature-branch)
3. Commit your changes (git commit -am 'Add new feature')
4. Push to the branch (git push origin feature-branch)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.