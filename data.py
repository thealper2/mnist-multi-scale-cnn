from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class MNISTDataModule:
    """Data module for loading and preprocessing MNIST dataset."""

    def __init__(self, data_dir: str, batch_size: int, val_split: float = 0.1) -> None:
        """
        Initialize the data module.

        Args:
            data_dir: Directory to store/load MNIST data
            batch_size: Batch size for data loaders
            val_split: Fraction of training data to use for validation
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.val_split = val_split

        # Data transformations
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
            ]
        )

        # Data augmentation for training
        self.train_transform = transforms.Compose(
            [
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self) -> None:
        """Download MNIST dataset if not already present."""
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create and return train, validation, and test data loaders.

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Load training dataset
        train_dataset = datasets.MNIST(
            self.data_dir, train=True, transform=self.train_transform
        )

        # Split training data into train and validation
        train_size = int((1 - self.val_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size

        train_subset, val_subset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        # Apply different transforms to validation set
        val_dataset = datasets.MNIST(
            self.data_dir, train=True, transform=self.transform
        )
        val_subset.dataset = val_dataset

        # Load test dataset
        test_dataset = datasets.MNIST(
            self.data_dir, train=False, transform=self.transform
        )

        # Create data loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        return train_loader, val_loader, test_loader
