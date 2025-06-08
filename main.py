from typing import Optional

import numpy as np
import torch
import typer

from config import setup_logging
from data import MNISTDataModule
from models import MultiScaleCNN
from trainer import Trainer


def main(
    data_dir: str = typer.Option("./data", help="Directory to store MNIST data"),
    batch_size: int = typer.Option(128, help="Batch size for training"),
    epochs: int = typer.Option(20, help="Number of training epochs"),
    learning_rate: float = typer.Option(0.001, help="Learning rate"),
    weight_decay: float = typer.Option(1e-4, help="Weight decay for regularization"),
    dropout_rate: float = typer.Option(0.5, help="Dropout rate"),
    val_split: float = typer.Option(0.1, help="Validation split ratio"),
    save_path: str = typer.Option(
        "./best_model.pth", help="Path to save the best model"
    ),
    seed: Optional[int] = typer.Option(42, help="Random seed for reproducibility"),
) -> None:
    """
    Train a Multi-scale CNN on the MNIST dataset.
    """
    logger = setup_logging()

    try:
        # Set random seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Device configuration
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Initialize data module
        data_module = MNISTDataModule(data_dir, batch_size, val_split)
        data_module.prepare_data()
        train_loader, val_loader, test_loader = data_module.get_dataloaders()

        # Initialize model
        model = MultiScaleCNN(num_classes=10, dropout_rate=dropout_rate)

        # Display model information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info("Model Information:")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        # Initialize trainer
        trainer = Trainer(model, device, learning_rate, weight_decay)

        # Train the model
        trainer.fit(train_loader, val_loader, epochs, save_path)

        # Test the model
        test_accuracy = trainer.test(test_loader)
        logger.info(f"Final Test Accuracy: {test_accuracy:.2f}%")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
