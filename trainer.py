from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """Trainer class for training and evaluating the Multi-scale CNN."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
    ) -> None:
        """
        Initialize the trainer.

        Args:
            model (nn.Module): The neural network model
            device (torch.device): Device to run computations on
            learning_rate (float): Learning rate for optimizer
            weight_decay (float): Weight decay for regularization
        """
        self.model = model.to(device)
        self.device = device

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.7, patience=5, verbose=True
        )

        # Training history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def train_epoch(
        self, train_loader: DataLoader, epoch: int, epochs: int
    ) -> Tuple[float, float]:
        """
        Train the model for one epoch.

        Args:
            train_loader (DataLoader): Training data loader
            epoch (int): Current epoch
            epochs (int): Total epoch

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            # Update progress bar
            if batch_idx % 100 == 0:
                pbar.set_postfix(
                    {"loss": loss.item(), "accuracy": 100.0 * correct / total}
                )

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model.

        Args:
            val_loader (DataLoader): Validation data loader

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        save_path: str,
    ) -> None:
        """
        Train the model for multiple epochs.

        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            epochs (int): Number of epochs to train
            save_path (str): Path to save the best model
        """
        best_val_acc = 0.0
        best_model_state = None

        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, epoch, epochs)

            # Validation
            val_loss, val_acc = self.validate(val_loader)

            # Update learning rate
            self.scheduler.step(val_loss)

            # Save history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()

        # Save best model
        if best_model_state is not None:
            torch.save(
                {
                    "model_state_dict": best_model_state,
                    "best_val_acc": best_val_acc,
                    "history": self.history,
                },
                save_path,
            )

    def test(self, test_loader: DataLoader) -> float:
        """
        Test the model on test dataset.

        Args:
            test_loader (DataLoader): Test data loader

        Returns:
            Test accuracy
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        accuracy = 100.0 * correct / total
        return accuracy
