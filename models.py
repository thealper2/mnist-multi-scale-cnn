import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleBlock(nn.Module):
    """
    Multi-scale convolutional block that processes input at different scales.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialize the multi-scale block.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels for each scale
        """
        super(MultiScaleBlock, self).__init__()

        # Different kernel sizes for multi-scale feature extraction
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)

        # Max pooling branch for capturing larger context
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv_pool = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

        # Batch normalization for each branch
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.bn4 = nn.BatchNorm2d(out_channels)

        # Dropout for regularization
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the multi-scale block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)

        Returns:
            Concatenated multi-scale features
        """
        # Apply different convolution scales
        branch1 = F.relu(self.bn1(self.conv1x1(x)))
        branch2 = F.relu(self.bn2(self.conv3x3(x)))
        branch3 = F.relu(self.bn3(self.conv5x5(x)))

        # Max pooling branch
        branch4 = self.maxpool(x)
        branch4 = F.relu(self.bn4(self.conv_pool(branch4)))

        # Concatenate all branches
        output = torch.cat([branch1, branch2, branch3, branch4], dim=1)

        # Apply dropout for regularization
        output = self.dropout(output)

        return output


class MultiScaleCNN(nn.Module):
    """
    Multi-scale Convolutional Neural Network for MNIST classification.
    """

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5) -> None:
        """
        Initialize the Multi-scale CNN.

        Args:
            num_classes (int): Number of output classes (10 for MNIST)
            dropout_rate (float): Dropout rate for regularization
        """
        super(MultiScaleCNN, self).__init__()

        # First multi-scale block
        self.multiscale1 = MultiScaleBlock(in_channels=1, out_channels=16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second multi-scale block
        self.multiscale2 = MultiScaleBlock(in_channels=64, out_channels=32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third multi-scale block
        self.multiscale3 = MultiScaleBlock(in_channels=128, out_channels=64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # First multi-scale block
        x = self.multiscale1(x)  # Output: (batch_size, 64, 28, 28)
        x = self.pool1(x)  # Output: (batch_size, 64, 14, 14)

        # Second multi-scale block
        x = self.multiscale2(x)  # Output: (batch_size, 128, 14, 14)
        x = self.pool2(x)  # Output: (batch_size, 128, 7, 7)

        # Third multi-scale block
        x = self.multiscale3(x)  # Output: (batch_size, 256, 7, 7)
        x = self.pool3(x)  # Output: (batch_size, 256, 3, 3)

        # Global average pooling
        x = self.global_avg_pool(x)  # Output: (batch_size, 256, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten: (batch_size, 256)

        # Classification
        x = self.classifier(x)

        return x
