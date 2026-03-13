"""
Denoising Autoencoder (DAE) Module.
============================================================================
Paper Reference: Section 4.1, Page 5
============================================================================
"Each agent receives packets from the network traffic, and uses them to
 extract feature representations. These features representations will be
 passed through a denoising autoencoder (DAE) to protect the model from
 the adversarial attacks. The output of the DAE is then passed to the data
 preprocessor to normalize the learned state representation."

Architecture:
  Input (feature_dim) -> Encoder -> Bottleneck (hidden_dim) -> Decoder -> Output (feature_dim)

The DAE adds Gaussian noise to the input during training and learns to
reconstruct the clean input, making the model more robust to adversarial
perturbations.
============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class DenoisingAutoencoder(nn.Module):
    """
    Denoising Autoencoder for feature denoising.

    Paper Reference: Section 4.1, Page 5
    "features representations will be passed through a denoising autoencoder
     (DAE) to protect the model from the adversarial attacks"

    The DAE learns a robust feature representation by training to reconstruct
    clean inputs from noisy versions, providing resilience against adversarial
    perturbations in network traffic data.
    """

    def __init__(self, input_dim, hidden_dim=64, noise_factor=0.3):
        """
        Args:
            input_dim: Dimension of input features (after preprocessing)
            hidden_dim: Bottleneck dimension (compressed representation)
            noise_factor: Standard deviation of Gaussian noise added during training
        """
        super(DenoisingAutoencoder, self).__init__()

        self.noise_factor = noise_factor

        # Encoder: Maps input to compressed representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.ReLU()
        )

        # Decoder: Reconstructs input from compressed representation
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # Output in [0,1] since features are MinMax scaled
        )

    def add_noise(self, x):
        """
        Add Gaussian noise to input for denoising training.
        The noise makes the model robust to adversarial perturbations.
        """
        noise = torch.randn_like(x) * self.noise_factor
        noisy_x = x + noise
        # Clamp to valid range [0, 1] since features are normalized
        return torch.clamp(noisy_x, 0.0, 1.0)

    def forward(self, x, add_noise=False):
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, input_dim]
            add_noise: Whether to add noise (True during training)
        Returns:
            reconstructed: Reconstructed input
            encoded: Bottleneck representation (used as denoised features)
        """
        if add_noise:
            x_input = self.add_noise(x)
        else:
            x_input = x

        encoded = self.encoder(x_input)
        reconstructed = self.decoder(encoded)
        return reconstructed, encoded

    def encode(self, x):
        """
        Get denoised feature representation (bottleneck output).
        Used during inference to get clean features for the DQN.

        Paper Reference: Section 4.1, Page 5
        "The output of the DAE is then passed to the data preprocessor
         to normalize the learned state representation."
        """
        return self.encoder(x)


def train_dae(dae, X_train, device, epochs=20, batch_size=256, lr=0.001):
    """
    Train the Denoising Autoencoder.

    Paper Reference: Section 4.1, Page 5
    The DAE is pre-trained on the available data to learn robust feature
    representations before being used in the DQN pipeline.

    Args:
        dae: DenoisingAutoencoder model
        X_train: Training data numpy array
        device: torch device
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
    Returns:
        dae: Trained DAE model
        losses: List of training losses per epoch
    """
    dae = dae.to(device)
    optimizer = optim.Adam(dae.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.FloatTensor(X_train))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    dae.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch_x,) in dataloader:
            batch_x = batch_x.to(device)

            # Forward: add noise during training
            reconstructed, _ = dae(batch_x, add_noise=True)

            # Reconstruction loss against clean input
            loss = criterion(reconstructed, batch_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_x)

        avg_loss = epoch_loss / len(X_train)
        losses.append(avg_loss)

        if (epoch + 1) % 5 == 0:
            print(f"  [DAE] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    return dae, losses


def transform_with_dae(dae, X, device, batch_size=1024):
    """
    Transform features through the trained DAE to get denoised representations.

    Paper Reference: Section 4.1, Page 5
    "The output of the DAE is then passed to the data preprocessor"

    Args:
        dae: Trained DAE model
        X: Input features numpy array
        device: torch device
        batch_size: Batch size for transformation
    Returns:
        X_denoised: Denoised features as numpy array
    """
    dae.eval()
    dataset = TensorDataset(torch.FloatTensor(X))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    encoded_features = []
    with torch.no_grad():
        for (batch_x,) in dataloader:
            batch_x = batch_x.to(device)
            encoded = dae.encode(batch_x)
            encoded_features.append(encoded.cpu().numpy())

    return np.concatenate(encoded_features, axis=0)
