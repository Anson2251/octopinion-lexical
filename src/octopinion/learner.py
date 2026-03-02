"""Codebook learning with Gumbel-Softmax"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
from .config import LexicalConfig
from .codebook import Codebook
from .decoder import LexicalDecoder
from .encoder import LexicalEncoder


class GumbelSoftmax(nn.Module):
    """Gumbel-Softmax for differentiable discrete sampling"""

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, logits: torch.Tensor, hard: bool = False) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_classes]
            hard: if True, returns one-hot (for inference), else soft
        Returns:
            Sampled distribution [batch_size, num_classes]
        """
        # Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)

        # Add noise and apply temperature
        y = logits + gumbel_noise
        y = F.softmax(y / self.temperature, dim=-1)

        if hard:
            # Straight-through estimator
            y_hard = torch.zeros_like(y)
            y_hard.scatter_(1, y.argmax(dim=1, keepdim=True), 1.0)
            y = (y_hard - y).detach() + y

        return y


class CodebookLearner(nn.Module):
    """
    Learning system for the codebook using Gumbel-Softmax and residual pursuit.
    """

    def __init__(self, config: LexicalConfig):
        super().__init__()
        self.config = config
        self.codebook = Codebook(config)
        self.gumbel_softmax = GumbelSoftmax(temperature=config.temperature_start)
        self.current_temperature = config.temperature_start

        # Decoder for training
        self.decoder = LexicalDecoder(config, self.codebook)

    def forward(
        self, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with differentiable Gumbel-Softmax.

        Args:
            targets: Batch of target vectors [batch_size, embedding_dim]
        Returns:
            - reconstructed: [batch_size, embedding_dim]
            - loss: scalar reconstruction loss
            - selected_indices: List of Gumbel-softmax distributions per step
        """
        batch_size = targets.size(0)
        device = targets.device

        # Initialize
        residual = targets.clone()
        reconstructed = torch.zeros_like(targets)
        selected_distributions = []

        codebook_vecs = self.codebook().to(device)  # [codebook_size, embedding_dim]

        for step in range(self.config.num_training_steps):
            # Compute logits (dot products)
            # [batch_size, codebook_size]
            logits = torch.matmul(residual, codebook_vecs.t())

            # Sample using Gumbel-Softmax
            gumbel_dist = self.gumbel_softmax(
                logits, hard=False
            )  # [batch_size, codebook_size]
            selected_distributions.append(gumbel_dist)

            # Compute weighted contribution
            # [batch_size, codebook_size] @ [codebook_size, embedding_dim] = [batch_size, embedding_dim]
            weighted_vectors = torch.matmul(gumbel_dist, codebook_vecs)

            decay = self.config.decay_factor**step
            contribution = decay * weighted_vectors

            # Update
            reconstructed += contribution
            residual = residual - contribution

        # Compute loss
        reconstruction_loss = F.mse_loss(reconstructed, targets)

        # Optional: entropy penalty to encourage peaky distributions
        entropy_penalty = self._compute_entropy_penalty(selected_distributions)

        total_loss = reconstruction_loss + 0.01 * entropy_penalty

        return reconstructed, total_loss, selected_distributions

    def _compute_entropy_penalty(
        self, distributions: List[torch.Tensor]
    ) -> torch.Tensor:
        """Compute entropy penalty to encourage peaky distributions"""
        total_entropy = torch.tensor(0.0)
        for dist in distributions:
            # Entropy: -sum(p * log(p))
            entropy = -torch.sum(dist * torch.log(dist + 1e-10), dim=-1)
            total_entropy = total_entropy + entropy.mean()
        return total_entropy / float(len(distributions))

    def update_temperature(self):
        """Decay temperature for annealing"""
        self.current_temperature = max(
            self.config.temperature_end,
            self.current_temperature * self.config.temperature_decay,
        )
        self.gumbel_softmax.temperature = self.current_temperature

    def train_step(
        self, targets: torch.Tensor, optimizer: torch.optim.Optimizer
    ) -> Dict[str, float]:
        """Single training step"""
        self.train()
        optimizer.zero_grad()

        reconstructed, loss, _ = self.forward(targets)
        loss.backward()

        # Normalize codebook after gradient update
        optimizer.step()
        self.codebook.normalize_codebook()

        # Update temperature
        self.update_temperature()

        return {"loss": loss.item(), "temperature": self.current_temperature}

    def get_discrete_sequence(self, target: torch.Tensor) -> List[int]:
        """Get discrete sequence using trained codebook (inference)"""
        encoder = LexicalEncoder(self.config, self.codebook)
        return encoder.encode(target)
