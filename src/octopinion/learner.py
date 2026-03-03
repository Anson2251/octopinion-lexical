"""Codebook learning with Gumbel-Softmax"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from .config import LexicalConfig, InitMethod
from .codebook import Codebook
from .decoder import LexicalDecoder
from .encoder import LexicalEncoder


def compute_pca_init(embeddings: torch.Tensor, codebook_size: int) -> torch.Tensor:
    """
    Initialize codebook vectors using top-k principal components (PCA).

    Args:
        embeddings: Tensor of shape [n_samples, embedding_dim]
        codebook_size: Number of codebook vectors to generate

    Returns:
        Tensor of shape [codebook_size, embedding_dim]
    """
    embeddings_np = embeddings.cpu().numpy()

    mean = np.mean(embeddings_np, axis=0)
    centered = embeddings_np - mean

    cov = np.cov(centered, rowvar=False)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    top_k = eigenvectors[:, :codebook_size]

    vectors = torch.from_numpy(top_k.T).float()

    vectors = F.normalize(vectors, p=2, dim=1)

    return vectors


def _kmeans_simple(data: np.ndarray, k: int, max_iter: int = 50, seed: int = 42) -> np.ndarray:
    """
    Simple k-means implementation using numpy.

    Args:
        data: Data to cluster [n_samples, dim]
        k: Number of clusters
        max_iter: Maximum iterations
        seed: Random seed

    Returns:
        Cluster centers [k, dim]
    """
    np.random.seed(seed)
    n_samples = data.shape[0]

    indices = np.random.choice(n_samples, k, replace=False)
    centers = data[indices].copy()

    for _ in range(max_iter):
        distances = np.zeros((n_samples, k))
        for j in range(k):
            distances[:, j] = np.linalg.norm(data - centers[j], axis=1)

        labels = np.argmin(distances, axis=1)

        new_centers = np.zeros_like(centers)
        for j in range(k):
            cluster_points = data[labels == j]
            if len(cluster_points) > 0:
                new_centers[j] = cluster_points.mean(axis=0)
            else:
                new_centers[j] = centers[j]

        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    return centers


def compute_balanced_pca_init(embeddings: torch.Tensor, codebook_size: int) -> torch.Tensor:
    """
    Initialize codebook vectors using Balanced PCA.

    This method:
    1. Clusters the corpus embeddings into k balanced groups using k-means
    2. Computes the first principal component within each cluster
    3. Uses these as initial codebook vectors

    This ensures the initial codebook vectors are representative of different
    regions of the embedding space, providing better coverage than standard PCA.

    Args:
        embeddings: Tensor of shape [n_samples, embedding_dim]
        codebook_size: Number of codebook vectors to generate

    Returns:
        Tensor of shape [codebook_size, embedding_dim]
    """
    embeddings_np = embeddings.cpu().numpy()

    embeddings_np = embeddings_np / np.linalg.norm(embeddings_np, axis=1, keepdims=True)

    cluster_centers = _kmeans_simple(embeddings_np, codebook_size, seed=42)

    vectors_list = []
    for i in range(codebook_size):
        cluster_points = embeddings_np

        distances = np.linalg.norm(cluster_points - cluster_centers[i], axis=1)
        threshold = np.percentile(distances, 50)
        mask = distances <= threshold

        if mask.sum() > 1:
            cluster_subset = cluster_points[mask]
            mean = np.mean(cluster_subset, axis=0)
            centered = cluster_subset - mean

            cov = np.cov(centered, rowvar=False)

            try:
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                idx = np.argsort(eigenvalues)[::-1]
                first_pc = eigenvectors[:, idx[0]]
            except np.linalg.LinAlgError:
                first_pc = cluster_centers[i]

            first_pc = first_pc / np.linalg.norm(first_pc)
        else:
            first_pc = cluster_centers[i] / np.linalg.norm(cluster_centers[i])

        vectors_list.append(first_pc)

    vectors = np.stack(vectors_list)
    vectors = torch.from_numpy(vectors).float()

    vectors = F.normalize(vectors, p=2, dim=1)

    return vectors


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

    def __init__(self, config: LexicalConfig, initial_vectors: torch.Tensor = None):
        super().__init__()
        self.config = config
        self.codebook = Codebook(config)
        self.gumbel_softmax = GumbelSoftmax(temperature=config.temperature_start)
        self.current_temperature = config.temperature_start

        # Initialize codebook vectors if provided
        if initial_vectors is not None:
            self.initialize_codebook_vectors(initial_vectors)

        # Decoder for training
        self.decoder = LexicalDecoder(config, self.codebook)

    def initialize_codebook_vectors(self, vectors: torch.Tensor):
        """
        Initialize codebook vectors from a tensor of embeddings.

        Args:
            vectors: Tensor of shape [codebook_size, embedding_dim] containing
                    initial codebook vectors (will be normalized)
        """
        if vectors.shape != (self.config.codebook_size, self.config.embedding_dim):
            raise ValueError(
                f"Expected vectors shape {(self.config.codebook_size, self.config.embedding_dim)}, got {vectors.shape}"
            )

        with torch.no_grad():
            self.codebook.vectors.data = vectors.clone()
            self.codebook.normalize_codebook()

        print(f"Codebook initialized with {self.config.codebook_size} vectors")

    def forward(self, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
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
            gumbel_dist = self.gumbel_softmax(logits, hard=False)  # [batch_size, codebook_size]
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

    def _compute_entropy_penalty(self, distributions: List[torch.Tensor]) -> torch.Tensor:
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

    def train_step(self, targets: torch.Tensor, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
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
