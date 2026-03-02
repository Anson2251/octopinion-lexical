"""Learnable codebook of semantic vectors"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import LexicalConfig


class Codebook(nn.Module):
    """Learnable codebook of semantic vectors (syllables)"""

    def __init__(self, config: LexicalConfig):
        super().__init__()
        self.config = config

        # Initialize codebook vectors randomly, then normalize
        self.vectors = nn.Parameter(
            torch.randn(config.codebook_size, config.embedding_dim)
        )
        self.normalize_codebook()

    def normalize_codebook(self):
        """Normalize all codebook vectors to unit length"""
        with torch.no_grad():
            self.vectors.data = F.normalize(self.vectors.data, p=2, dim=1)

    def forward(self) -> torch.Tensor:
        """Return normalized codebook vectors"""
        return F.normalize(self.vectors, p=2, dim=1)

    def get_vector(self, index: int) -> torch.Tensor:
        """Get a specific codebook vector by index"""
        return F.normalize(self.vectors[index], p=2, dim=0)

    def get_vector_batch(self, indices: torch.Tensor) -> torch.Tensor:
        """Get multiple codebook vectors by indices"""
        vecs = self.vectors[indices]
        return F.normalize(vecs, p=2, dim=1)
