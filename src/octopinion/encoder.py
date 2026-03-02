"""Encoding system - greedy residual pursuit"""

import torch
import torch.nn as nn
from typing import List
from .config import LexicalConfig
from .codebook import Codebook


class LexicalEncoder(nn.Module):
    """
    Encoding system: Converts semantic vectors to syllable sequences.
    Uses greedy residual pursuit for discrete encoding.
    """

    def __init__(self, config: LexicalConfig, codebook: Codebook):
        super().__init__()
        self.config = config
        self.codebook = codebook

    @torch.no_grad()
    def encode(self, target: torch.Tensor) -> List[int]:
        """
        Greedy residual pursuit encoding.

        Args:
            target: Target semantic vector [embedding_dim]
        Returns:
            List of syllable indices
        """
        device = target.device
        target = target.to(device)

        residual = target.clone()
        sequence = []

        for step in range(self.config.max_word_length):
            # Check termination condition
            residual_norm = torch.norm(residual)
            if residual_norm < self.config.residual_threshold:
                break

            # Compute scores: dot product with all codebook vectors
            codebook_vecs = self.codebook().to(device)  # [codebook_size, embedding_dim]
            scores = torch.matmul(codebook_vecs, residual)  # [codebook_size]

            # Select best syllable (argmax)
            best_idx = torch.argmax(scores).item()
            sequence.append(best_idx)

            # Update residual
            decay = self.config.decay_factor**step
            contribution = decay * codebook_vecs[best_idx]
            residual = residual - contribution

        return sequence

    def encode_batch(self, targets: torch.Tensor) -> List[List[int]]:
        """Encode multiple target vectors"""
        return [self.encode(t) for t in targets]
