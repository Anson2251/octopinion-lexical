"""Decoding system - linear composition"""

import torch
from typing import List
from .config import LexicalConfig
from .codebook import Codebook


class LexicalDecoder:
    """
    Decoding system: Converts syllable sequences to semantic vectors.
    Supports signed sequences where negative indices indicate negative contribution.
    """

    def __init__(self, config: LexicalConfig, codebook: Codebook):
        self.config = config
        self.codebook = codebook

    def decode(self, sequence: List[int]) -> torch.Tensor:
        """
        Decode signed syllable sequence to semantic vector.

        Args:
            sequence: List of signed syllable indices (negative = subtract, positive = add)
        Returns:
            Decoded semantic vector [embedding_dim]
        """
        if not sequence:
            return torch.zeros(self.config.embedding_dim)

        meaning = torch.zeros(self.config.embedding_dim)

        for j, signed_idx in enumerate(sequence):
            # Extract actual index and sign
            if signed_idx < 0:
                actual_idx = abs(signed_idx) - 1  # Convert -1, -2, -3... to 0, 1, 2...
                sign = -1
            else:
                actual_idx = signed_idx
                sign = 1

            syllable_vec = self.codebook.get_vector(actual_idx)
            weight = self.config.decay_factor ** (j + 1)
            meaning += sign * weight * syllable_vec

        return meaning

    def decode_batch(self, sequences: List[List[int]]) -> torch.Tensor:
        """Decode multiple sequences"""
        decoded = [self.decode(seq) for seq in sequences]
        return torch.stack(decoded)
