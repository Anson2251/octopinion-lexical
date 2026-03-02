"""Decoding system - linear composition"""

import torch
from typing import List
from .config import LexicalConfig
from .codebook import Codebook


class LexicalDecoder:
    """
    Decoding system: Converts syllable sequences to semantic vectors.
    """

    def __init__(self, config: LexicalConfig, codebook: Codebook):
        self.config = config
        self.codebook = codebook

    def decode(self, sequence: List[int]) -> torch.Tensor:
        """
        Decode syllable sequence to semantic vector.

        Args:
            sequence: List of syllable indices
        Returns:
            Decoded semantic vector [embedding_dim]
        """
        if not sequence:
            return torch.zeros(self.config.embedding_dim)

        meaning = torch.zeros(self.config.embedding_dim)

        for j, idx in enumerate(sequence):
            syllable_vec = self.codebook.get_vector(idx)
            weight = self.config.decay_factor**j
            meaning += weight * syllable_vec

        return meaning

    def decode_batch(self, sequences: List[List[int]]) -> torch.Tensor:
        """Decode multiple sequences"""
        decoded = [self.decode(seq) for seq in sequences]
        return torch.stack(decoded)
