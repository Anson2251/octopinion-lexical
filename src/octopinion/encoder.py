"""Encoding system - greedy residual pursuit with cosine similarity"""

import torch
import torch.nn as nn
from typing import List
from .config import LexicalConfig
from .codebook import Codebook


class LexicalEncoder(nn.Module):
    """
    Encoding system: Converts semantic vectors to syllable sequences.
    Uses greedy residual pursuit with cosine similarity for direction-based selection.
    Supports signed sequences where negative indices indicate negative contribution.
    """

    def __init__(self, config: LexicalConfig, codebook: Codebook):
        super().__init__()
        self.config = config
        self.codebook = codebook

    @torch.no_grad()
    def encode(self, target: torch.Tensor) -> List[int]:
        """
        Greedy residual pursuit encoding with direction-based selection.
        Uses cosine similarity to find codebook vectors that best align with residual direction.

        Args:
            target: Target semantic vector [embedding_dim]
        Returns:
            List of signed syllable indices (negative = subtract, positive = add)
        """
        device = target.device
        target = target.to(device)
        target_norm = torch.norm(target)

        # Early exit for zero vector
        if target_norm < 1e-8:
            return []

        residual = target.clone()
        sequence = []

        for step in range(self.config.max_word_length):
            residual_norm = torch.norm(residual)
            if residual_norm < self.config.residual_threshold:
                break

            codebook_vecs = self.codebook().to(device)  # [codebook_size, embedding_dim]

            if self.config.allow_negative_signs:
                # Normalize residual for cosine similarity
                residual_normalized = residual / residual_norm

                # Compute cosine similarities with all codebook vectors
                # cos_sim = dot(residual_normalized, v_i) since ||v_i|| = 1
                cos_sims = torch.matmul(codebook_vecs, residual_normalized)  # [codebook_size]

                # Find syllable with maximum absolute cosine similarity
                abs_cos_sims = torch.abs(cos_sims)
                best_idx = int(torch.argmax(abs_cos_sims).item())
                best_cos_sim = float(cos_sims[best_idx].item())

                # Determine sign: positive if cos_sim > 0, negative if cos_sim < 0
                # This makes the selected vector point in the same direction as residual
                if best_cos_sim >= 0:
                    signed_idx = best_idx
                else:
                    signed_idx = -(best_idx + 1)
            else:
                # Original: use dot product (equivalent to cosine since ||v|| = 1)
                scores = torch.matmul(codebook_vecs, residual)
                best_idx = int(torch.argmax(scores).item())
                signed_idx = best_idx

            sequence.append(signed_idx)

            # Update residual: subtract the actual contribution from reconstruction
            decay = self.config.decay_factor ** (step + 1)
            actual_idx = int(abs(signed_idx) - 1 if signed_idx < 0 else signed_idx)
            sign = -1 if signed_idx < 0 else 1
            contribution = sign * decay * codebook_vecs[actual_idx]
            residual = residual - contribution

        return sequence

    def encode_batch(self, targets: torch.Tensor) -> List[List[int]]:
        """Encode multiple target vectors"""
        return [self.encode(t) for t in targets]
