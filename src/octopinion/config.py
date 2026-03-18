"""Octopinion Lexical System - Configuration"""

from dataclasses import dataclass
from enum import Enum


class InitMethod(Enum):
    """Method for initializing codebook vectors"""

    RANDOM = "random"
    PRIMITIVES = "primitives"
    PCA = "pca"
    BALANCED_PCA = "balanced_pca"


@dataclass
class LexicalConfig:
    """Configuration for the Lexical System"""

    codebook_size: int = 64  # Number of syllables (should be power of 2 or octal-friendly)
    embedding_dim: int = 1024  # Dimension of semantic vectors (matches BGE-large)
    decay_factor: float = 0.7  # Lambda - positional weight decay
    max_word_length: int = 6  # Maximum syllables per word
    residual_threshold: float = 0.05  # Threshold for encoding termination
    temperature_start: float = 10.0  # Initial temperature for Gumbel-Softmax
    temperature_end: float = 0.001  # Final temperature for Gumbel-Softmax
    temperature_decay: float = 0.998  # Temperature decay per epoch
    learning_rate: float = 1e-6  # Learning rate for SGD
    momentum: float = 0.9  # Momentum for SGD
    num_training_steps: int = 4  # Fixed steps for training (K in spec)
    api_model: str = "BAAI/bge-large-en-v1.5"
    api_url: str = "https://api.siliconflow.cn/v1/embeddings"
    api_batch_size: int = 30  # Batch size for API calls (SiliconFlow supports batch input)
    init_method: InitMethod = InitMethod.PRIMITIVES  # Method for initializing codebook vectors
    allow_negative_signs: bool = True  # Allow negative decay factors (signed syllables)


DEFAULT_PRIMITIVES = [
    "opaque",
    "hard",
    "rough",
    "big",
    "near",
    "deep",
    "bright",
    "warm",
    "fast",
    "enclosed",
    "self",
    "alive",
    "dangerous",
    "hidden",
    "moving",
    "grip",
    "approach",
    "active",
    "edible",
    "familiar",
    "many",
    "now",
    "not",
    "I",
    "animal",
    "interior",
    "kin",
    "abstract",
    "confirmed",
    "whole",
    "vertical",
    "dense",
]
