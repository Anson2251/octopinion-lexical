"""
Octopinion Lexical System

A Vector Quantization-based encoding/decoding system for the Octopinion constructed language.

Usage:
    from octopinion import LexicalSystem, LexicalConfig

    config = LexicalConfig(codebook_size=26, decay_factor=0.5)
    system = LexicalSystem(config, api_token="your_token")
    system.train(corpus, epochs=100)
    sequence = system.encode_text("fish")
"""

__version__ = "0.1.0"

from .config import LexicalConfig
from .system import LexicalSystem
from .embedder import SiliconFlowEmbedding
from .codebook import Codebook
from .encoder import LexicalEncoder
from .decoder import LexicalDecoder
from .learner import CodebookLearner
from .cache import EmbeddingCache

__all__ = [
    "LexicalConfig",
    "LexicalSystem",
    "SiliconFlowEmbedding",
    "Codebook",
    "LexicalEncoder",
    "LexicalDecoder",
    "CodebookLearner",
    "EmbeddingCache",
]
