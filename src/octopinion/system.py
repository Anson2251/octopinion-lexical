"""
Main Lexical System - integrates all components
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from .config import LexicalConfig
from .embedder import SiliconFlowEmbedding
from .codebook import Codebook
from .encoder import LexicalEncoder
from .decoder import LexicalDecoder
from .learner import CodebookLearner


class LexicalSystem:
    """
    Main interface for the Octopinion Lexical System.
    Integrates embedding API, encoding, decoding, and codebook learning.
    """

    def __init__(self, config: Optional[LexicalConfig] = None, api_token: Optional[str] = None):
        self.config = config or LexicalConfig()
        self.embedder = SiliconFlowEmbedding(api_token, model=self.config.api_model)

        # Initialize codebook learner (which includes codebook)
        self.learner = CodebookLearner(self.config)
        self.encoder = None  # Will be created after training
        self.decoder = None  # Will be created after training

    def encode_text(self, text: str) -> List[int]:
        """
        Encode text directly to syllable sequence.

        Args:
            text: Input text to encode
        Returns:
            Syllable sequence (list of indices)
        """
        # Get embedding
        embedding = self.embedder.get_embedding(text)

        # Ensure correct dimension
        if embedding.size(0) != self.config.embedding_dim:
            # Pad or truncate
            if embedding.size(0) < self.config.embedding_dim:
                padding = torch.zeros(self.config.embedding_dim - embedding.size(0))
                embedding = torch.cat([embedding, padding])
            else:
                embedding = embedding[: self.config.embedding_dim]

        # Encode
        if self.encoder is None:
            self.encoder = LexicalEncoder(self.config, self.learner.codebook)

        return self.encoder.encode(embedding)

    def decode_sequence(self, sequence: List[int]) -> torch.Tensor:
        """
        Decode syllable sequence to semantic vector.

        Args:
            sequence: List of syllable indices
        Returns:
            Semantic vector
        """
        if self.decoder is None:
            self.decoder = LexicalDecoder(self.config, self.learner.codebook)

        return self.decoder.decode(sequence)

    def decode_to_text(self, sequence: List[int], vocabulary: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Decode syllable sequence to closest text word.

        Args:
            sequence: List of syllable indices
            vocabulary: Optional list of words to search from. If None, uses cached embeddings.

        Returns:
            Dictionary with decoded vector and closest word info
        """
        vector = self.decode_sequence(sequence)

        if vocabulary:
            embeddings = self.embedder.get_embeddings_batch(vocabulary, show_progress=False)
            vocab_matrix = torch.stack(embeddings)
            vocab_matrix = vocab_matrix / torch.norm(vocab_matrix, dim=1, keepdim=True)
        else:
            cached = self.embedder.cache.get_all()
            if not cached:
                raise ValueError("No cached embeddings found. Provide a vocabulary or train with a corpus first.")
            vocabulary = [text for text, _ in cached]
            vocab_matrix = torch.stack([emb for _, emb in cached])

        vector_normalized = vector / torch.norm(vector)
        similarities = torch.mm(vector_normalized.view(1, -1), vocab_matrix.t()).squeeze()
        best_idx = similarities.argmax().item()
        best_word = vocabulary[best_idx]
        best_similarity = similarities[best_idx].item()

        return {
            "vector": vector,
            "word": best_word,
            "similarity": best_similarity,
            "all_words": sorted(
                [(vocabulary[i], similarities[i].item()) for i in range(len(vocabulary))],
                key=lambda x: x[1],
                reverse=True,
            )[:10],
        }

    def sequence_to_string(self, sequence: List[int]) -> str:
        """Convert syllable sequence to human-readable string"""
        return "-".join(f"S{i}" for i in sequence)

    def train(
        self,
        corpus: List[str],
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True,
    ):
        """
        Train the codebook on a corpus of text.

        Args:
            corpus: List of text strings
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        if verbose:
            print(f"Training codebook on {len(corpus)} concepts...")

        # Get embeddings for corpus using batch API
        if verbose:
            print(f"Fetching embeddings (API batch size: {self.config.api_batch_size})...")
        embeddings = self.embedder.get_embeddings_batch(
            corpus, batch_size=self.config.api_batch_size, show_progress=verbose
        )

        if not embeddings:
            raise RuntimeError("No valid embeddings retrieved. Cannot train.")

        # Convert to tensor
        embeddings_tensor = torch.stack(embeddings)

        # Setup optimizer (SGD with momentum)
        optimizer = torch.optim.SGD(
            self.learner.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum
        )

        # Training loop
        if verbose:
            print(f"\nTraining for {epochs} epochs...")
        dataset_size = len(embeddings_tensor)

        epoch_range = tqdm(range(epochs), desc="Training") if verbose else range(epochs)

        for epoch in epoch_range:
            # Shuffle data
            indices = torch.randperm(dataset_size)
            total_loss = 0
            num_batches = 0
            last_metrics = None

            for i in range(0, dataset_size, batch_size):
                batch_indices = indices[i : i + batch_size]
                batch = embeddings_tensor[batch_indices]

                metrics = self.learner.train_step(batch, optimizer)
                total_loss += metrics["loss"]
                num_batches += 1
                last_metrics = metrics

            avg_loss = total_loss / num_batches

            if verbose:
                temp = last_metrics["temperature"] if last_metrics else 0.0
                epoch_range.set_postfix(loss=f"{avg_loss:.4f}", temp=f"{temp:.2f}")

        # Create encoder/decoder after training
        self.encoder = LexicalEncoder(self.config, self.learner.codebook)
        self.decoder = LexicalDecoder(self.config, self.learner.codebook)

        if verbose:
            print("\nTraining complete!")

    def encode_corpus(self, corpus: List[str]) -> Dict[str, List[int]]:
        """
        Encode entire corpus and return mapping.

        Args:
            corpus: List of text strings
        Returns:
            Dictionary mapping text to syllable sequence
        """
        result = {}
        for text in corpus:
            try:
                sequence = self.encode_text(text)
                result[text] = sequence
            except Exception as e:
                print(f"Warning: Failed to encode '{text}': {e}")
        return result

    def analyze_codebook(self) -> Dict[str, Any]:
        """Analyze the learned codebook"""
        with torch.no_grad():
            codebook = self.learner.codebook().numpy()

            # Compute pairwise similarities
            similarities = np.dot(codebook, codebook.T)

            # Find average similarity (excluding diagonal)
            n = codebook.shape[0]
            mask = ~np.eye(n, dtype=bool)
            avg_similarity = similarities[mask].mean()

            return {
                "codebook_size": n,
                "embedding_dim": codebook.shape[1],
                "avg_pairwise_similarity": float(avg_similarity),
                "min_similarity": float(similarities[mask].min()),
                "max_similarity": float(similarities[mask].max()),
            }

    def export_codebook_words(
        self, vocabulary: List[str], top_k: int = 10, show_progress: bool = False
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Export words that correspond to each codebook item.

        Uses cosine similarity to find the closest words in the vocabulary
        to each codebook vector.

        Args:
            vocabulary: List of words to search from
            top_k: Number of closest words to return per codebook item
            show_progress: Whether to show progress bar

        Returns:
            Dict mapping codebook index to list of {word, similarity} dicts
        """
        if not vocabulary:
            raise ValueError("Vocabulary cannot be empty")

        if show_progress:
            print(f"Getting embeddings for {len(vocabulary)} vocabulary words...")

        vocab_embeddings = self.embedder.get_embeddings_batch(
            vocabulary, batch_size=self.config.api_batch_size, show_progress=show_progress
        )

        vocab_matrix = torch.stack(vocab_embeddings).numpy()
        vocab_matrix = vocab_matrix / np.linalg.norm(vocab_matrix, axis=1, keepdims=True)

        codebook_vectors = self.learner.codebook().detach().numpy()
        codebook_vectors = codebook_vectors / np.linalg.norm(codebook_vectors, axis=1, keepdims=True)

        similarities = np.dot(codebook_vectors, vocab_matrix.T)

        results = {}
        for i in range(len(codebook_vectors)):
            top_indices = np.argsort(similarities[i])[::-1][:top_k]
            results[i] = [{"word": vocabulary[idx], "similarity": float(similarities[i][idx])} for idx in top_indices]

        return results

    def save(self, path: str):
        """Save the trained system"""
        torch.save(
            {
                "config": self.config,
                "codebook_state": self.learner.codebook.state_dict(),
                "encoder": self.encoder is not None,
                "decoder": self.decoder is not None,
            },
            path,
        )
        print(f"System saved to {path}")

    @classmethod
    def load(cls, path: str, api_token: Optional[str] = None):
        """Load a trained system"""
        # Register safe globals to allow loading LexicalConfig
        # Required for PyTorch 2.6+ weights_only=True default
        from torch.serialization import add_safe_globals
        from .config import LexicalConfig

        add_safe_globals([LexicalConfig])

        checkpoint = torch.load(path, weights_only=True)
        config = checkpoint["config"]

        system = cls(config, api_token)
        system.learner.codebook.load_state_dict(checkpoint["codebook_state"])

        if checkpoint["encoder"]:
            system.encoder = LexicalEncoder(config, system.learner.codebook)
        if checkpoint["decoder"]:
            system.decoder = LexicalDecoder(config, system.learner.codebook)

        print(f"System loaded from {path}")
        return system
