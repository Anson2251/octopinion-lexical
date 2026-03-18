"""
Main Lexical System - integrates all components
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from .config import LexicalConfig, InitMethod, DEFAULT_PRIMITIVES
from .embedder import SiliconFlowEmbedding
from .codebook import Codebook
from .encoder import LexicalEncoder
from .decoder import LexicalDecoder
from .learner import CodebookLearner, compute_pca_init, compute_balanced_pca_init


# Default primitive words for codebook initialization


class LexicalSystem:
    """
    Main interface for the Octopinion Lexical System.
    Integrates embedding API, encoding, decoding, and codebook learning.
    """

    def __init__(
        self,
        config: Optional[LexicalConfig] = None,
        api_token: Optional[str] = None,
        use_primitives: bool = True,
        primitives: Optional[List[str]] = None,
        auto_initialize: bool = True,
    ):
        self.config = config or LexicalConfig()
        self.embedder = SiliconFlowEmbedding(api_token, model=self.config.api_model)

        # Initialize codebook learner (which includes codebook)
        self.learner = CodebookLearner(self.config)
        self.encoder = None  # Will be created after training
        self.decoder = None  # Will be created after training
        self.corpus = None  # Will store training corpus vocabulary
        self._initialized = False  # Track if codebook has been initialized

        # Auto-initialize based on config.init_method if requested and API token available
        if auto_initialize and self.embedder.api_token:
            if self.config.init_method == InitMethod.PRIMITIVES and use_primitives:
                primitives_to_use = primitives if primitives is not None else DEFAULT_PRIMITIVES
                if len(primitives_to_use) == self.config.codebook_size:
                    try:
                        self.initialize_from_primitives(primitives_to_use, verbose=False)
                        self._initialized = True
                    except Exception as e:
                        pass
            elif self.config.init_method == InitMethod.PCA or self.config.init_method == InitMethod.BALANCED_PCA:
                pass  # Will be initialized after training corpus is provided

    def initialize_from_primitives(self, primitives: List[str], verbose: bool = True):
        """
        Initialize codebook vectors with semantic embeddings of primitive words.

        Args:
            primitives: List of primitive words to use for initialization.
                       Must have exactly codebook_size words.
            verbose: Whether to print progress

        Returns:
            List of primitive words used for initialization
        """
        if len(primitives) != self.config.codebook_size:
            raise ValueError(
                f"Number of primitives ({len(primitives)}) must match codebook_size ({self.config.codebook_size})"
            )

        if verbose:
            print(f"Initializing codebook with {len(primitives)} primitive words...")

        # Fetch embeddings for primitives
        if verbose:
            print(f"Fetching embeddings (API batch size: {self.config.api_batch_size})...")
        embeddings = self.embedder.get_embeddings_batch(
            primitives, batch_size=self.config.api_batch_size, show_progress=verbose
        )

        if not embeddings:
            raise RuntimeError("No valid embeddings retrieved. Cannot initialize codebook.")

        # Stack into tensor and initialize codebook
        embeddings_tensor = torch.stack(embeddings)
        self.learner.initialize_codebook_vectors(embeddings_tensor)

        if verbose:
            print(f"\nCodebook initialized successfully!")
            print("\nPrimitive-to-index mapping:")
            for i, word in enumerate(primitives):
                print(f"  [{i:2d}] {word}")

        return primitives

    def initialize_from_pca(self, corpus: List[str], balanced: bool = False, verbose: bool = True):
        """
        Initialize codebook vectors using PCA on corpus embeddings.

        Args:
            corpus: List of text strings to use for PCA
            balanced: If True, use Balanced PCA (cluster + PCA per cluster)
            verbose: Whether to print progress

        Returns:
            Method used for initialization
        """
        if verbose:
            method_name = "Balanced PCA" if balanced else "PCA"
            print(f"Initializing codebook with {method_name} on {len(corpus)} corpus items...")

        embeddings = self.embedder.get_embeddings_batch(
            corpus, batch_size=self.config.api_batch_size, show_progress=verbose
        )

        if not embeddings:
            raise RuntimeError("No valid embeddings retrieved. Cannot initialize codebook.")

        embeddings_tensor = torch.stack(embeddings)

        if balanced:
            init_vectors = compute_balanced_pca_init(embeddings_tensor, self.config.codebook_size)
            method = "balanced_pca"
            method_name = "Balanced PCA"
        else:
            init_vectors = compute_pca_init(embeddings_tensor, self.config.codebook_size)
            method = "pca"
            method_name = "PCA"

        self.learner.initialize_codebook_vectors(init_vectors)
        self._initialized = True

        if verbose:
            print(f"Codebook initialized successfully with {method_name}!")

        return method

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
            vocabulary: Optional list of words to search from.
                       If None, uses the training corpus vocabulary.

        Returns:
            Dictionary with decoded vector and closest word info
        """
        vector = self.decode_sequence(sequence)

        # Determine vocabulary to use
        if vocabulary is not None:
            # Use provided vocabulary
            search_vocab = vocabulary
        elif self.corpus is not None:
            # Use training corpus by default
            search_vocab = self.corpus
        else:
            # Fallback to cached embeddings (legacy behavior)
            cached = self.embedder.cache.get_all()
            if not cached:
                raise ValueError("No vocabulary available. Train with a corpus first or provide a vocabulary.")
            search_vocab = [text for text, _ in cached]
            embeddings = [emb for _, emb in cached]
            vocab_matrix = torch.stack(embeddings)
            vocab_matrix = vocab_matrix / torch.norm(vocab_matrix, dim=1, keepdim=True)

            # Compute similarities
            vector_normalized = vector / torch.norm(vector)
            similarities = torch.mm(vector_normalized.view(1, -1), vocab_matrix.t()).squeeze()
            best_idx = similarities.argmax().item()
            best_word = search_vocab[best_idx]
            best_similarity = similarities[best_idx].item()

            return {
                "vector": vector,
                "word": best_word,
                "similarity": best_similarity,
                "all_words": sorted(
                    [(search_vocab[i], similarities[i].item()) for i in range(len(search_vocab))],
                    key=lambda x: x[1],
                    reverse=True,
                )[:10],
            }

        # Fetch embeddings for the vocabulary
        embeddings = self.embedder.get_embeddings_batch(search_vocab, show_progress=False)
        vocab_matrix = torch.stack(embeddings)
        vocab_matrix = vocab_matrix / torch.norm(vocab_matrix, dim=1, keepdim=True)

        vector_normalized = vector / torch.norm(vector)
        similarities = torch.mm(vector_normalized.view(1, -1), vocab_matrix.t()).squeeze()
        best_idx = similarities.argmax().item()
        best_word = search_vocab[best_idx]
        best_similarity = similarities[best_idx].item()

        return {
            "vector": vector,
            "word": best_word,
            "similarity": best_similarity,
            "all_words": sorted(
                [(search_vocab[i], similarities[i].item()) for i in range(len(search_vocab))],
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

        # Initialize from PCA if configured and not already initialized
        if not self._initialized:
            if self.config.init_method == InitMethod.PCA:
                if verbose:
                    print(f"\nInitializing codebook with PCA...")
                init_vectors = compute_pca_init(embeddings_tensor, self.config.codebook_size)
                self.learner.initialize_codebook_vectors(init_vectors)
                self._initialized = True
            elif self.config.init_method == InitMethod.BALANCED_PCA:
                if verbose:
                    print(f"\nInitializing codebook with Balanced PCA...")
                init_vectors = compute_balanced_pca_init(embeddings_tensor, self.config.codebook_size)
                self.learner.initialize_codebook_vectors(init_vectors)
                self._initialized = True

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
            total_recon_loss = 0
            num_batches = 0
            last_metrics = None

            for i in range(0, dataset_size, batch_size):
                batch_indices = indices[i : i + batch_size]
                batch = embeddings_tensor[batch_indices]

                metrics = self.learner.train_step(batch, optimizer)
                total_loss += metrics["loss"]
                total_recon_loss += metrics["reconstruction_loss"]
                num_batches += 1
                last_metrics = metrics

            avg_loss = total_loss / num_batches
            avg_recon_loss = total_recon_loss / num_batches

            if verbose:
                temp = last_metrics["temperature"] if last_metrics else 0.0
                epoch_range.set_postfix(loss=f"{avg_loss:.4f}", recon_loss=f"{avg_recon_loss:.4f}", temp=f"{temp:.2f}")

        # Final reconstruction loss
        final_recon_loss = avg_recon_loss

        # Create encoder/decoder after training
        self.encoder = LexicalEncoder(self.config, self.learner.codebook)
        self.decoder = LexicalDecoder(self.config, self.learner.codebook)

        # Store the corpus vocabulary for decoding
        self.corpus = list(dict.fromkeys(corpus))  # Remove duplicates while preserving order

        if verbose:
            print(f"\nTraining complete! Final reconstruction loss: {final_recon_loss:.6f}")

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

        unique_vocab = list(dict.fromkeys(vocabulary))
        if len(unique_vocab) < len(vocabulary):
            print(f"Warning: Removed {len(vocabulary) - len(unique_vocab)} duplicate words from vocabulary")

        if show_progress:
            print(f"Getting embeddings for {len(unique_vocab)} vocabulary words...")

        vocab_embeddings = self.embedder.get_embeddings_batch(
            unique_vocab, batch_size=self.config.api_batch_size, show_progress=show_progress
        )

        vocab_matrix = torch.stack(vocab_embeddings).numpy()
        vocab_matrix = vocab_matrix / np.linalg.norm(vocab_matrix, axis=1, keepdims=True)

        codebook_vectors = self.learner.codebook().detach().numpy()
        codebook_vectors = codebook_vectors / np.linalg.norm(codebook_vectors, axis=1, keepdims=True)

        similarities = np.dot(codebook_vectors, vocab_matrix.T)

        results = {}
        for i in range(len(codebook_vectors)):
            top_indices = np.argsort(similarities[i])[::-1][:top_k]
            seen_words = set()
            unique_results = []
            for idx in top_indices:
                word = unique_vocab[idx]
                if word not in seen_words:
                    seen_words.add(word)
                    unique_results.append({"word": word, "similarity": float(similarities[i][idx])})
                if len(unique_results) >= top_k:
                    break
            results[i] = unique_results

        return results

    def save(self, path: str):
        """Save the trained system"""
        torch.save(
            {
                "config": self.config,
                "codebook_state": self.learner.codebook.state_dict(),
                "encoder": self.encoder is not None,
                "decoder": self.decoder is not None,
                "corpus": self.corpus,
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
        from .config import LexicalConfig, InitMethod

        add_safe_globals([LexicalConfig, InitMethod])

        checkpoint = torch.load(path, weights_only=True)
        config = checkpoint["config"]

        system = cls(config, api_token, auto_initialize=False)
        system.learner.codebook.load_state_dict(checkpoint["codebook_state"])

        if checkpoint["encoder"]:
            system.encoder = LexicalEncoder(config, system.learner.codebook)
        if checkpoint["decoder"]:
            system.decoder = LexicalDecoder(config, system.learner.codebook)

        # Restore corpus vocabulary if present
        if "corpus" in checkpoint and checkpoint["corpus"] is not None:
            system.corpus = checkpoint["corpus"]

        return system
