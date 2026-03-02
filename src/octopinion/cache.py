"""Embedding cache using SQLite"""

import sqlite3
import hashlib
import os
from typing import List, Optional, Tuple
import torch


class EmbeddingCache:
    """SQLite-based embedding cache"""

    def __init__(self, cache_path: str = ".octopinion_cache.db", model: str = "default"):
        """
        Initialize embedding cache.

        Args:
            cache_path: Path to SQLite database file
            model: Embedding model identifier (e.g., "BAAI/bge-large-zh-v1.5")
        """
        self.cache_path = cache_path
        self.model = model
        self._init_db()

    def _init_db(self):
        """Create database and table if not exists"""
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()

        # Create table with model-specific storage
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                dimension INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(model, text_hash)
            )
        """)

        # Create index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_hash 
            ON embeddings(model, text_hash)
        """)

        conn.commit()
        conn.close()

    def _hash_text(self, text: str) -> str:
        """Generate hash for text"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def get(self, text: str) -> Optional[torch.Tensor]:
        """
        Get embedding from cache.

        Args:
            text: Input text

        Returns:
            Embedding tensor if cached, None otherwise
        """
        text_hash = self._hash_text(text)

        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT embedding, dimension 
            FROM embeddings 
            WHERE model = ? AND text_hash = ?
        """,
            (self.model, text_hash),
        )

        result = cursor.fetchone()
        conn.close()

        if result:
            import numpy as np

            embedding_bytes, dimension = result
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            return torch.tensor(embedding, dtype=torch.float32)

        return None

    def get_batch(self, texts: List[str]) -> List[Optional[torch.Tensor]]:
        """
        Get multiple embeddings from cache.

        Args:
            texts: List of input texts

        Returns:
            List of embedding tensors (None if not cached)
        """
        if not texts:
            return []

        text_hashes = [(self.model, self._hash_text(t)) for t in texts]

        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()

        # Query all at once
        placeholders = ",".join(["(?, ?)"] * len(texts))
        query = f"""
            SELECT text_hash, embedding, dimension 
            FROM embeddings 
            WHERE (model, text_hash) IN ({placeholders})
        """

        params = []
        for model, h in text_hashes:
            params.extend([model, h])

        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()

        # Build lookup dict
        import numpy as np

        cache = {}
        for text_hash, emb_bytes, dimension in results:
            embedding = np.frombuffer(emb_bytes, dtype=np.float32)
            cache[text_hash] = torch.tensor(embedding, dtype=torch.float32)

        # Return in original order
        return [cache.get(self._hash_text(t)) for t in texts]

    def set(self, text: str, embedding: torch.Tensor):
        """
        Store embedding in cache.

        Args:
            text: Input text
            embedding: Embedding tensor
        """
        text_hash = self._hash_text(text)

        # Convert tensor to bytes
        import numpy as np

        embedding_bytes = embedding.cpu().numpy().astype(np.float32).tobytes()
        dimension = embedding.shape[0]

        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO embeddings 
            (model, text_hash, text, embedding, dimension)
            VALUES (?, ?, ?, ?, ?)
        """,
            (self.model, text_hash, text, embedding_bytes, dimension),
        )

        conn.commit()
        conn.close()

    def set_batch(self, texts: List[str], embeddings: List[torch.Tensor]):
        """
        Store multiple embeddings in cache.

        Args:
            texts: List of input texts
            embeddings: List of embedding tensors
        """
        if not texts:
            return

        import numpy as np

        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()

        data = []
        for text, embedding in zip(texts, embeddings):
            text_hash = self._hash_text(text)
            embedding_bytes = embedding.cpu().numpy().astype(np.float32).tobytes()
            dimension = embedding.shape[0]
            data.append((self.model, text_hash, text, embedding_bytes, dimension))

        cursor.executemany(
            """
            INSERT OR REPLACE INTO embeddings 
            (model, text_hash, text, embedding, dimension)
            VALUES (?, ?, ?, ?, ?)
        """,
            data,
        )

        conn.commit()
        conn.close()

    def get_missing(self, texts: List[str]) -> Tuple[List[str], List[int]]:
        """
        Get texts that are not in cache.

        Args:
            texts: List of input texts

        Returns:
            Tuple of (missing texts, their original indices)
        """
        if not texts:
            return [], []

        # Get all from cache
        cached = self.get_batch(texts)

        missing_texts = []
        missing_indices = []

        for i, (text, emb) in enumerate(zip(texts, cached)):
            if emb is None:
                missing_texts.append(text)
                missing_indices.append(i)

        return missing_texts, missing_indices

    def stats(self) -> dict:
        """Get cache statistics"""
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT COUNT(*), SUM(dimension), model 
            FROM embeddings 
            WHERE model = ?
            GROUP BY model
        """,
            (self.model,),
        )

        result = cursor.fetchone()
        conn.close()

        if result:
            count, total_dim, _ = result
            return {"count": count, "total_dimensions": total_dim, "model": self.model, "cache_path": self.cache_path}

        return {"count": 0, "model": self.model, "cache_path": self.cache_path}

    def clear(self, model: Optional[str] = None):
        """Clear cache for specific model or all"""
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()

        if model:
            cursor.execute("DELETE FROM embeddings WHERE model = ?", (model,))
        else:
            cursor.execute("DELETE FROM embeddings")

        conn.commit()
        conn.close()

    def get_all(self) -> List[Tuple[str, torch.Tensor]]:
        """
        Get all cached embeddings.

        Returns:
            List of (text, embedding) tuples
        """
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT text, embedding, dimension FROM embeddings WHERE model = ?",
            (self.model,),
        )

        results = cursor.fetchall()
        conn.close()

        import numpy as np

        return [
            (text, torch.tensor(np.frombuffer(emb_bytes, dtype=np.float32), dtype=torch.float32))
            for text, emb_bytes, _ in results
        ]
