"""Embedding provider - SiliconFlow API client with cache support"""

import os
import requests
import torch
from typing import List, Optional, Union, Dict, Any
from tqdm import tqdm
from .cache import EmbeddingCache


class SiliconFlowEmbedding:
    """Client for SiliconFlow Embedding API with optional cache"""

    def __init__(
        self,
        api_token: Optional[str] = None,
        model: Optional[str] = None,
        api_url: Optional[str] = None,
        cache_path: Optional[str] = None,
        use_cache: bool = True,
    ):
        self.api_token = api_token or os.getenv("SILICONFLOW_API_TOKEN")
        self.api_url = api_url or "https://api.siliconflow.cn/v1/embeddings"
        self.model = model or "BAAI/bge-large-zh-v1.5"

        # Initialize cache
        self.use_cache = use_cache
        if use_cache:
            cache_file = cache_path or ".octopinion_cache.db"
            self.cache = EmbeddingCache(cache_file, model=self.model)
        else:
            self.cache = None

    def _make_request(self, input_data: Union[str, List[str]]) -> dict:
        """Make API request and return parsed response"""
        if not self.api_token:
            raise ValueError("API token not provided. Set SILICONFLOW_API_TOKEN environment variable.")

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }

        data = {"model": self.model, "input": input_data}

        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {e}")

    def get_embedding(self, text: str, use_cache: Optional[bool] = None) -> torch.Tensor:
        """
        Get embedding vector for single text.

        Args:
            text: Input text
            use_cache: Override cache setting for this call

        Returns:
            Embedding tensor
        """
        should_cache = use_cache if use_cache is not None else self.use_cache

        # Check cache first
        if should_cache and self.cache:
            cached = self.cache.get(text)
            if cached is not None:
                return cached

        # Fetch from API
        result = self._make_request(text)

        try:
            if "data" in result and len(result["data"]) > 0:
                embedding = result["data"][0]["embedding"]
                embedding_tensor = torch.tensor(embedding, dtype=torch.float32)

                # Cache the result
                if should_cache and self.cache:
                    self.cache.set(text, embedding_tensor)

                return embedding_tensor
            else:
                raise ValueError(f"Unexpected API response format: {result}")
        except (KeyError, IndexError) as e:
            raise ValueError(f"Failed to parse API response: {e}")

    def get_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 30,
        show_progress: bool = False,
        use_cache: Optional[bool] = None,
    ) -> List[torch.Tensor]:
        """
        Get embeddings for multiple texts using batch API with cache support.

        Args:
            texts: List of texts to encode
            batch_size: Number of texts per API call
            show_progress: Whether to show progress
            use_cache: Override cache setting for this call

        Returns:
            List of embedding tensors in the same order as input texts
        """
        if not texts:
            return []

        should_cache = use_cache if use_cache is not None else self.use_cache

        # If caching enabled, check cache first
        if should_cache and self.cache:
            cached_embeddings = self.cache.get_batch(texts)

            # Find missing texts
            missing_texts = []
            missing_indices = []
            for i, emb in enumerate(cached_embeddings):
                if emb is None:
                    missing_texts.append(texts[i])
                    missing_indices.append(i)

            if show_progress and missing_texts:
                cached_count = len(texts) - len(missing_texts)
                print(f"  Cache hits: {cached_count}/{len(texts)}, fetching {len(missing_texts)} from API")

            # Initialize result with cached embeddings
            result_embeddings: List[Optional[torch.Tensor]] = [None] * len(texts)
            for i, emb in enumerate(cached_embeddings):
                if emb is not None:
                    result_embeddings[i] = emb

            # Fetch missing from API if any
            if missing_texts:
                fetched = self._fetch_from_api(missing_texts, batch_size, show_progress)

                # Fill in missing spots
                for idx, emb in zip(missing_indices, fetched):
                    result_embeddings[idx] = emb

                # Cache the newly fetched embeddings
                self.cache.set_batch(missing_texts, fetched)

            # Convert None to zeros (shouldn't happen if logic is correct)
            final_embeddings: List[torch.Tensor] = []
            for emb in result_embeddings:
                if emb is None:
                    final_embeddings.append(torch.zeros(self.get_embedding_dim()))
                else:
                    final_embeddings.append(emb)

            return final_embeddings

        # No cache - fetch all from API
        return self._fetch_from_api(texts, batch_size, show_progress)

    def _fetch_from_api(self, texts: List[str], batch_size: int, show_progress: bool) -> List[torch.Tensor]:
        """Fetch embeddings from API (internal method)"""
        all_embeddings = []
        total = len(texts)
        num_batches = (total + batch_size - 1) // batch_size

        iterator = (
            tqdm(range(0, total, batch_size), desc="Fetching embeddings", unit="batch")
            if show_progress
            else range(0, total, batch_size)
        )

        for i in iterator:
            batch = texts[i : i + batch_size]

            try:
                result = self._make_request(batch)

                if "data" not in result:
                    raise ValueError(f"Unexpected API response: {result}")

                data_items = sorted(result["data"], key=lambda x: x.get("index", 0))

                for item in data_items:
                    embedding = item["embedding"]
                    all_embeddings.append(torch.tensor(embedding, dtype=torch.float32))

            except Exception as e:
                print(f"Warning: Batch request failed ({i}-{i + len(batch)}): {e}")
                for _ in batch:
                    all_embeddings.append(torch.zeros(self.get_embedding_dim()))

        return all_embeddings

    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings from this model"""
        return 1024

    def cache_stats(self) -> dict:
        """Get cache statistics"""
        if self.cache:
            return self.cache.stats()
        return {"count": 0, "cache_disabled": True}

    def clear_cache(self, model: Optional[str] = None):
        """Clear the embedding cache"""
        if self.cache:
            self.cache.clear(model)
            print(f"Cache cleared for model: {model or 'all'}")
