"""Embedding provider - SiliconFlow API client"""

import os
import requests
import torch
from typing import List, Optional, Union


class SiliconFlowEmbedding:
    """Client for SiliconFlow Embedding API with batch support"""

    def __init__(
        self,
        api_token: Optional[str] = None,
        model: Optional[str] = None,
        api_url: Optional[str] = None,
    ):
        self.api_token = api_token or os.getenv("SILICONFLOW_API_TOKEN")
        self.api_url = api_url or "https://api.siliconflow.cn/v1/embeddings"
        self.model = model or "BAAI/bge-large-zh-v1.5"

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

    def get_embedding(self, text: str) -> torch.Tensor:
        """Get embedding vector for single text"""
        result = self._make_request(text)

        try:
            if "data" in result and len(result["data"]) > 0:
                embedding = result["data"][0]["embedding"]
                return torch.tensor(embedding, dtype=torch.float32)
            else:
                raise ValueError(f"Unexpected API response format: {result}")
        except (KeyError, IndexError) as e:
            raise ValueError(f"Failed to parse API response: {e}")

    def get_embeddings_batch(
        self, texts: List[str], batch_size: int = 10, show_progress: bool = False
    ) -> List[torch.Tensor]:
        """
        Get embeddings for multiple texts using batch API.

        Args:
            texts: List of texts to encode
            batch_size: Number of texts per API call (default: 100)
            show_progress: Whether to show progress

        Returns:
            List of embedding tensors in the same order as input texts
        """
        if not texts:
            return []

        all_embeddings = []
        total = len(texts)

        # Process in batches
        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]

            if show_progress:
                print(f"  Progress: {i}/{total} (batch size: {len(batch)})")

            try:
                # Make batch request - API accepts string[] in input field
                result = self._make_request(batch)

                # Parse response - returns array of embeddings
                if "data" not in result:
                    raise ValueError(f"Unexpected API response: {result}")

                # Sort by index to maintain order
                data_items = sorted(result["data"], key=lambda x: x.get("index", 0))

                for item in data_items:
                    embedding = item["embedding"]
                    all_embeddings.append(torch.tensor(embedding, dtype=torch.float32))

            except Exception as e:
                print(f"Warning: Batch request failed ({i}-{i + len(batch)}): {e}")
                # Fallback: return zero vectors for this batch
                for _ in batch:
                    all_embeddings.append(torch.zeros(self.get_embedding_dim()))

        return all_embeddings

    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings from this model"""
        # BAAI/bge-large-zh-v1.5 produces 1024-dimensional embeddings
        return 1024
