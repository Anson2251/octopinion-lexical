"""Tests for the Octopinion Lexical System"""

import torch
import pytest
from octopinion import LexicalConfig, LexicalSystem
from octopinion.config import LexicalConfig
from octopinion.codebook import Codebook
from octopinion.encoder import LexicalEncoder
from octopinion.decoder import LexicalDecoder
from octopinion.learner import CodebookLearner, GumbelSoftmax


class TestConfig:
    def test_default_config(self):
        config = LexicalConfig()
        assert config.codebook_size == 64
        assert config.embedding_dim == 1024
        assert config.decay_factor == 0.5
        assert config.max_word_length == 5


class TestCodebook:
    def test_codebook_initialization(self):
        config = LexicalConfig(codebook_size=10, embedding_dim=64)
        codebook = Codebook(config)

        assert codebook.vectors.shape == (10, 64)

        # Check normalization
        vecs = codebook()
        norms = torch.norm(vecs, dim=1)
        assert torch.allclose(norms, torch.ones(10), atol=1e-6)

    def test_get_vector(self):
        config = LexicalConfig(codebook_size=5, embedding_dim=32)
        codebook = Codebook(config)

        vec = codebook.get_vector(0)
        assert vec.shape == (32,)
        assert torch.abs(torch.norm(vec) - 1.0) < 1e-6


class TestEncoder:
    def test_encode_shape(self):
        config = LexicalConfig(codebook_size=10, embedding_dim=32, max_word_length=3)
        codebook = Codebook(config)
        encoder = LexicalEncoder(config, codebook)

        target = torch.randn(32)
        sequence = encoder.encode(target)

        assert isinstance(sequence, list)
        assert len(sequence) <= config.max_word_length
        assert all(isinstance(idx, int) for idx in sequence)
        assert all(0 <= idx < config.codebook_size for idx in sequence)


class TestDecoder:
    def test_decode_shape(self):
        config = LexicalConfig(codebook_size=10, embedding_dim=32)
        codebook = Codebook(config)
        decoder = LexicalDecoder(config, codebook)

        sequence = [0, 1, 2]
        vector = decoder.decode(sequence)

        assert vector.shape == (32,)

    def test_decode_empty(self):
        config = LexicalConfig(codebook_size=10, embedding_dim=32)
        codebook = Codebook(config)
        decoder = LexicalDecoder(config, codebook)

        vector = decoder.decode([])
        assert vector.shape == (32,)
        assert torch.allclose(vector, torch.zeros(32))


class TestGumbelSoftmax:
    def test_gumbel_shape(self):
        gs = GumbelSoftmax(temperature=1.0)
        logits = torch.randn(10, 5)

        result = gs(logits)
        assert result.shape == (10, 5)

        # Check it's a valid distribution
        sums = result.sum(dim=1)
        assert torch.allclose(sums, torch.ones(10), atol=1e-5)


class TestCodebookLearner:
    def test_forward_shape(self):
        config = LexicalConfig(codebook_size=10, embedding_dim=32)
        learner = CodebookLearner(config)

        targets = torch.randn(8, 32)
        reconstructed, loss, distributions = learner(targets)

        assert reconstructed.shape == (8, 32)
        assert isinstance(loss.item(), float)
        assert len(distributions) == config.num_training_steps


class TestLexicalSystem:
    def test_system_initialization(self):
        config = LexicalConfig(codebook_size=10, embedding_dim=32)
        system = LexicalSystem(config)

        assert system.config == config
        assert system.encoder is None
        assert system.decoder is None

    def test_sequence_to_string(self):
        config = LexicalConfig()
        system = LexicalSystem(config)

        assert system.sequence_to_string([1, 2, 3]) == "S1-S2-S3"
        assert system.sequence_to_string([]) == ""
        assert system.sequence_to_string([0]) == "S0"


class TestIntegration:
    def test_encode_decode_roundtrip(self):
        """Test that encode -> decode is consistent"""
        config = LexicalConfig(codebook_size=20, embedding_dim=64, max_word_length=4, decay_factor=0.5)

        # Create system
        system = LexicalSystem(config)

        # Generate synthetic data and train briefly
        corpus = torch.randn(50, 64)
        optimizer = torch.optim.SGD(system.learner.parameters(), lr=0.01, momentum=0.9)

        for _ in range(10):
            indices = torch.randperm(50)[:16]
            batch = corpus[indices]
            system.learner.train_step(batch, optimizer)

        # Setup encoder/decoder
        system.encoder = LexicalEncoder(config, system.learner.codebook)
        system.decoder = LexicalDecoder(config, system.learner.codebook)

        # Test roundtrip
        target = corpus[0]
        sequence = system.encoder.encode(target)
        decoded = system.decoder.decode(sequence)

        # Should produce valid output
        assert len(sequence) > 0
        assert decoded.shape == target.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
