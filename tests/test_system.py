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
        assert config.decay_factor == 0.7
        assert config.max_word_length == 5
        assert config.allow_negative_signs == True


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
        config = LexicalConfig(codebook_size=10, embedding_dim=32, max_word_length=3, allow_negative_signs=False)
        codebook = Codebook(config)
        encoder = LexicalEncoder(config, codebook)

        target = torch.randn(32)
        sequence = encoder.encode(target)

        assert isinstance(sequence, list)
        assert len(sequence) <= config.max_word_length
        assert all(isinstance(idx, int) for idx in sequence)
        assert all(0 <= idx < config.codebook_size for idx in sequence)

    def test_encode_signed_sequence(self):
        """Test encoding with signed sequences enabled"""
        config = LexicalConfig(codebook_size=10, embedding_dim=32, max_word_length=4, allow_negative_signs=True)
        codebook = Codebook(config)
        encoder = LexicalEncoder(config, codebook)

        # Create a target vector
        target = torch.randn(32)
        sequence = encoder.encode(target)

        assert isinstance(sequence, list)
        assert len(sequence) <= config.max_word_length
        assert all(isinstance(idx, int) for idx in sequence)

        # Check that indices can be negative or positive
        for idx in sequence:
            if idx < 0:
                # Negative index should be in range [-codebook_size, -1]
                assert -config.codebook_size <= idx <= -1
            else:
                # Positive index should be in range [0, codebook_size-1]
                assert 0 <= idx < config.codebook_size

    def test_signed_encoding_reduces_residual(self):
        """Test that signed encoding actually reduces the residual"""
        config = LexicalConfig(codebook_size=10, embedding_dim=32, max_word_length=3, allow_negative_signs=True)
        codebook = Codebook(config)
        encoder = LexicalEncoder(config, codebook)
        decoder = LexicalDecoder(config, codebook)

        target = torch.randn(32)
        initial_norm = torch.norm(target).item()

        # Encode
        sequence = encoder.encode(target)

        # Decode
        decoded = decoder.decode(sequence)

        # Compute residual
        residual = target - decoded
        residual_norm = torch.norm(residual).item()

        # Residual should be smaller than original
        assert residual_norm < initial_norm


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

    def test_decode_signed_sequence(self):
        """Test decoding with negative indices (signed sequence)"""
        config = LexicalConfig(codebook_size=10, embedding_dim=32, allow_negative_signs=True)
        codebook = Codebook(config)
        decoder = LexicalDecoder(config, codebook)

        # Test positive sequence (original behavior)
        sequence_pos = [0, 1, 2]
        vector_pos = decoder.decode(sequence_pos)
        assert vector_pos.shape == (32,)

        # Test signed sequence with negative indices
        # -1 means syllable 0 with negative sign
        # -2 means syllable 1 with negative sign
        sequence_signed = [0, -2, 3]  # +v0, -v1, +v3
        vector_signed = decoder.decode(sequence_signed)
        assert vector_signed.shape == (32,)

        # Verify that negative index subtracts instead of adds
        # -2 maps to index 1 with sign -1
        vec0 = codebook.get_vector(0)
        vec1 = codebook.get_vector(1)
        vec3 = codebook.get_vector(3)

        decay = config.decay_factor
        expected = decay**1 * vec0 - decay**2 * vec1 + decay**3 * vec3
        assert torch.allclose(vector_signed, expected, atol=1e-6)


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
        reconstructed, total_loss, recon_loss, distributions = learner(targets)

        assert reconstructed.shape == (8, 32)
        assert isinstance(total_loss.item(), float)
        assert isinstance(recon_loss.item(), float)
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

    def test_sequence_to_string_signed(self):
        """Test sequence_to_string with signed sequences"""
        config = LexicalConfig()
        system = LexicalSystem(config)

        # Test signed sequence
        # -1 represents index 0 with negative sign
        # -2 represents index 1 with negative sign
        assert system.sequence_to_string([1, -2, 3]) == "S1--S1-S3"
        assert system.sequence_to_string([-1, -2, -3]) == "-S0--S1--S2"
        assert system.sequence_to_string([0, 1, -3]) == "S0-S1--S2"


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
