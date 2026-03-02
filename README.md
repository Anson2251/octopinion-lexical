# Octopinion Lexical System

A Vector Quantization-based encoding/decoding system for the Octopinion constructed language.

## Overview

This system implements the lexical system described in `spec.md`, which encodes semantic concepts as sequences of syllables using:

- **Greedy Residual Pursuit** encoding algorithm
- **Gumbel-Softmax** codebook learning
- **SiliconFlow API** for semantic embeddings

## Installation

```bash
# Clone the repository
git clone <repo>
cd octopinion-lexical

# Install dependencies
uv pip install -e .
```

## Quick Start

### 1. Set API Token

Get your API token from [SiliconFlow](https://siliconflow.cn) and set it:

```bash
export SILICONFLOW_API_TOKEN="your_token_here"
```

### 2. Train a Model

Create a corpus file with one concept per line:

```bash
cat > concepts.txt << EOF
fish
crab
shrimp
coral
rock
water
ocean
hunt
prey
camouflage
EOF
```

Train the codebook:

```bash
octopinion train --corpus concepts.txt --epochs 100 --output model.pt
```

### 3. Use the Model

```bash
# Encode text
octopinion encode "fish" --model model.pt

# Decode sequence
octopinion decode "3,7,2,1" --model model.pt

# Interactive mode
octopinion interactive --model model.pt

# Analyze codebook
octopinion analyze --model model.pt
```

## CLI Commands

### `train`
Train a codebook on a corpus:

```bash
octopinion train \
  --corpus concepts.txt \
  --output model.pt \
  --epochs 100 \
  --codebook-size 26 \
  --decay 0.5 \
  --max-length 5
```

Options:
- `--corpus, -c`: Path to corpus file (one concept per line) [required]
- `--output, -o`: Output model path [required]
- `--epochs, -e`: Number of training epochs (default: 100)
- `--batch-size, -b`: Batch size (default: 32)
- `--codebook-size, -s`: Number of syllables (default: 26)
- `--decay, -d`: Decay factor lambda (default: 0.5)
- `--max-length, -l`: Maximum word length (default: 5)
- `--api-token`: SiliconFlow API token

### `encode`
Encode text to syllable sequence:

```bash
octopinion encode "fish" --model model.pt
```

Output:
```
Syllable sequence: [3, 7, 2, 1]
Word form: S3-S7-S2-S1
Length: 4 syllables
```

### `decode`
Decode syllable sequence to vector:

```bash
octopinion decode "3,7,2,1" --model model.pt
```

### `interactive`
Interactive mode for encoding/decoding:

```bash
octopinion interactive --model model.pt
```

Commands:
- `encode <text>` - Encode text to syllables
- `decode <seq>` - Decode syllable sequence (comma-separated)
- `quit` - Exit

### `analyze`
Analyze trained codebook statistics:

```bash
octopinion analyze --model model.pt
```

### `vocabulary`
Generate vocabulary mapping from corpus:

```bash
octopinion vocabulary \
  --corpus concepts.txt \
  --model model.pt \
  --output vocab.json
```

### `demo`
Run demo with synthetic data (no API needed):

```bash
octopinion demo --codebook-size 26 --epochs 100
```

## Python API

```python
from octopinion import LexicalSystem, LexicalConfig

# Create configuration
config = LexicalConfig(
    codebook_size=26,      # Number of syllables
    embedding_dim=1024,    # Matches BGE-large
    decay_factor=0.5,      # Lambda
    max_word_length=5      # Max syllables per word
)

# Initialize system
system = LexicalSystem(config, api_token="your_token")

# Train
corpus = ["fish", "crab", "hunt", "prey"]
system.train(corpus, epochs=100)

# Encode
sequence = system.encode_text("fish")
print(f"Sequence: {sequence}")  # [3, 7, 2, 1]
print(f"Word: {system.sequence_to_string(sequence)}")  # S3-S7-S2-S1

# Decode
vector = system.decode_sequence(sequence)
print(f"Vector shape: {vector.shape}")

# Save/Load
system.save("model.pt")
loaded_system = LexicalSystem.load("model.pt")
```

## Project Structure

```
octopinion-lexical/
├── src/
│   └── octopinion/
│       ├── __init__.py      # Package exports
│       ├── cli.py           # Command-line interface
│       ├── config.py        # Configuration
│       ├── embedder.py      # SiliconFlow API client
│       ├── codebook.py      # Learnable codebook
│       ├── encoder.py       # Greedy residual pursuit
│       ├── decoder.py       # Linear composition
│       ├── learner.py       # Gumbel-Softmax training
│       └── system.py        # Main system integration
├── tests/                   # Test suite
├── spec.md                  # Full specification
├── pyproject.toml           # Package config
└── README.md                # This file
```

## Architecture

The system consists of four main components:

1. **Encoder** (`encoder.py`): Greedy residual pursuit algorithm
   - Iteratively selects syllables that best reduce the residual
   - Uses dot product similarity to find best matches
   - Applies positional decay (λ^step)

2. **Decoder** (`decoder.py`): Linear composition
   - Weighted sum of syllable vectors
   - Deterministic reconstruction

3. **Codebook** (`codebook.py`): Learnable syllable vectors
   - Unit-normalized semantic vectors
   - Initialized randomly, trained via backprop

4. **Learner** (`learner.py`): Training with Gumbel-Softmax
   - Differentiable discrete sampling
   - Annealed temperature for convergence
   - MSE reconstruction loss

## Mathematical Foundation

### Encoding

For target vector **t**, find sequence σ = ⟨s₁, s₂, ..., sₖ⟩:

```
residual = t
for step in 0..K-1:
    scores[i] = dot(residual, v_i) for all i
    best = argmax(scores)
    residual -= λ^step * v_best
```

### Decoding

From sequence σ, reconstruct meaning:

```
meaning(σ) = Σ λ^(j-1) * v_σ[j]
```

### Training

Use Gumbel-Softmax to make discrete selection differentiable:

```
gumbel_dist = GumbelSoftmax(dot(residual, codebook))
reconstructed = Σ λ^step * gumbel_dist @ codebook
loss = ||target - reconstructed||²
```

## License

MIT
