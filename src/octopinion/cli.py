"""Command-line interface for Octopinion Lexical System"""

import argparse
import os
import sys
import json
from typing import List
import torch

from .config import LexicalConfig
from .system import LexicalSystem


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser"""
    parser = argparse.ArgumentParser(
        prog="octopinion",
        description="Octopinion Lexical System - A Vector Quantization-based encoding/decoding system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on a corpus file (one concept per line)
  octopinion train --corpus concepts.txt --epochs 100 --output model.pt

  # Encode text to syllable sequence
  octopinion encode "fish" --model model.pt

  # Decode syllable sequence to vector
  octopinion decode 3,7,2,1 --model model.pt

  # Interactive mode
  octopinion interactive --model model.pt

  # Analyze codebook
  octopinion analyze --model model.pt

  # Demo with synthetic data
  octopinion demo
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train codebook on a corpus")
    train_parser.add_argument(
        "--corpus",
        "-c",
        required=True,
        help="Path to corpus file (one concept per line)",
    )
    train_parser.add_argument("--output", "-o", required=True, help="Output model path")
    train_parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of training epochs")
    train_parser.add_argument("--batch-size", "-b", type=int, default=32, help="Batch size")
    train_parser.add_argument(
        "--codebook-size",
        "-s",
        type=int,
        default=26,
        help="Codebook size (number of syllables)",
    )
    train_parser.add_argument("--decay", "-d", type=float, default=0.5, help="Decay factor lambda")
    train_parser.add_argument("--max-length", "-l", type=int, default=5, help="Maximum word length")
    train_parser.add_argument(
        "--api-batch-size",
        type=int,
        default=10,
        help="Batch size for API calls (default: 10)",
    )
    train_parser.add_argument("--api-token", help="SiliconFlow API token (or set SILICONFLOW_API_TOKEN)")

    # Encode command
    encode_parser = subparsers.add_parser("encode", help="Encode text to syllable sequence")
    encode_parser.add_argument("text", help="Text to encode")
    encode_parser.add_argument("--model", "-m", required=True, help="Path to trained model")
    encode_parser.add_argument("--api-token", help="SiliconFlow API token")

    # Decode command
    decode_parser = subparsers.add_parser("decode", help="Decode syllable sequence")
    decode_parser.add_argument("sequence", help='Syllable sequence (comma-separated indices, e.g., "3,7,2,1")')
    decode_parser.add_argument("--model", "-m", required=True, help="Path to trained model")
    decode_parser.add_argument(
        "--output-format",
        "-f",
        choices=["vector", "summary"],
        default="summary",
        help="Output format",
    )

    # Interactive command
    interactive_parser = subparsers.add_parser("interactive", help="Interactive mode")
    interactive_parser.add_argument("--model", "-m", required=True, help="Path to trained model")
    interactive_parser.add_argument("--api-token", help="SiliconFlow API token")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze trained codebook")
    analyze_parser.add_argument("--model", "-m", required=True, help="Path to trained model")

    # Vocabulary command
    vocab_parser = subparsers.add_parser("vocabulary", help="Generate vocabulary from corpus")
    vocab_parser.add_argument("--corpus", "-c", required=True, help="Path to corpus file")
    vocab_parser.add_argument("--model", "-m", required=True, help="Path to trained model")
    vocab_parser.add_argument("--output", "-o", help="Output JSON file")
    vocab_parser.add_argument("--api-token", help="SiliconFlow API token")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demo with synthetic data")
    demo_parser.add_argument("--codebook-size", "-s", type=int, default=26, help="Codebook size")
    demo_parser.add_argument("--epochs", "-e", type=int, default=100, help="Training epochs")

    return parser


def load_corpus(path: str) -> List[str]:
    """Load corpus from file"""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def cmd_train(args):
    """Train command"""
    print("=" * 60)
    print("Octopinion - Training Codebook")
    print("=" * 60)

    # Load corpus
    print(f"\nLoading corpus from {args.corpus}...")
    corpus = load_corpus(args.corpus)
    print(f"Loaded {len(corpus)} concepts")

    # Create config
    config = LexicalConfig(
        codebook_size=args.codebook_size,
        decay_factor=args.decay,
        max_word_length=args.max_length,
        api_batch_size=args.api_batch_size,
    )

    # Create system
    api_token = args.api_token or os.getenv("SILICONFLOW_API_TOKEN")
    if not api_token:
        print("\nWarning: No API token provided. Set SILICONFLOW_API_TOKEN or use --api-token")
        print("Training will fail when fetching embeddings.")
        return 1

    system = LexicalSystem(config, api_token)

    # Train
    system.train(corpus, epochs=args.epochs, batch_size=args.batch_size)

    # Save
    system.save(args.output)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model saved to: {args.output}")
    print("=" * 60)

    return 0


def cmd_encode(args):
    """Encode command"""
    print("=" * 60)
    print("Octopinion - Encoding")
    print("=" * 60)

    # Load system
    api_token = args.api_token or os.getenv("SILICONFLOW_API_TOKEN")
    system = LexicalSystem.load(args.model, api_token)

    # Encode
    print(f"\nEncoding: '{args.text}'")
    try:
        sequence = system.encode_text(args.text)
        word = system.sequence_to_string(sequence)

        print(f"\nSyllable sequence: {sequence}")
        print(f"Word form: {word}")
        print(f"Length: {len(sequence)} syllables")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


def cmd_decode(args):
    """Decode command"""
    print("=" * 60)
    print("Octopinion - Decoding")
    print("=" * 60)

    # Load system
    system = LexicalSystem.load(args.model)

    # Parse sequence
    try:
        sequence = [int(x.strip()) for x in args.sequence.split(",")]
    except ValueError:
        print("Error: Sequence must be comma-separated integers (e.g., '3,7,2,1')")
        return 1

    print(f"\nDecoding sequence: {sequence}")

    # Decode
    try:
        vector = system.decode_sequence(sequence)
        word = system.sequence_to_string(sequence)

        print(f"\nWord form: {word}")
        print(f"Vector shape: {vector.shape}")
        print(f"Vector norm: {torch.norm(vector):.4f}")

        if args.output_format == "vector":
            print(f"\nVector values:")
            print(vector.tolist())

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


def cmd_interactive(args):
    """Interactive command"""
    print("=" * 60)
    print("Octopinion - Interactive Mode")
    print("=" * 60)

    # Load system
    api_token = args.api_token or os.getenv("SILICONFLOW_API_TOKEN")
    system = LexicalSystem.load(args.model, api_token)

    print("\nCommands:")
    print("  encode <text>  - Encode text to syllables")
    print("  decode <seq>   - Decode syllable sequence (comma-separated)")
    print("  quit           - Exit")
    print()

    while True:
        try:
            user_input = input("> ").strip()

            if not user_input:
                continue

            if user_input.lower() == "quit":
                break

            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("Error: Need command and argument")
                continue

            cmd, arg = parts

            if cmd == "encode":
                try:
                    sequence = system.encode_text(arg)
                    word = system.sequence_to_string(sequence)
                    print(f"  Sequence: {sequence}")
                    print(f"  Word: {word}")
                except Exception as e:
                    print(f"  Error: {e}")

            elif cmd == "decode":
                try:
                    sequence = [int(x.strip()) for x in arg.split(",")]
                    vector = system.decode_sequence(sequence)
                    word = system.sequence_to_string(sequence)
                    print(f"  Word: {word}")
                    print(f"  Vector norm: {torch.norm(vector):.4f}")
                except Exception as e:
                    print(f"  Error: {e}")

            else:
                print(f"Unknown command: {cmd}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            break

    return 0


def cmd_analyze(args):
    """Analyze command"""
    print("=" * 60)
    print("Octopinion - Codebook Analysis")
    print("=" * 60)

    # Load system
    system = LexicalSystem.load(args.model)

    # Analyze
    analysis = system.analyze_codebook()

    print("\nCodebook Statistics:")
    print(f"  Codebook size: {analysis['codebook_size']} syllables")
    print(f"  Embedding dimension: {analysis['embedding_dim']}")
    print(f"  Average pairwise similarity: {analysis['avg_pairwise_similarity']:.6f}")
    print(f"  Min similarity: {analysis['min_similarity']:.6f}")
    print(f"  Max similarity: {analysis['max_similarity']:.6f}")

    # Interpretation
    print("\nInterpretation:")
    avg_sim = analysis["avg_pairwise_similarity"]
    if avg_sim < 0.1:
        print("  Codebook vectors are well-distributed (low similarity)")
    elif avg_sim < 0.3:
        print("  Codebook vectors have moderate overlap")
    else:
        print("  Codebook vectors are highly similar (may need more training)")

    return 0


def cmd_vocabulary(args):
    """Vocabulary command"""
    print("=" * 60)
    print("Octopinion - Vocabulary Generation")
    print("=" * 60)

    # Load corpus
    print(f"\nLoading corpus from {args.corpus}...")
    corpus = load_corpus(args.corpus)
    print(f"Loaded {len(corpus)} concepts")

    # Load system
    api_token = args.api_token or os.getenv("SILICONFLOW_API_TOKEN")
    system = LexicalSystem.load(args.model, api_token)

    # Generate vocabulary
    print("\nGenerating vocabulary...")
    vocab = system.encode_corpus(corpus)

    # Count unique words
    unique_words = {}
    for concept, sequence in vocab.items():
        word = system.sequence_to_string(sequence)
        if word not in unique_words:
            unique_words[word] = []
        unique_words[word].append(concept)

    print(f"\nResults:")
    print(f"  Total concepts: {len(vocab)}")
    print(f"  Unique words: {len(unique_words)}")
    print(f"  Average concepts per word: {len(vocab) / len(unique_words):.2f}")

    # Show some examples
    print("\nSample mappings:")
    for concept, sequence in list(vocab.items())[:10]:
        word = system.sequence_to_string(sequence)
        print(f"  {concept:20} → {word}")

    # Show synonyms
    synonyms = [(w, c) for w, c in unique_words.items() if len(c) > 1]
    if synonyms:
        print(f"\nSynonyms found: {len(synonyms)} words")
        for word, concepts in synonyms[:5]:
            print(f"  {word}: {concepts}")

    # Save if requested
    if args.output:
        output_data = {
            "vocabulary": {k: system.sequence_to_string(v) for k, v in vocab.items()},
            "unique_words": unique_words,
            "statistics": {
                "total_concepts": len(vocab),
                "unique_words": len(unique_words),
                "synonyms": len(synonyms),
            },
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nVocabulary saved to {args.output}")

    return 0


def cmd_demo(args):
    """Demo command with synthetic data"""
    import numpy as np
    from .encoder import LexicalEncoder
    from .decoder import LexicalDecoder

    print("=" * 60)
    print("Octopinion - Demo with Synthetic Data")
    print("=" * 60)

    # Create config
    config = LexicalConfig(
        codebook_size=args.codebook_size,
        decay_factor=0.5,
        max_word_length=5,
        num_training_steps=4,
    )

    # Create system without API
    system = LexicalSystem(config)

    # Generate synthetic corpus
    torch.manual_seed(42)
    np.random.seed(42)

    num_concepts = 200
    num_clusters = 8
    print(f"\nGenerating {num_concepts} synthetic concept vectors...")
    print(f"Organized into {num_clusters} clusters")

    cluster_centers = torch.randn(num_clusters, config.embedding_dim)
    structured_corpus = []
    for i in range(num_concepts):
        cluster_idx = i % num_clusters
        noise = torch.randn(config.embedding_dim) * 0.3
        concept = cluster_centers[cluster_idx] + noise
        structured_corpus.append(concept)

    structured_corpus = torch.stack(structured_corpus)

    # Train
    print(f"\nTraining codebook ({config.codebook_size} syllables)...")
    optimizer = torch.optim.Adam(system.learner.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        indices = torch.randperm(num_concepts)[:32]
        batch = structured_corpus[indices]
        metrics = system.learner.train_step(batch, optimizer)

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {metrics['loss']:.6f}, Temp: {metrics['temperature']:.4f}")

    # Initialize encoder/decoder
    system.encoder = LexicalEncoder(config, system.learner.codebook)
    system.decoder = LexicalDecoder(config, system.learner.codebook)

    # Test encoding/decoding
    print("\n" + "=" * 60)
    print("Testing Encoding/Decoding")
    print("=" * 60)

    test_concepts = [0, 10, 25, 50, 75, 99]
    total_error = 0

    for concept_idx in test_concepts:
        target = structured_corpus[concept_idx]
        sequence = system.encoder.encode(target)
        decoded = system.decoder.decode(sequence)

        error = torch.norm(target - decoded).item()
        total_error += error

        print(f"\nConcept {concept_idx}: {system.sequence_to_string(sequence)}")
        print(f"  Reconstruction error: {error:.4f}")

    avg_error = total_error / len(test_concepts)
    print(f"\nAverage reconstruction error: {avg_error:.4f}")

    # Generate vocabulary
    print("\n" + "=" * 60)
    print("Vocabulary Generation")
    print("=" * 60)

    vocabulary = {}
    for i in range(num_concepts):
        target = structured_corpus[i]
        sequence = system.encoder.encode(target)
        word = system.sequence_to_string(sequence)

        if word not in vocabulary:
            vocabulary[word] = []
        vocabulary[word].append(i)

    print(f"\nGenerated {len(vocabulary)} unique words for {num_concepts} concepts")
    print(f"Average concepts per word: {num_concepts / len(vocabulary):.2f}")

    synonyms = [item for item in vocabulary.items() if len(item[1]) > 1]
    print(f"Words with multiple concepts (synonyms): {len(synonyms)}")

    # Analyze codebook
    print("\n" + "=" * 60)
    print("Codebook Analysis")
    print("=" * 60)
    analysis = system.analyze_codebook()
    for key, value in analysis.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)

    return 0


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Route to command handler
    commands = {
        "train": cmd_train,
        "encode": cmd_encode,
        "decode": cmd_decode,
        "interactive": cmd_interactive,
        "analyze": cmd_analyze,
        "vocabulary": cmd_vocabulary,
        "demo": cmd_demo,
    }

    if args.command in commands:
        return commands[args.command](args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
