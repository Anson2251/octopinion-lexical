"""Command-line interface for Octopinion Lexical System"""

import argparse
import os
import sys
import json
from typing import List
import torch
from tqdm import tqdm

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

  # Decode syllable sequence to closest word
  octopinion decode 3,7,2,1 --model model.pt

  # Decode with custom vocabulary
  octopinion decode 3,7,2,1 --model model.pt --vocabulary words.txt

  # Interactive mode
  octopinion interactive --model model.pt

  # Interactive mode with vocabulary
  octopinion interactive --model model.pt --vocabulary words.txt

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
    train_parser.add_argument("--decay", "-d", type=float, default=LexicalConfig.decay_factor, help="Decay factor lambda")
    train_parser.add_argument("--max-length", "-l", type=int, default=LexicalConfig.max_word_length, help="Maximum word length")
    train_parser.add_argument(
        "--api-batch-size",
        type=int,
        default=30,
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
    decode_parser.add_argument("--api-token", help="SiliconFlow API token")
    decode_parser.add_argument("--vocabulary", "-v", help="Path to vocabulary file (one word per line)")
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
    interactive_parser.add_argument("--vocabulary", "-v", help="Path to vocabulary file (one word per line)")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze trained codebook")
    analyze_parser.add_argument("--model", "-m", required=True, help="Path to trained model")

    # Export codebook words command
    export_parser = subparsers.add_parser("export-codebook", help="Export words for each codebook item")
    export_parser.add_argument("--model", "-m", required=True, help="Path to trained model")
    export_parser.add_argument("--vocabulary", "-v", required=True, help="Path to vocabulary file (one word per line)")
    export_parser.add_argument("--top-k", "-k", type=int, default=10, help="Number of words per codebook item")
    export_parser.add_argument("--output", "-o", help="Output JSON file (default: print to stdout)")
    export_parser.add_argument("--api-token", help="SiliconFlow API token")

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

    # Cache command
    cache_parser = subparsers.add_parser("cache", help="Manage embedding cache")
    cache_parser.add_argument("--stats", action="store_true", help="Show cache statistics")
    cache_parser.add_argument("--clear", action="store_true", help="Clear cache")
    cache_parser.add_argument("--model", default="BAAI/bge-large-zh-v1.5", help="Embedding model")

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
    api_token = args.api_token or os.getenv("SILICONFLOW_API_TOKEN")
    system = LexicalSystem.load(args.model, api_token)

    # Load vocabulary if provided
    vocabulary = None
    if args.vocabulary:
        print(f"\nLoading vocabulary from {args.vocabulary}...")
        with open(args.vocabulary, "r", encoding="utf-8") as f:
            vocabulary = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(vocabulary)} words")

    # Parse sequence
    try:
        sequence = [int(x.strip()) for x in args.sequence.split(",")]
    except ValueError:
        print("Error: Sequence must be comma-separated integers (e.g., '3,7,2,1')")
        return 1

    print(f"\nDecoding sequence: {sequence}")
    print(f"Word form: {system.sequence_to_string(sequence)}")

    # Decode
    try:
        result = system.decode_to_text(sequence, vocabulary=vocabulary)

        print(f"\nClosest match: {result['word']}")
        print(f"Similarity: {result['similarity']:.4f}")
        print(f"Vector norm: {torch.norm(result['vector']):.4f}")

        print("\nTop 5 matches:")
        for word, sim in result["all_words"][:5]:
            print(f"  {word}: {sim:.4f}")

        if args.output_format == "vector":
            print(f"\nVector values:")
            print(result["vector"].tolist())

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

    # Load vocabulary if provided
    vocabulary = None
    if args.vocabulary:
        print(f"\nLoading vocabulary from {args.vocabulary}...")
        with open(args.vocabulary, "r", encoding="utf-8") as f:
            vocabulary = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(vocabulary)} words")

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
                    result = system.decode_to_text(sequence, vocabulary=vocabulary)
                    print(f"  Word: {result['word']}")
                    print(f"  Similarity: {result['similarity']:.4f}")
                    print(f"  Vector norm: {torch.norm(result['vector']):.4f}")
                    print(f"  Top 5 matches:")
                    for word, sim in result["all_words"][:5]:
                        print(f"    {word}: {sim:.4f}")
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


def cmd_export_codebook(args):
    """Export codebook words command"""
    import json
    from .system import LexicalSystem

    print("=" * 60)
    print("Octopinion - Export Codebook Words")
    print("=" * 60)

    # Load vocabulary
    print(f"\nLoading vocabulary from {args.vocabulary}...")
    with open(args.vocabulary, "r", encoding="utf-8") as f:
        vocabulary = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(vocabulary)} words")

    # Set API token if provided
    if args.api_token:
        import os

        os.environ["SILICONFLOW_API_TOKEN"] = args.api_token

    # Load system
    print(f"\nLoading model from {args.model}...")
    system = LexicalSystem.load(args.model)

    # Export codebook words
    print(f"\nFinding top {args.top_k} words for each codebook item...")
    results = system.export_codebook_words(vocabulary, top_k=args.top_k, show_progress=True)

    # Output results
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.output}")
    else:
        print("\nCodebook Words:")
        for idx, words in results.items():
            print(f"  [{idx}]: {', '.join([w['word'] for w in words])}")

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
    optimizer = torch.optim.SGD(system.learner.parameters(), lr=config.learning_rate, momentum=config.momentum)

    for epoch in tqdm(range(args.epochs), desc="Training"):
        indices = torch.randperm(num_concepts)[:32]
        batch = structured_corpus[indices]
        metrics = system.learner.train_step(batch, optimizer)

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


def cmd_cache(args):
    """Cache management command"""
    from .embedder import SiliconFlowEmbedding

    print("=" * 60)
    print("Octopinion - Cache Management")
    print("=" * 60)

    # Create embedder to access cache
    embedder = SiliconFlowEmbedding(model=args.model)

    if args.stats:
        print("\nCache Statistics:")
        stats = embedder.cache_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")

    if args.clear:
        print("\nClearing cache...")
        embedder.clear_cache(args.model)
        print("Cache cleared!")

    if not args.stats and not args.clear:
        print("\nUse --stats to show cache statistics or --clear to clear cache")

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
        "export-codebook": cmd_export_codebook,
        "vocabulary": cmd_vocabulary,
        "demo": cmd_demo,
        "cache": cmd_cache,
    }

    if args.command in commands:
        return commands[args.command](args)
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
