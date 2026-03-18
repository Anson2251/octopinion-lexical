#!/usr/bin/env python3
"""
Export dot product matrix of codebook vectors.

This script computes the pairwise dot products between all codebook vectors
and exports the result as a JSON file.
"""

import argparse
import json
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.octopinion.system import LexicalSystem


def export_dot_product(model_path: str, output_path: str, normalize: bool = False, plot_path: str = None):
    """
    Export dot product matrix of codebook vectors.

    Args:
        model_path: Path to trained model
        output_path: Path to output JSON file
        normalize: If True, normalize vectors before computing dot products
    """
    print(f"Loading model from {model_path}...")
    import torch

    checkpoint = torch.load(model_path, weights_only=False)
    from src.octopinion.config import LexicalConfig

    config = checkpoint["config"]
    from src.octopinion.system import LexicalSystem

    system = LexicalSystem(config, auto_initialize=False)
    system.learner.codebook.load_state_dict(checkpoint["codebook_state"])
    if checkpoint["encoder"]:
        from src.octopinion.encoder import LexicalEncoder

        system.encoder = LexicalEncoder(config, system.learner.codebook)
    if checkpoint["decoder"]:
        from src.octopinion.decoder import LexicalDecoder

        system.decoder = LexicalDecoder(config, system.learner.codebook)
    if "corpus" in checkpoint and checkpoint["corpus"] is not None:
        system.corpus = checkpoint["corpus"]

    print("Computing dot product matrix...")
    codebook = system.learner.codebook().detach().numpy()

    if normalize:
        codebook = codebook / (np.linalg.norm(codebook, axis=1, keepdims=True) + 1e-8)

    dot_product_matrix = np.dot(codebook, codebook.T).tolist()

    result = {
        "matrix": dot_product_matrix,
        "size": len(dot_product_matrix),
        "normalized": normalize,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Dot product matrix saved to {output_path}")
    print(f"Matrix size: {len(dot_product_matrix)}x{len(dot_product_matrix)}")

    if plot_path:
        print(f"\nGenerating heatmap...")
        import matplotlib.pyplot as plt

        matrix = np.array(dot_product_matrix)

        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)

        ax.set_xticks(range(len(matrix)))
        ax.set_yticks(range(len(matrix)))
        ax.set_xticklabels([str(i) for i in range(len(matrix))], fontsize=8)
        ax.set_yticklabels([str(i) for i in range(len(matrix))], fontsize=8)

        ax.set_xlabel("Codebook Index")
        ax.set_ylabel("Codebook Index")
        title = "Dot Product Matrix of Codebook Vectors"
        if normalize:
            title += " (Cosine Similarity)"
        ax.set_title(title)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Dot Product" if not normalize else "Cosine Similarity")

        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"Heatmap saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Export dot product matrix of codebook vectors")
    parser.add_argument("--model", "-m", required=True, help="Path to trained model")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file path")
    parser.add_argument(
        "--normalize",
        "-n",
        action="store_true",
        help="Normalize vectors before computing dot products (cosine similarity)",
    )
    parser.add_argument(
        "--plot",
        "-p",
        help="Path to save heatmap image (PNG). If not provided, no plot is generated.",
    )

    args = parser.parse_args()

    export_dot_product(args.model, args.output, args.normalize, args.plot)


if __name__ == "__main__":
    main()
