#!/usr/bin/env python3
"""Convert encodings.json to LaTeX appendix with two-column multi-page table."""

import json
import sys


def escape_latex(word: str) -> str:
    specials = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde",
        "^": r"\textasciicircum",
    }
    result: list[str] = []
    for c in word:
        result.append(specials.get(c, c))
    return "".join(result)


def load_encodings(path, limit=None):
    with open(path, "r") as f:
        data = json.load(f)
    if limit:
        # keep only first 'limit' items after sorting
        items = sorted(data.items(), key=lambda x: x[0].lower())
        items = items[:limit]
        return dict(items)
    return data


def generate_latex(encodings):
    lines = [
        r"\documentclass[a4paper,10pt]{article}",
        r"\usepackage[margin=1cm]{geometry}",
        r"\usepackage{multicol}",
        r"\usepackage{supertabular}",
        r"\usepackage{booktabs}",
        r"",
        r"% Compact table settings",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0pt}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{1.2}",
        r"",
        r"\begin{document}",
        r"",
        r"\appendix",
        r"\section*{Appendix: Word Encodings (6‑Syllable IDs)}",
        r"",
        r"\begin{multicols}{2}",
        r"\tablehead{",
        r"\toprule",
        r"\textbf{Word} & \textbf{S1} & \textbf{S2} & \textbf{S3} & \textbf{S4} & \textbf{S5} & \textbf{S6} \\",
        r"\midrule",
        r"}",
        r"\tabletail{",
        r"\bottomrule",
        r"}",
        r"\begin{supertabular}{l*{6}{c}}",
    ]

    # Sort words alphabetically, case-insensitive
    sorted_words = sorted(encodings.keys(), key=lambda x: x.lower())

    for word in sorted_words:
        ids = encodings[word]
        if len(ids) > 6:
            print(f"Warning: {word} has {len(ids)} IDs, truncating to 6", file=sys.stderr)
            ids = ids[:6]
        # Ensure we have exactly 6 cells (pad with empty string if fewer)
        cells = [str(i) for i in ids]
        while len(cells) < 6:
            cells.append("")
        escaped_word = escape_latex(word)
        id_cells = " & ".join(cells)
        lines.append(f"{escaped_word} & {id_cells} \\\\")

    lines.extend([r"\end{supertabular}", r"\end{multicols}", r"", r"\end{document}", ""])
    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate LaTeX appendix from encodings.json")
    parser.add_argument("input", nargs="?", default="encodings.json", help="Input JSON file")
    parser.add_argument("-o", "--output", default="encodings_appendix.tex", help="Output LaTeX file")
    parser.add_argument("--limit", type=int, help="Limit to first N entries (after sorting)")
    args = parser.parse_args()

    encodings = load_encodings(args.input, limit=args.limit)
    print(f"Loaded {len(encodings)} entries", file=sys.stderr)

    latex_content = generate_latex(encodings)

    with open(args.output, "w") as f:
        f.write(latex_content)

    print(f"Generated {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
