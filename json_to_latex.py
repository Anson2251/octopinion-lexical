#!/usr/bin/env python3
"""Convert codebook_words.json to LaTeX appendix with dense tables."""

import json


def load_codebook(path):
    with open(path, "r") as f:
        return json.load(f)


def similarity_to_pct(similarity):
    return f"{similarity * 100:.1f}"


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


def format_cell(word: str, similarity: float) -> str:
    pct = similarity_to_pct(similarity)
    return f"\\textbf{{{word}}}\\hfill\\tiny({pct}\\%)"


def generate_latex(codebook):
    lines = [
        r"\documentclass[a4paper,10pt]{article}",
        r"\usepackage[margin=0.5cm,landscape]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{tabularx}",
        r"\usepackage{lscape}",
        r"\usepackage{adjustbox}",
        r"\usepackage{graphicx}",
        r"",
        r"% Compact table settings",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0pt}",
        r"\renewcommand{\arraystretch}{1.5}",
        r"\setlength{\tabcolsep}{2pt}",
        r"",
        r"\begin{document}",
        r"",
        r"\appendix",
        r"\section*{Appendix: Codebook Word Similarities}",
        r"",
        r"\begin{adjustbox}{width=\linewidth,center}",
        r"\footnotesize",
        r"\begin{tabularx}{\linewidth}{c|*{10}{|>{\raggedright\arraybackslash}X}}",
        r"\toprule",
        r"\hline",
        r"\textbf{}", # left blank to save space
    ]

    # Add simple column headers
    for i in range(1, 11):
        lines.append(r" & \textbf{\footnotesize Word " + str(i) + r"}")
    lines.append(r" \\")
    lines.append(r"\hline")

    for code_id in sorted(codebook.keys(), key=lambda x: int(x)):
        entries = codebook[code_id]

        cells = [code_id]
        for entry in entries:
            word = escape_latex(entry["word"])
            cells.append(format_cell(word, entry["similarity"]))

        line = " & ".join(cells) + r" \\"
        lines.append(line)

    lines.extend([r"\hline", r"\bottomrule", r"\end{tabularx}", r"\end{adjustbox}", r"", r"\end{document}", ""])

    return "\n".join(lines)


def main():
    input_file = "codebook_words.json"
    output_file = "codebook_appendix.tex"

    codebook = load_codebook(input_file)
    latex_content = generate_latex(codebook)

    with open(output_file, "w") as f:
        f.write(latex_content)

    print(f"Generated {output_file} with {len(codebook)} codebook entries")


if __name__ == "__main__":
    main()
