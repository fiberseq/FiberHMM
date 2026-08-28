#!/usr/bin/env python3
"""Plot the matched caller comparison from summarize_calls.py output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ORDER = (
    "PacBio existing",
    "PacBio current no-recaller",
    "ONT early supplied",
    "ONT early no-recaller",
    "ONT late supplied",
    "ONT late no-recaller",
)

DISPLAY = {
    "PacBio existing": "PacBio\nexisting",
    "PacBio current no-recaller": "PacBio\ncurrent,\nno recaller",
    "ONT early supplied": "ONT 1.5–3 h\nsupplied",
    "ONT early no-recaller": "ONT 1.5–3 h\nno recaller",
    "ONT late supplied": "ONT 3–4.5 h\nsupplied",
    "ONT late no-recaller": "ONT 3–4.5 h\nno recaller",
}

COLORS = {
    "PacBio existing": "#777777",
    "PacBio current no-recaller": "#AAAAAA",
    "ONT early supplied": "#5B8FF9",
    "ONT early no-recaller": "#A9C4FF",
    "ONT late supplied": "#E07B39",
    "ONT late no-recaller": "#F2B98E",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.summary.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    lookup = {(row["label"], row["group"]): row for row in rows}
    missing = [label for label in ORDER if (label, "overall") not in lookup]
    if missing:
        raise ValueError(f"missing labels in summary: {missing}")

    x = np.arange(len(ORDER))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True)
    panels = (
        ("nuc_fraction", "Nucleosome-tag coverage"),
        ("msp_fraction", "MSP coverage (length ≥60 bp)"),
    )
    for axis, (column, title) in zip(axes, panels):
        overall = [float(lookup[(label, "overall")][column]) for label in ORDER]
        axis.bar(
            x,
            overall,
            color=[COLORS[label] for label in ORDER],
            edgecolor="#333333",
            linewidth=0.6,
            width=0.72,
            zorder=2,
        )
        for index, label in enumerate(ORDER):
            for offset, group, marker in (
                (-0.08, "train", "o"),
                (0.08, "holdout", "s"),
            ):
                row = lookup.get((label, group))
                if row is None:
                    continue
                axis.scatter(
                    index + offset,
                    float(row[column]),
                    marker=marker,
                    s=26,
                    facecolor="white",
                    edgecolor="#222222",
                    linewidth=0.8,
                    zorder=3,
                )
        axis.set_title(title)
        axis.set_ylim(0, 1 if column == "nuc_fraction" else 0.30)
        axis.set_ylabel("Fraction of analyzed read bases")
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.7, zorder=0)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xticks(x, [DISPLAY[label] for label in ORDER], fontsize=8.5)

    axes[1].scatter(
        [], [], marker="o", facecolor="white", edgecolor="#222222", label="train"
    )
    axes[1].scatter(
        [], [], marker="s", facecolor="white", edgecolor="#222222", label="holdout"
    )
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle(
        "Matched Drosophila subsets: caller configuration explains excess ONT openness",
        fontsize=12,
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
