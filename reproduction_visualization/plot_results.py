"""
Plot: One figure with 4 horizontal bar charts — one per (task, metric).
Each plot has a title. Single legend 2×3 (no gap, same width). No border line, no task labels.
Run: python reproduction_visualization/plot_results.py
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUTDIR = os.path.dirname(os.path.abspath(__file__))

METHODS = [
    "DeLLMa-Pairs", "DeLLMa-Top1", "DeLLMa-Naive",
    "Zero-shot", "CoT", "Self-consistency",
]
DELLMA_NAMES = {"DeLLMa-Pairs", "DeLLMa-Top1", "DeLLMa-Naive"}

ACC = np.array([
    [70.0, 60.8], [66.7, 65.0], [55.8, 30.8],
    [30.8, 27.5], [34.2, 23.3], [30.0, 25.8],
])
OPT = np.array([
    [95.5, 65.5], [94.6, 66.9], [87.7, 51.1],
    [82.7, 53.4], [80.5, 49.6], [82.9, 51.8],
])

COLORS_DELLMA = ["#cb181d", "#ef3b2c", "#fb6a4a"]
COLORS_BASELINE = ["#006d2c", "#238b45", "#41ae76"]


def color_for_method(method: str, dellma_colors: list, baseline_colors: list):
    if method in DELLMA_NAMES:
        return dellma_colors[list(METHODS).index(method)]
    return baseline_colors[list(METHODS).index(method) - 3]


def label_bars(ax, vals, y_pos, fmt="{:.1f}"):
    """Add value labels on each bar in a horizontal bar chart."""
    for i, (y, v) in enumerate(zip(y_pos, vals)):
        ax.text(v + 1, y, fmt.format(v), va="center", ha="left", fontsize=8)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    xlabels = ["Accuracy (%)", "Optimality (%)"]
    plot_titles = [
        "Agriculture — Accuracy (%)",
        "Agriculture — Optimality (%)",
        "Stocks — Accuracy (%)",
        "Stocks — Optimality (%)",
    ]
    panels = [
        (axes[0, 0], ACC[:, 0], 0, plot_titles[0]),
        (axes[0, 1], OPT[:, 0], 1, plot_titles[1]),
        (axes[1, 0], ACC[:, 1], 0, plot_titles[2]),
        (axes[1, 1], OPT[:, 1], 1, plot_titles[3]),
    ]

    for ax, values, col, title in panels:
        pairs = list(zip(METHODS, values))
        pairs.sort(key=lambda p: p[1], reverse=True)
        labels = [p[0] for p in pairs]
        vals = [p[1] for p in pairs]
        colors = [color_for_method(m, COLORS_DELLMA, COLORS_BASELINE) for m in labels]
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, vals, color=colors)
        label_bars(ax, vals, y_pos)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel(xlabels[col])
        ax.set_title(title)
        ax.set_xlim(0, 105)
        ax.invert_yaxis()

    # Leave room for legends; top will be set after legends are placed to add gap
    plt.subplots_adjust(left=0.12, right=0.92, top=0.82, bottom=0.08, hspace=0.40, wspace=0.35)

    # Main title
    fig.suptitle("DeLLMa vs Baselines Performance (reproduced)", fontsize=14, y=0.98)

    # Two separate legends: same width (bbox 4-tuple), no gap between them, gap below second legend
    legend_width = 0.5
    legend_left = (1 - legend_width) / 2  # center: 0.25
    handles_dellma = [mpatches.Patch(color=c, label=m) for m, c in zip(["DeLLMa-Pairs", "DeLLMa-Top1", "DeLLMa-Naive"], COLORS_DELLMA)]
    handles_base = [mpatches.Patch(color=c, label=m) for m, c in zip(["Zero-shot", "CoT", "Self-consistency"], COLORS_BASELINE)]
    leg_top = fig.legend(
        handles_dellma,
        ["DeLLMa-Pairs", "DeLLMa-Top1", "DeLLMa-Naive"],
        title="DeLLMa variants",
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(legend_left, 0.92, legend_width, 0.01),
        frameon=True,
        fontsize=9,
    )
    fig.add_artist(leg_top)
    fig.canvas.draw()
    bbox_top = leg_top.get_window_extent().transformed(fig.transFigure.inverted())
    y_bottom_of_top = bbox_top.y0
    leg_bottom = fig.legend(
        handles_base,
        ["Zero-shot", "CoT", "Self-consistency"],
        title="Baselines",
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(legend_left, y_bottom_of_top, legend_width, 0.01),
        frameon=True,
        fontsize=9,
    )
    fig.canvas.draw()
    bbox_bottom = leg_bottom.get_window_extent().transformed(fig.transFigure.inverted())
    gap_below_legends = 0.025
    top_plots = bbox_bottom.y0 - gap_below_legends
    plt.subplots_adjust(top=top_plots)

    plt.savefig(os.path.join(OUTDIR, "dellma_vs_baselines.png"), dpi=150, bbox_inches="tight")
    print("Saved dellma_vs_baselines.png")
    plt.close()
    print("Done. Open dellma_vs_baselines.png")


if __name__ == "__main__":
    main()
