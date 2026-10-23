"""Figure 2: labels selected per item against the Jaccard index, from results/setvalued/test.json."""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from common import ROOT  # noqa: E402

TASKS = [("tweet_emotion", "TweetEmotion"), ("tweet_topic", "TweetTopic"), ("jecqa", "JEC-QA")]
TIERS = {"small": ("#eb6834", "o"), "inexpensive": ("#1baf7a", "s"), "frontier": ("#2a78d6", "^")}
MODELS = {"Qwen3.5-4B": "small", "Qwen3-8B": "small", "GLM-4-9B-0414": "small", "Qwen3.5-122B-A10B": "inexpensive",
          "DeepSeek-V4-Pro": "frontier", "Kimi-K2.6": "frontier"}
INK, MUTED = "#0b0b0b", "#52514e"
# labels for the points that the legend alone does not identify: (task, model) -> (text, dx, dy, alignment)
LABELS = {
    ("tweet_emotion", "Qwen3-8B"): ("Qwen3-8B", -0.07, 0.0, "right"),
    ("tweet_emotion", "DeepSeek-V4-Pro"): ("V4-Pro", 0.07, 0.0, "left"),
    ("tweet_emotion", "Kimi-K2.6"): ("Kimi", 0.0, -1.7, "center"),
    ("tweet_topic", "DeepSeek-V4-Pro"): ("V4-Pro", 0.07, 0.0, "left"),
    ("tweet_topic", "Kimi-K2.6"): ("Kimi", -0.07, 0.0, "right"),
    ("jecqa", "DeepSeek-V4-Pro"): ("V4-Pro", 0.07, 0.0, "left"),
    ("jecqa", "Kimi-K2.6"): ("Kimi", 0.07, 0.0, "left"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "fig_selection.pdf"))
    args = ap.parse_args()
    sel = json.loads((ROOT / "results" / "setvalued" / "test.json").read_text(encoding="utf-8"))["selection"]

    plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.edgecolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 3, figsize=(5.4, 2.2), sharex=True)
    for ax, (task, name) in zip(axes, TASKS):
        s = sel[task]
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.axvline(s["gold_labels_per_item"], color=MUTED, linestyle=(0, (3, 2)), linewidth=0.8, zorder=1)
        for m, tier in MODELS.items():
            color, marker = TIERS[tier]
            x, y = s[m]["labels_per_item"], s[m]["jaccard"]
            ax.scatter(x, y, s=26, color=color, marker=marker, edgecolors="white", linewidths=0.6, zorder=3)
            if (task, m) in LABELS:
                text, dx, dy, ha = LABELS[(task, m)]
                ax.annotate(text, (x + dx, y + dy), ha=ha, va="center", fontsize=6.5, color=INK)
        ax.scatter(s["fusion"]["labels_per_item"], s["fusion"]["jaccard"], s=75, color=INK, marker="*",
                   edgecolors="white", linewidths=0.6, zorder=4)
        ax.margins(y=0.12)
        ax.set_title(name, fontsize=8, loc="left")
        ax.set_xlim(0.9, 2.75)
        ax.set_xticks([1.0, 1.5, 2.0, 2.5])
    axes[0].set_ylabel(r"Jaccard index (\%)")
    axes[1].set_xlabel("labels selected per item")

    def mark(marker, color, size, text):
        return Line2D([], [], linestyle="", marker=marker, color=color, markersize=size, label=text)

    handles = [mark("o", TIERS["small"][0], 5, "small models"),
               mark("s", TIERS["inexpensive"][0], 5, "Qwen3.5-122B"),
               mark("^", TIERS["frontier"][0], 5, "frontier models"),
               mark("*", INK, 8, "fusion"),
               Line2D([], [], color=MUTED, linestyle=(0, (3, 2)), linewidth=0.8, label="gold sets (mean size)")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=7, handletextpad=0.3,
               columnspacing=1.0, handlelength=1.6, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.19, top=0.8, wspace=0.3)
    fig.savefig(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
