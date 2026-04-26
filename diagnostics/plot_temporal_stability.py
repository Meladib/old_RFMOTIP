"""
Companion plotting script for the RF-MOTIP temporal stability diagnostic.

Reads the JSON produced by diag_temporal_stability_script.py and generates
two PNG files:
  - temporal_stability_histogram.png  — overlaid histogram for all pairs
  - temporal_stability_per_pair.png   — line chart of mean ± std per pair

Usage:
    python diagnostics/plot_temporal_stability.py \
        --results_json diagnostics/temporal_stability_results.json \
        --output_dir diagnostics/
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # no GUI required
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Plot RF-MOTIP temporal stability results")
    parser.add_argument("--results_json", required=True,
                        help="Path to temporal_stability_results.json")
    parser.add_argument("--output_dir", default="diagnostics/",
                        help="Directory to write output PNGs")
    return parser.parse_args()


def load_results(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


def plot_histogram_overlay(pairs, output_path, layer_index):
    """
    Overlaid histogram: one translucent bar-chart per frame pair,
    x-axis = cosine similarity bins [-1, 1].
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    cmap = plt.get_cmap("tab10")
    bin_centers = None

    for i, pair in enumerate(pairs):
        hist = pair["histogram"]
        bins = hist["bins"]
        counts = np.array(hist["counts"], dtype=float)
        centers = np.array([(b[0] + b[1]) / 2 for b in bins])
        widths = np.array([b[1] - b[0] for b in bins])

        if bin_centers is None:
            bin_centers = centers

        # Normalize to fraction
        total = counts.sum()
        if total > 0:
            counts = counts / total

        ax.bar(
            centers, counts, width=widths * 0.8,
            alpha=0.45,
            color=cmap(i % 10),
            label=f"Pair {pair['pair_id']}  "
                  f"(T{pair['frame_t']:04d}→T{pair['frame_t1']:04d})",
            edgecolor="none",
        )

    # Threshold lines
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1.2, label="Unstable threshold (0.5)")
    ax.axvline(0.9, color="green", linestyle="--", linewidth=1.2, label="Stable threshold (0.9)")

    ax.set_xlabel("Cosine Similarity (T → T+1)", fontsize=11)
    ax.set_ylabel("Fraction of Queries", fontsize=11)
    layer_str = f"Layer {layer_index}" if layer_index >= 0 else "Final Layer"
    ax.set_title(f"RF-DETR Query Temporal Stability — {layer_str}", fontsize=12)
    ax.set_xlim(-1.05, 1.05)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Histogram saved to {output_path}")


def plot_per_pair_line(pairs, summary, output_path, layer_index):
    """
    Line chart: mean ± std cosine similarity per frame pair.
    Horizontal bands show threshold zones.
    """
    pair_ids = [p["pair_id"] for p in pairs]
    means = [p["mean_cos_sim"] for p in pairs]
    stds = [p["std_cos_sim"] for p in pairs]
    mins_ = [p["min_cos_sim"] for p in pairs]
    maxs_ = [p["max_cos_sim"] for p in pairs]

    x = np.array(pair_ids)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Stability zone backgrounds
    ax.axhspan(0.95, 1.0, alpha=0.08, color="green", label="Stable zone (>0.95)")
    ax.axhspan(0.85, 0.95, alpha=0.06, color="yellowgreen")
    ax.axhspan(0.70, 0.85, alpha=0.06, color="yellow")
    ax.axhspan(0.50, 0.70, alpha=0.06, color="orange", label="Severe zone (<0.70)")
    ax.axhspan(-1.0, 0.50, alpha=0.08, color="red", label="Critical zone (<0.50)")

    # Range band (min to max)
    ax.fill_between(x, mins_, maxs_, alpha=0.15, color="steelblue", label="Min–Max range")

    # Std band
    means_arr = np.array(means)
    stds_arr = np.array(stds)
    ax.fill_between(x, means_arr - stds_arr, means_arr + stds_arr,
                    alpha=0.35, color="steelblue", label="Mean ± Std")

    # Mean line
    ax.plot(x, means_arr, "o-", color="steelblue", linewidth=2.0, markersize=5, label="Mean")

    # Summary mean horizontal line
    ax.axhline(summary["mean_cos_sim"], color="navy", linestyle=":",
               linewidth=1.5, label=f"Overall mean ({summary['mean_cos_sim']:.3f})")

    # Threshold lines
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axhline(0.9, color="green", linestyle="--", linewidth=1.0, alpha=0.8)

    ax.set_xlabel("Frame Pair Index", fontsize=11)
    ax.set_ylabel("Cosine Similarity (T → T+1)", fontsize=11)
    layer_str = f"Layer {layer_index}" if layer_index >= 0 else "Final Layer"
    ax.set_title(f"RF-DETR Query Temporal Stability per Pair — {layer_str}", fontsize=12)
    ax.set_xticks(x)
    pair_labels = [f"{p['frame_t']:04d}\n→{p['frame_t1']:04d}" for p in pairs]
    ax.set_xticklabels(pair_labels, fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Per-pair chart saved to {output_path}")


def main():
    args = parse_args()
    results = load_results(args.results_json)

    pairs = results["pairs"]
    summary = results["summary"]
    layer_index = results["layer_index"]

    if not pairs:
        print("No pairs found in results JSON.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    hist_path = os.path.join(args.output_dir, "temporal_stability_histogram.png")
    per_pair_path = os.path.join(args.output_dir, "temporal_stability_per_pair.png")

    plot_histogram_overlay(pairs, hist_path, layer_index)
    plot_per_pair_line(pairs, summary, per_pair_path, layer_index)

    print(f"\nSummary: mean={summary['mean_cos_sim']:.4f}, "
          f"std={summary['std_cos_sim']:.4f}, "
          f"n_unstable={summary['n_unstable_total']}, "
          f"n_stable={summary['n_stable_total']}")


if __name__ == "__main__":
    main()
