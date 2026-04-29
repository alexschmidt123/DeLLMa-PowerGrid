#!/usr/bin/env python3
"""
Summarize `stab` distribution for the full UCI table and for each k-means cluster.

Reads `Data_for_UCI_named_with_cluster.csv` (produced by `cluster_split.py`).
Writes:
  - stab_summary.csv
  - stab_hist_overall.png
  - stab_hist_by_cluster.png
  - mean_stab_by_cluster.png  (per-regime mean stab bar chart)

Run from repo:  python data/powergrid/stab_distribution.py
Or from here:  python stab_distribution.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "Data_for_UCI_named_with_cluster.csv",
        help="CSV with cluster_id column",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Where to write summary CSV and PNGs",
    )
    args = parser.parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.input.is_file():
        raise FileNotFoundError(
            f"Missing {args.input}. Run cluster_split.py first to create it."
        )

    df = pd.read_csv(args.input)
    if "stab" not in df.columns:
        raise ValueError("Expected column 'stab'")
    if "cluster_id" not in df.columns:
        raise ValueError("Expected column 'cluster_id' — run cluster_split.py")

    df["stable"] = df["stabf"].astype(str).str.lower().eq("stable")

    def stats_block(name: str, s: pd.Series, stable: pd.Series) -> dict:
        x = s.astype(float)
        return {
            "subset": name,
            "n": len(x),
            "mean_stab": x.mean(),
            "std_stab": x.std(ddof=1) if len(x) > 1 else np.nan,
            "min_stab": x.min(),
            "q25_stab": x.quantile(0.25),
            "median_stab": x.median(),
            "q75_stab": x.quantile(0.75),
            "max_stab": x.max(),
            "frac_stable": stable.mean(),
        }

    rows = [stats_block("all_data", df["stab"], df["stable"])]
    for cid in sorted(df["cluster_id"].unique()):
        sub = df[df["cluster_id"] == cid]
        rows.append(
            stats_block(
                f"cluster_{int(cid):02d}",
                sub["stab"],
                sub["stable"],
            )
        )

    summary = pd.DataFrame(rows)
    csv_path = out_dir / "stab_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(summary.to_string(index=False))
    print(f"\nWrote: {csv_path}")

    # Per-regime mean stab (bar chart) — k-means actions / oracle context
    sub = summary[summary["subset"].str.startswith("cluster_")].copy()
    if not sub.empty:
        labels = sub["subset"].tolist()
        means = sub["mean_stab"].astype(float).values
        x = np.arange(len(labels))
        figb, axb = plt.subplots(figsize=(max(6, 0.9 * len(labels)), 4))
        axb.bar(x, means, color="steelblue", edgecolor="white", linewidth=0.8)
        axb.axhline(0.0, color="black", linewidth=0.8, linestyle="-")
        axb.set_xticks(x)
        axb.set_xticklabels(labels, rotation=25, ha="right")
        axb.set_ylabel("mean stab")
        axb.set_xlabel("regime (k-means cluster)")
        axb.set_title("Per-regime mean stability margin (stab)")
        lo, hi = float(np.nanmin(means)), float(np.nanmax(means))
        pad = 0.02 * max(hi - lo, 1e-6)
        axb.set_ylim(lo - pad, hi + pad)
        figb.tight_layout()
        pb = out_dir / "mean_stab_by_cluster.png"
        figb.savefig(pb, dpi=150, bbox_inches="tight")
        plt.close(figb)
        print(f"Wrote: {pb}")

    # --- Figures ---
    stab_all = df["stab"].astype(float)

    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.hist(stab_all, bins=40, color="steelblue", edgecolor="white", alpha=0.9)
    ax1.axvline(stab_all.mean(), color="darkred", linestyle="--", label=f"mean={stab_all.mean():.4f}")
    ax1.set_xlabel("stab (stability margin)")
    ax1.set_ylabel("count")
    ax1.set_title("stab distribution — entire dataset")
    ax1.legend()
    fig1.tight_layout()
    p1 = out_dir / "stab_hist_overall.png"
    fig1.savefig(p1, dpi=150)
    plt.close(fig1)
    print(f"Wrote: {p1}")

    n_clust = df["cluster_id"].nunique()
    ncols = min(4, n_clust)
    nrows = int(np.ceil(n_clust / ncols))
    fig2, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows))
    axes = np.atleast_2d(axes)
    for i, cid in enumerate(sorted(df["cluster_id"].unique())):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        sub = df[df["cluster_id"] == cid]["stab"].astype(float)
        ax.hist(sub, bins=25, color="seagreen", edgecolor="white", alpha=0.85)
        ax.axvline(sub.mean(), color="darkred", linestyle="--", linewidth=1)
        ax.set_title(f"cluster_{int(cid):02d} (n={len(sub)})")
        ax.set_xlabel("stab")
        ax.set_ylabel("count")
    # hide empty axes
    for j in range(i + 1, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].set_visible(False)
    fig2.suptitle("stab distribution by cluster", y=1.02)
    fig2.tight_layout()
    p2 = out_dir / "stab_hist_by_cluster.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Wrote: {p2}")


if __name__ == "__main__":
    main()
