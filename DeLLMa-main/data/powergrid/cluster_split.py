#!/usr/bin/env python3
"""
Cluster rows of Data_for_UCI_named.csv on (tau, p, g) with k-means,
then write:
  - Data_for_UCI_named_with_cluster.csv  (all rows + cluster_id)
  - cluster_00.csv .. cluster_{k-1}.csv    (one file per cluster)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = (
    [f"tau{i}" for i in range(1, 5)]
    + [f"p{i}" for i in range(1, 5)]
    + [f"g{i}" for i in range(1, 5)]
)


def main() -> None:
    parser = argparse.ArgumentParser(description="K-means split of powergrid UCI CSV into per-cluster files.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "Data_for_UCI_named.csv",
        help="Input CSV path",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for outputs",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=7,
        help="Number of clusters (default: 7, aligned with POWERGRID_CLUSTERS / 120 runs)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for k-means")
    parser.add_argument(
        "--n-init",
        type=int,
        default=10,
        help="Number of k-means runs with different centroid seeds",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[FEATURE_COLS].to_numpy(dtype=float)
    Xz = StandardScaler().fit_transform(X)

    km = KMeans(n_clusters=args.k, random_state=args.seed, n_init=args.n_init)
    labels = km.fit_predict(Xz)
    df_out = df.copy()
    df_out["cluster_id"] = labels

    combined_path = out_dir / "Data_for_UCI_named_with_cluster.csv"
    df_out.to_csv(combined_path, index=False)

    for cid in range(args.k):
        part = df_out[df_out["cluster_id"] == cid]
        part_path = out_dir / f"cluster_{cid:02d}.csv"
        part.to_csv(part_path, index=False)

    # Small summary for logs
    counts = df_out["cluster_id"].value_counts().sort_index()
    print(f"Wrote: {combined_path}")
    print(f"Wrote: {args.k} files cluster_00.csv .. cluster_{args.k - 1:02d}.csv in {out_dir}")
    print("Cluster sizes:")
    for cid, n in counts.items():
        print(f"  cluster_{int(cid):02d}: {int(n)}")
    print(f"Inertia (SSE): {km.inertia_:.6f}")


if __name__ == "__main__":
    main()
