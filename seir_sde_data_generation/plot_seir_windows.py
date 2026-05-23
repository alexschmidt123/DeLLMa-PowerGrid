"""
Generate per-file line plots for SEIR window CSV datasets.

Input root (default):
    DeLLMa-main/data/seir

Expected input subfolders:
    observed_window/
    future_window_no_vaccine/
    future_window_with_vaccine/

Output root:
    <input_root>/plots
with mirrored subfolders and one PNG per CSV (same base filename).
Also writes combined city plots under <input_root>/plots/combined/.
"""

import argparse
import os
from typing import List

import pandas as pd

PLOTS_DIR_NAME = "plots"
WINDOW_FOLDERS = (
    "observed_window",
    "future_window_no_vaccine",
    "future_window_with_vaccine",
)
COMBINED_DIR_NAME = "combined"
REQUIRED_WINDOW_COLUMNS = {
    "day",
    "infected_population",
    "recovered_population",
    "city_id",
    "data_role",
}


def _parse_args() -> argparse.Namespace:
    here = os.path.dirname(os.path.abspath(__file__))
    default_data_root = os.path.abspath(os.path.join(here, "../DeLLMa-main/data/seir"))
    parser = argparse.ArgumentParser(description="Plot all SEIR window CSV files.")
    parser.add_argument(
        "--data-root",
        type=str,
        default=default_data_root,
        help=f"SEIR data root containing three window subfolders (default: {default_data_root})",
    )
    return parser.parse_args()


def _list_csvs(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(
        [
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if name.lower().endswith(".csv")
        ]
    )


def _city_file_index(path: str) -> int:
    stem, ext = os.path.splitext(os.path.basename(path))
    if ext.lower() not in {".csv", ".png"}:
        return 10**9
    if not stem.startswith("city"):
        return 10**9
    try:
        return int(stem.removeprefix("city"))
    except ValueError:
        return 10**9


def _ensure_clean_plot_dir(plot_dir: str) -> None:
    os.makedirs(plot_dir, exist_ok=True)
    for name in os.listdir(plot_dir):
        if name.lower().endswith(".png"):
            os.remove(os.path.join(plot_dir, name))


def _plot_one(csv_path: str, png_path: str, window_name: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    missing = set(REQUIRED_WINDOW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")

    df = df.sort_values("day")
    city_id = str(df["city_id"].iloc[0]) if len(df) > 0 else "unknown_city"
    data_role = str(df["data_role"].iloc[0]) if len(df) > 0 else "unknown_role"

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["day"], df["infected_population"], label="Infected", color="#d62728", linewidth=2.0)
    ax.plot(df["day"], df["recovered_population"], label="Recovered", color="#2ca02c", linewidth=2.0)
    ax.set_xlabel("Day")
    ax.set_ylabel("Population")
    ax.set_title(f"{city_id} | {window_name} | {data_role}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=140)
    plt.close(fig)


def _load_window_file(data_root: str, folder: str, city_id: str) -> pd.DataFrame:
    path = os.path.join(data_root, folder, f"{city_id}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing required CSV for combined plot: {path}")
    df = pd.read_csv(path)
    missing = set(REQUIRED_WINDOW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    return df.sort_values("day")


def _plot_combined_city(data_root: str, plot_dir: str, city_id: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    observed = _load_window_file(data_root, "observed_window", city_id)
    no_vaccine = _load_window_file(data_root, "future_window_no_vaccine", city_id)
    with_vaccine = _load_window_file(data_root, "future_window_with_vaccine", city_id)

    no_vaccine_full = pd.concat([observed, no_vaccine], ignore_index=True).sort_values("day")
    city_label = str(observed["city_id"].iloc[0]) if len(observed) > 0 else city_id

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(
        no_vaccine_full["day"],
        no_vaccine_full["infected_population"],
        label="Infected, no vaccine",
        color="#d62728",
        linestyle="-",
        linewidth=2.2,
    )
    ax.plot(
        no_vaccine_full["day"],
        no_vaccine_full["recovered_population"],
        label="Recovered, no vaccine",
        color="#2ca02c",
        linestyle="-",
        linewidth=2.2,
    )
    ax.plot(
        with_vaccine["day"],
        with_vaccine["infected_population"],
        label="Infected, with vaccine",
        color="#d62728",
        linestyle=":",
        linewidth=2.8,
    )
    ax.plot(
        with_vaccine["day"],
        with_vaccine["recovered_population"],
        label="Recovered, with vaccine",
        color="#2ca02c",
        linestyle=":",
        linewidth=2.8,
    )
    ax.axvline(30, color="#555555", linestyle="--", linewidth=1.2, alpha=0.8, label="Vaccine decision day")
    ax.set_xlabel("Day")
    ax.set_ylabel("Population")
    ax.set_title(f"{city_id} | complete SEIR window with vaccine counterfactual")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"{city_label}.png"), dpi=140)
    plt.close(fig)


def _plot_combined_all(data_root: str, plot_root: str) -> int:
    combined_dir = os.path.join(plot_root, COMBINED_DIR_NAME)
    _ensure_clean_plot_dir(combined_dir)
    observed_dir = os.path.join(data_root, "observed_window")
    city_files = sorted(
        [f for f in os.listdir(observed_dir) if f.startswith("city") and f.endswith(".csv")],
        key=_city_file_index,
    )
    for city_file in city_files:
        city_id = os.path.splitext(city_file)[0]
        _plot_combined_city(data_root, combined_dir, city_id)
    return len(city_files)


def main() -> None:
    args = _parse_args()
    data_root = os.path.abspath(args.data_root)
    plot_root = os.path.join(data_root, PLOTS_DIR_NAME)

    total_csv = 0
    total_png = 0

    print(f"SEIR data root: {data_root}")
    print(f"Plot output root: {plot_root}")

    for window in WINDOW_FOLDERS:
        csv_dir = os.path.join(data_root, window)
        plot_dir = os.path.join(plot_root, window)
        _ensure_clean_plot_dir(plot_dir)
        csv_files = sorted(_list_csvs(csv_dir), key=_city_file_index)
        total_csv += len(csv_files)
        print(f"\n[{window}] found {len(csv_files)} CSV files")

        for csv_path in csv_files:
            city_id = os.path.splitext(os.path.basename(csv_path))[0]
            png_path = os.path.join(plot_dir, f"{city_id}.png")
            _plot_one(csv_path, png_path, window_name=window)
            total_png += 1

    combined_png = _plot_combined_all(data_root, plot_root)
    total_png += combined_png
    print(f"\n[{COMBINED_DIR_NAME}] generated {combined_png} combined PNG files")

    print("\n==============================================================")
    print(f"CSV files processed : {total_csv}")
    print(f"PNG files generated : {total_png}")
    print("==============================================================")


if __name__ == "__main__":
    main()
