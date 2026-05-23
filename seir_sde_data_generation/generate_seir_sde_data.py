"""Generate the SEIR-SDE benchmark dataset for DeLLMa and evaluation."""

import argparse
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
EXPECTED_CITY_COUNT = 10


def _euler_maruyama_step(
    S: float, E: float, I: float, R: float,
    beta: float, sigma: float, gamma: float, N: float,
    dt: float, rng: np.random.Generator,
) -> Tuple[float, float, float, float]:
    rate_si = beta * S * I / N
    rate_ei = sigma * E
    rate_ir = gamma * I
    v_se = np.sqrt(max(rate_si, 0.0))
    v_ei = np.sqrt(max(rate_ei, 0.0))
    v_ir = np.sqrt(max(rate_ir, 0.0))
    dW = rng.standard_normal(4) * np.sqrt(dt)
    f = np.array([-rate_si, rate_si - rate_ei, rate_ei - rate_ir, rate_ir], dtype=float)
    g = np.array([
        [-v_se, v_se, 0.0, 0.0],
        [0.0, -v_ei, v_ei, 0.0],
        [0.0, 0.0, -v_ir, v_ir],
        [0.0, 0.0, 0.0, 0.0],
    ], dtype=float)
    X = np.array([S, E, I, R], dtype=float) + f * dt + dW @ g
    S_n, E_n, I_n, R_n = [max(x, 0.0) for x in X.tolist()]
    total = S_n + E_n + I_n + R_n
    if total > 0.0:
        scale = N / total
        S_n, E_n, I_n, R_n = S_n * scale, E_n * scale, I_n * scale, R_n * scale
    return S_n, E_n, I_n, R_n


def simulate_seir_sde(
    beta: float, sigma: float, gamma: float, N: float,
    S0: float, E0: float, I0: float, R0: float,
    T: float, dt: float, seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_steps = int(round(T / dt))
    times = np.linspace(0.0, T, n_steps + 1)
    S_arr = np.empty(n_steps + 1)
    E_arr = np.empty(n_steps + 1)
    I_arr = np.empty(n_steps + 1)
    R_arr = np.empty(n_steps + 1)
    S, E, I, R = float(S0), float(E0), float(I0), float(R0)
    S_arr[0], E_arr[0], I_arr[0], R_arr[0] = S, E, I, R
    for k in range(n_steps):
        S, E, I, R = _euler_maruyama_step(S, E, I, R, beta, sigma, gamma, N, dt, rng)
        S_arr[k + 1], E_arr[k + 1], I_arr[k + 1], R_arr[k + 1] = S, E, I, R
    return times, S_arr, E_arr, I_arr, R_arr


def nearest_state_at_time(
    times: np.ndarray, S_arr: np.ndarray, E_arr: np.ndarray,
    I_arr: np.ndarray, R_arr: np.ndarray, target_time: float,
) -> Tuple[float, float, float, float]:
    idx = int(np.argmin(np.abs(times - float(target_time))))
    return float(S_arr[idx]), float(E_arr[idx]), float(I_arr[idx]), float(R_arr[idx])


def apply_vaccine_counterfactual(
    S: float, E: float, I: float, R: float,
    beta: float, sigma: float, gamma: float,
    vaccine_effectiveness: float, vaccine_quantity: float,
    vaccine_beta_reduction: float,
) -> Tuple[float, float, float, float, float, float, float, float]:
    protected = min(float(vaccine_effectiveness) * float(vaccine_quantity), float(S))
    S_v = float(S) - protected
    R_v = float(R) + protected
    beta_v = (1.0 - float(vaccine_beta_reduction)) * float(beta)
    return S_v, float(E), float(I), R_v, protected, beta_v, float(sigma), float(gamma)


def discrete_burden_between_days(
    times: np.ndarray, I_arr: np.ndarray, start_day: int, end_day: int,
) -> float:
    total = 0.0
    for day in range(int(start_day), int(end_day) + 1):
        idx = int(np.argmin(np.abs(times - float(day))))
        total += float(I_arr[idx])
    return total


def trajectory_statistics(I_arr: np.ndarray, N: float) -> dict:
    rate = I_arr / N
    return {
        "mean_infected": float(np.mean(I_arr)),
        "max_infected": float(np.max(I_arr)),
        "mean_infected_rate": float(np.mean(rate)),
        "max_infected_rate": float(np.max(rate)),
    }


def passes_filter(stats: dict, cfg: dict) -> bool:
    if not cfg.get("enable_filtering", True):
        return True
    if stats["mean_infected"] < cfg.get("filter_min_mean_infected", 0.0):
        return False
    if stats["max_infected"] < cfg.get("filter_min_max_infected", 0.0):
        return False
    if stats["max_infected_rate"] < cfg.get("filter_min_max_infected_rate", 0.0):
        return False
    return True


WINDOW_SPECS = {
    "observed": {
        "folder": "observed_window",
        "day_start": 1,
        "day_end": 30,
        "data_role": "dellma_input",
    },
    "future_no_vaccine": {
        "folder": "future_window_no_vaccine",
        "day_start": 31,
        "day_end": 60,
        "data_role": "evaluation_only",
    },
    "future_with_vaccine": {
        "folder": "future_window_with_vaccine",
        "day_start": 31,
        "day_end": 60,
        "data_role": "evaluation_only",
    },
}


class ProgressReporter:
    """Small console progress reporter."""

    def __init__(self, total: int, label: str = "samples", update_every: int = 100):
        self.total = total
        self.label = label
        self.update_every = update_every
        self.count = 0
        self.kept = 0
        self.attempts = 0
        self._t0 = time.time()
        self._last_print = -1

    def update(self, kept: bool = True) -> None:
        self.attempts += 1
        if kept:
            self.count += 1
            self.kept += 1
        if self.count % self.update_every == 0 and self.count != self._last_print:
            self._last_print = self.count
            elapsed = time.time() - self._t0
            rate = self.count / elapsed if elapsed > 0 else float("nan")
            eta = (self.total - self.count) / rate if rate > 0 else float("nan")
            pct = 100.0 * self.count / self.total
            filter_rate = 100.0 * self.kept / self.attempts if self.attempts > 0 else 100.0
            print(
                f"  [{self.count:>6d}/{self.total}] {pct:5.1f}%  "
                f"rate={rate:.1f}/s  ETA={eta:.0f}s  "
                f"filter_pass={filter_rate:.1f}%",
                flush=True,
            )

    def done(self) -> None:
        elapsed = time.time() - self._t0
        print(
            f"  Done: {self.count} {self.label} in {elapsed:.1f}s  "
            f"(attempts={self.attempts}, "
            f"filter_pass={100.0 * self.kept / max(self.attempts, 1):.1f}%)"
        )


def _resolve_path(base: str, value: str) -> str:
    return value if os.path.isabs(value) else os.path.abspath(os.path.join(base, value))


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.num_samples is not None:
        cfg["num_samples"] = int(args.num_samples)
    if args.output_num_samples is not None:
        cfg["output_num_samples"] = int(args.output_num_samples)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.full_output_dir is not None:
        cfg["full_output_dir"] = args.full_output_dir
    if args.no_filtering:
        cfg["enable_filtering"] = False
    return cfg


def _ensure_output_dir(output_dir: str) -> str:
    abs_path = os.path.abspath(output_dir)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def _validate_output_dir(full_output_dir: str) -> None:
    blocked = os.path.abspath(os.path.join(_HERE, "generated_full_data"))
    resolved = os.path.abspath(full_output_dir)
    if resolved == blocked:
        raise ValueError(
            "Output directory 'seir_sde_data_generation/generated_full_data' is blocked. "
            "Use '../DeLLMa-main/data/seir' (or another DeLLMa data path) instead."
        )


def _fixed_theta(cfg: Dict[str, Any]) -> Tuple[float, float, float]:
    fixed = cfg.get("fixed_theta", {})
    return float(fixed["beta"]), float(fixed["sigma"]), float(fixed["gamma"])


def _select_initial_state(cfg: Dict[str, Any], sample_id: int) -> Tuple[float, float, float, float]:
    N = float(cfg["N"])
    pool = cfg.get("city_initial_pool")
    if pool:
        state = pool[sample_id % len(pool)]
        S0 = float(state["S0"])
        E0 = float(state["E0"])
        I0 = float(state["I0"])
        R0 = float(state.get("R0", cfg.get("R0", 0.0)))
    else:
        S0 = float(cfg["S0"])
        E0 = float(cfg["E0"])
        I0 = float(cfg["I0"])
        R0 = float(cfg["R0"])

    total = S0 + E0 + I0 + R0
    if abs(total - N) > 1e-6:
        raise ValueError(
            f"Initial compartments must sum to N={N}, got S0+E0+I0+R0={total} for sample_id={sample_id}."
        )
    return S0, E0, I0, R0


def _city_id_for_sample(cfg: Dict[str, Any], sample_id: int) -> str:
    pool = cfg.get("city_initial_pool")
    if pool:
        state = pool[sample_id % len(pool)]
        city_id = state.get("city_id")
        if city_id:
            return str(city_id)
    return f"city_{sample_id + 1}"


def _compute_action_utility(
    cfg: Dict[str, Any],
    beta: float,
    sigma: float,
    gamma: float,
    no_vax_times: np.ndarray,
    no_vax_S: np.ndarray,
    no_vax_E: np.ndarray,
    no_vax_I: np.ndarray,
    no_vax_R: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    vaccine_day = int(cfg.get("vaccine_day", 30))
    eval_day_start = int(cfg.get("evaluation_day_start", 31))
    eval_day_end = int(cfg.get("evaluation_day_end", 60))
    ve = float(cfg.get("vaccine_effectiveness", 1.0))
    q = float(cfg.get("vaccine_quantity", 0.0))
    beta_reduction = float(cfg.get("vaccine_beta_reduction", 0.6))

    S_30, E_30, I_30, R_30 = nearest_state_at_time(
        no_vax_times, no_vax_S, no_vax_E, no_vax_I, no_vax_R, vaccine_day
    )
    S_v0, E_v0, I_v0, R_v0, protected, beta_v, sigma_v, gamma_v = apply_vaccine_counterfactual(
        S_30, E_30, I_30, R_30,
        beta=beta, sigma=sigma, gamma=gamma,
        vaccine_effectiveness=ve, vaccine_quantity=q,
        vaccine_beta_reduction=beta_reduction,
    )

    vax_seed = int(rng.integers(0, 2**31))
    vax_times_delta, _, _, vax_I, vax_R = simulate_seir_sde(
        beta=beta_v,
        sigma=sigma_v,
        gamma=gamma_v,
        N=float(cfg["N"]),
        S0=S_v0,
        E0=E_v0,
        I0=I_v0,
        R0=R_v0,
        T=float(cfg["T"]) - float(vaccine_day),
        dt=float(cfg["dt"]),
        seed=vax_seed,
    )
    vax_times = vax_times_delta + float(vaccine_day)

    burden_no_vax = discrete_burden_between_days(
        no_vax_times, no_vax_I, start_day=eval_day_start, end_day=eval_day_end
    )
    burden_vax = discrete_burden_between_days(
        vax_times, vax_I, start_day=eval_day_start, end_day=eval_day_end
    )

    utility = {
        "vaccine_day": float(vaccine_day),
        "evaluation_day_start": float(eval_day_start),
        "evaluation_day_end": float(eval_day_end),
        "vaccine_protected_population": float(protected),
        "beta_no_vaccine": float(beta),
        "sigma_no_vaccine": float(sigma),
        "gamma_no_vaccine": float(gamma),
        "beta_with_vaccine": float(beta_v),
        "sigma_with_vaccine": float(sigma_v),
        "gamma_with_vaccine": float(gamma_v),
        "future_burden_no_vaccine": float(burden_no_vax),
        "future_burden_with_vaccine": float(burden_vax),
        "action_utility": float(burden_no_vax - burden_vax),
    }
    return utility, vax_times, vax_I, vax_R


def generate_pool(
    n_target: int,
    cfg: Dict[str, Any],
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    N, T, dt = cfg["N"], cfg["T"], cfg["dt"]
    beta, sigma, gamma = _fixed_theta(cfg)
    rows_meta: List[Dict[str, Any]] = []
    rows_traj: List[Dict[str, Any]] = []
    rows_vax_traj: List[Dict[str, Any]] = []
    sample_id = 0
    progress = ProgressReporter(n_target, label="pool samples", update_every=500)

    while sample_id < n_target:
        city_id = _city_id_for_sample(cfg, sample_id)
        S0, E0, I0, R0 = _select_initial_state(cfg, sample_id)
        sim_seed = int(rng.integers(0, 2**31))
        times, S_arr, E_arr, I_arr, R_arr = simulate_seir_sde(
            beta=beta, sigma=sigma, gamma=gamma,
            N=N, S0=S0, E0=E0, I0=I0, R0=R0,
            T=T, dt=dt, seed=sim_seed,
        )

        stats = trajectory_statistics(I_arr, N)
        utility, vax_times, vax_I, vax_R = _compute_action_utility(
            cfg=cfg,
            beta=beta,
            sigma=sigma,
            gamma=gamma,
            no_vax_times=times,
            no_vax_S=S_arr,
            no_vax_E=E_arr,
            no_vax_I=I_arr,
            no_vax_R=R_arr,
            rng=rng,
        )
        if not passes_filter(stats, cfg):
            progress.update(kept=False)
            continue

        rows_meta.append({
            "sample_id": sample_id,
            "city_id": city_id,
            "beta": beta,
            "sigma": sigma,
            "gamma": gamma,
            **utility,
        })
        rows_traj.extend(_trajectory_rows(sample_id, city_id, times, I_arr, R_arr))
        for k in range(len(vax_times)):
            rows_vax_traj.append({
                "sample_id": sample_id,
                "city_id": city_id,
                "time": round(float(vax_times[k]), 6),
                "I": float(vax_I[k]),
                "R": float(vax_R[k]),
            })
        sample_id += 1
        progress.update(kept=True)

    progress.done()
    return pd.DataFrame(rows_meta), pd.DataFrame(rows_traj), pd.DataFrame(rows_vax_traj)


def _trajectory_rows(
    sample_id: int,
    city_id: str,
    times: np.ndarray,
    I_arr: np.ndarray,
    R_arr: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for k in range(len(times)):
        rows.append({
            "sample_id": sample_id,
            "city_id": city_id,
            "time": round(float(times[k]), 6),
            "I": float(I_arr[k]),
            "R": float(R_arr[k]),
        })
    return rows


def _select_first_filtered_samples(
    meta_df: pd.DataFrame,
    traj_df: pd.DataFrame,
    output_num_samples: int | None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if output_num_samples is None:
        return meta_df, traj_df

    n_keep = int(output_num_samples)
    if n_keep <= 0:
        raise ValueError(f"output_num_samples must be positive, got {n_keep}.")
    if n_keep > len(meta_df):
        raise ValueError(
            f"output_num_samples={n_keep} exceeds available filtered pool size={len(meta_df)}."
        )

    kept_ids = meta_df["sample_id"].head(n_keep).to_numpy()
    kept_meta = meta_df.head(n_keep).copy()
    kept_traj = traj_df[traj_df["sample_id"].isin(kept_ids)].copy()
    return kept_meta, kept_traj


def _extract_daily_ir(
    traj_df: pd.DataFrame,
    sample_id: int,
    day_start: int,
    day_end: int,
) -> pd.DataFrame:
    sub = traj_df[traj_df["sample_id"] == sample_id].copy()
    rows: List[Dict[str, Any]] = []
    for day in range(day_start, day_end + 1):
        idx = (sub["time"] - float(day)).abs().idxmin()
        row = sub.loc[idx]
        rows.append({
            "day": int(day),
            "infected_population": float(row["I"]),
            "recovered_population": float(row["R"]),
        })
    return pd.DataFrame(rows)


def _clear_existing_csvs(output_dir: str) -> None:
    for name in os.listdir(output_dir):
        if name.lower().endswith(".csv"):
            os.remove(os.path.join(output_dir, name))


def _ensure_clean_subfolder(base_dir: str, folder_name: str) -> str:
    path = os.path.join(base_dir, folder_name)
    os.makedirs(path, exist_ok=True)
    for name in os.listdir(path):
        if name.lower().endswith(".csv"):
            os.remove(os.path.join(path, name))
    return path


def _write_30_city_csvs(
    output_dir: str,
    meta_df: pd.DataFrame,
    no_vax_traj_df: pd.DataFrame,
    vax_traj_df: pd.DataFrame,
) -> None:
    if len(meta_df) != EXPECTED_CITY_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_CITY_COUNT} cities for export, got {len(meta_df)} rows."
        )

    _clear_existing_csvs(output_dir)
    output_dirs = {
        key: _ensure_clean_subfolder(output_dir, spec["folder"])
        for key, spec in WINDOW_SPECS.items()
    }

    for _, m in meta_df.sort_values("sample_id").iterrows():
        sample_id = int(m["sample_id"])
        city_id = str(m["city_id"])

        _write_window_file(
            source_df=no_vax_traj_df,
            sample_id=sample_id,
            city_id=city_id,
            spec=WINDOW_SPECS["observed"],
            output_path=os.path.join(output_dirs["observed"], f"{city_id}.csv"),
        )
        _write_window_file(
            source_df=no_vax_traj_df,
            sample_id=sample_id,
            city_id=city_id,
            spec=WINDOW_SPECS["future_no_vaccine"],
            output_path=os.path.join(output_dirs["future_no_vaccine"], f"{city_id}.csv"),
        )
        _write_window_file(
            source_df=vax_traj_df,
            sample_id=sample_id,
            city_id=city_id,
            spec=WINDOW_SPECS["future_with_vaccine"],
            output_path=os.path.join(output_dirs["future_with_vaccine"], f"{city_id}.csv"),
        )


def _write_window_file(
    source_df: pd.DataFrame,
    sample_id: int,
    city_id: str,
    spec: Dict[str, Any],
    output_path: str,
) -> None:
    window_df = _extract_daily_ir(source_df, sample_id, spec["day_start"], spec["day_end"])
    window_df["city_id"] = city_id
    window_df["data_role"] = spec["data_role"]
    window_df.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate SEIR-SDE trajectories and metadata CSVs.")
    p.add_argument("--config", type=str, default=os.path.join(_HERE, "config.yaml"))
    p.add_argument("--num-samples", type=int, default=None)
    p.add_argument("--output-num-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--full-output-dir", type=str, default=None)
    p.add_argument("--no-filtering", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading config: {args.config}")
    cfg = _load_config(args.config)
    cfg = _apply_cli_overrides(cfg, args)

    seed = int(cfg["seed"])
    n_samples = int(cfg["num_samples"])
    output_num_samples = cfg.get("output_num_samples")
    output_num_samples = int(output_num_samples) if output_num_samples is not None else None
    if output_num_samples is not None and output_num_samples > n_samples:
        raise ValueError(
            f"output_num_samples={output_num_samples} cannot exceed num_samples={n_samples}."
        )
    full_output_dir = _resolve_path(_HERE, cfg["full_output_dir"])
    _validate_output_dir(full_output_dir)
    full_output_dir = _ensure_output_dir(full_output_dir)

    print(f"Fixed theta: {cfg['fixed_theta']}")

    print(f"\nGenerating {n_samples:,} trajectories for pooled dataset ...")
    pool_meta, pool_traj, pool_vax_traj = generate_pool(
        n_target=n_samples,
        cfg=cfg,
        rng=np.random.default_rng(seed),
    )
    export_meta, export_traj = _select_first_filtered_samples(
        meta_df=pool_meta,
        traj_df=pool_traj,
        output_num_samples=output_num_samples,
    )
    export_vax_traj = pool_vax_traj[pool_vax_traj["sample_id"].isin(export_meta["sample_id"])].copy()

    print("\nWriting 30 city CSVs for DeLLMa/evaluation ...")
    _write_30_city_csvs(
        output_dir=full_output_dir,
        meta_df=export_meta,
        no_vax_traj_df=export_traj,
        vax_traj_df=export_vax_traj,
    )

    print("\n==============================================================")
    print("SEIR generation complete")
    print(f"Full data directory      : {full_output_dir}")
    print(f"Filtered pool size       : {len(pool_meta):,}")
    print(f"Exported city count      : {len(export_meta):,}")
    print("Files written            : 30 CSV files")
    print("==============================================================")


if __name__ == "__main__":
    main()
