from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import os
import re
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# scipy is imported inside run_swing_simulation only (farmer/trader avoid loading it on import).

FRUITS = {
    "2021": [
        "apple",
        "avocado",
        "grape",
        "grapefruit",
        "lemon",
        "peach",
        "pear",
    ],
}

AGNOSTIC_STATES = [
    "climate condition",
    "supply chain disruptions",
    "economic health",
    "market sentiment and investor psychology",
    "political events and government policies",
    "natural disasters and other 'black swan' events",
    "geopolitical issues",
]

FRUIT_STATES = {
    "2021": {
        # product-agnostic state variables
        "agnostic": {
            "climate condition": "the climate condition of the next agricultural season in California",
            "supply chain disruptions": "the supply chain disruptions of the next agricultural season in California",
        },
        # product-specific state variables
        "specific": {
            # 'demand change': 'the demand change of the next agricultural season in California',
            "price change": lambda c: f"the change in price per unit of {c} for the next agricultural season in California",
            "yield change": lambda c: f"the change in yield of {c} for the next agricultural season in California",
        },
    },
}

STOCKS = ["AMD", "DIS", "GME", "GOOGL", "META", "NVDA", "SPY"]
STOCKS_SYMBOL_TO_NAME_MAP = {
    "AMD": "Advanced Micro Devices",
    "DIS": "The Walt Disney Company",
    "GME": "GameStop Corp",
    "GOOGL": "Alphabet, i.e. Google",
    "META": "Meta Platforms, i.e. Facebook",
    "NVDA": "NVIDIA",
    "SPY": "S&P 500",
}

# --- Power grid (IEEE-14 probing) domain: discrete action library v1 ---
GRID_ACTIONS = [
    "probe_bus_3_amp_0.05_dur_1.0",
    "probe_bus_3_amp_0.10_dur_1.5",
    "probe_bus_7_amp_0.05_dur_1.0",
    "probe_bus_7_amp_0.10_dur_2.0",
    "probe_bus_10_amp_0.05_dur_1.0",
    "probe_bus_10_amp_0.15_dur_1.5",
    "probe_bus_6_amp_0.08_dur_1.2",
    "probe_bus_8_amp_0.06_dur_1.0",
]

GRID_STATES = {
    "agnostic": {
        "inertia level": "aggregate inertia time constants (homogeneous M scale) for synchronous machines",
        "coupling level": "effective synchronizing strength / coupling (homogeneous K scale) in the reduced model",
        "response regime": "qualitative oscillatory / damping behavior after disturbances",
    }
}

# For rank-response parsing (same role as AGNOSTIC_STATES for evaluation)
GRID_AGNOSTIC_STATES = [
    "inertia level",
    "coupling level",
    "response regime",
]


def get_grid_actions() -> List[str]:
    return list(GRID_ACTIONS)


def get_grid_states() -> dict:
    """State variable descriptions for StateConfig.states (sequential mode)."""
    return GRID_STATES["agnostic"].copy()


def get_combinations(
    agent_name: str, source_year: Optional[str] = None
) -> List[Tuple[str, ...]]:
    combs = []
    if agent_name == "farmer":
        products = FRUITS[source_year]
    elif agent_name == "trader":
        products = STOCKS
    elif agent_name == "grid":
        # Single experiment: full discrete probe library as one choice tuple.
        return [tuple(GRID_ACTIONS)]
    else:
        raise ValueError("agent_name must be 'farmer', 'trader', or 'grid'")

    for i in range(2, len(products) + 1):
        for c in combinations(products, i):
            combs.append(c)

    return combs


def merge_by_commodity(
    df_x: pd.DataFrame | str,
    df_y: pd.DataFrame | str,
    on: str = "Commodity",
) -> pd.DataFrame:
    if type(df_x) == str:
        df_x = pd.read_csv(df_x)
    if type(df_y) == str:
        df_y = pd.read_csv(df_y)
    df = pd.merge(df_x, df_y, on=on)
    return df


# =============================================================================
# Grid domain (IEEE-14 swing): structured types — used by solver, metrics, prompts
# =============================================================================


@dataclass
class SimConfig:
    """Numerical integration settings for swing simulation."""

    t_end: float = 10.0
    dt_max: float = 0.02
    rtol: float = 1e-6
    atol: float = 1e-8


@dataclass
class ProbeActionSpec:
    """Discrete probing design (enumerable action id)."""

    action_id: str
    bus: int
    amplitude_pu: float
    duration_s: float
    waveform: str = "step"


@dataclass
class SwingSimResult:
    """Raw simulator output before prompt formatting."""

    success: bool
    action_id: str
    bus: int
    amplitude_pu: float
    duration_s: float
    theta_M: float
    theta_K: float
    t: List[float] = field(default_factory=list)
    freq_hz: List[float] = field(default_factory=list)
    rocof_hz_s: List[float] = field(default_factory=list)
    gen_angles_rad: List[List[float]] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbeFeatureSummary:
    """Compact features extracted for context and metrics."""

    max_abs_freq_dev_hz: float
    max_abs_rocof_hz_s: float
    final_freq_hz: float
    energy_like: float
    valid: bool
    stability_flag: str
    notes: str = ""


@dataclass
class PosteriorSummary:
    """Optional belief summary over discrete M–K / regime hypotheses (for prompts)."""

    inertia_level: str
    coupling_level: str
    response_regime: str
    entropy_bits: Optional[float] = None


@dataclass
class GridDecisionEval:
    """Code-only evaluation record (not LLM judgment)."""

    chosen_action_index: int
    chosen_action_id: str
    metric_value: float
    baseline_values: Dict[str, float] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Grid domain: IEEE-14 on-disk case (pandapower CSV export path)
# =============================================================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRID_CASE_DIR = os.path.join(
    PROJECT_ROOT, "data", "powergrid", "simulation", "pandapower", "ieee14"
)


def load_ieee14_csv_summary() -> Optional[Dict[str, Any]]:
    """
    Return compact numbers read from exported pandapower case14 CSVs.
    If the folder is missing, return None.
    """
    if not os.path.isdir(GRID_CASE_DIR):
        return None
    try:
        import pandas as pd_local
    except ImportError:
        return None

    out: Dict[str, Any] = {
        "source_dir": "data/powergrid/simulation/pandapower/ieee14",
    }
    loads_path = os.path.join(GRID_CASE_DIR, "loads.csv")
    buses_path = os.path.join(GRID_CASE_DIR, "buses.csv")
    lines_path = os.path.join(GRID_CASE_DIR, "lines.csv")
    gen_path = os.path.join(GRID_CASE_DIR, "generators.csv")
    ext_path = os.path.join(GRID_CASE_DIR, "ext_grid.csv")

    if os.path.isfile(loads_path):
        loads = pd_local.read_csv(loads_path)
        out["n_loads"] = int(len(loads))
        out["total_load_p_mw"] = float(loads["p_mw"].sum())
        out["total_load_q_mvar"] = float(loads["q_mvar"].sum())
    if os.path.isfile(buses_path):
        buses = pd_local.read_csv(buses_path)
        out["n_buses"] = int(len(buses))
    if os.path.isfile(lines_path):
        lines = pd_local.read_csv(lines_path)
        out["n_lines"] = int(len(lines))
    if os.path.isfile(gen_path):
        gen = pd_local.read_csv(gen_path)
        out["n_pv_generators"] = int(len(gen))
    if os.path.isfile(ext_path):
        ext = pd_local.read_csv(ext_path)
        out["n_slack_sources"] = int(len(ext))

    summary_txt = os.path.join(GRID_CASE_DIR, "powerflow_summary.txt")
    if os.path.isfile(summary_txt):
        out["powerflow_summary_file"] = "powerflow_summary.txt"

    return out if len(out) > 1 else None


def export_ieee14_case_csvs() -> None:
    """Write pandapower case14 + runpp results under data/powergrid/simulation/pandapower/ieee14/."""
    import pandapower as pp
    import pandapower.networks as nw

    os.makedirs(GRID_CASE_DIR, exist_ok=True)
    net = nw.case14()
    pp.runpp(net)

    net.bus.to_csv(os.path.join(GRID_CASE_DIR, "buses.csv"), index=False)
    net.line.to_csv(os.path.join(GRID_CASE_DIR, "lines.csv"), index=False)
    net.gen.to_csv(os.path.join(GRID_CASE_DIR, "generators.csv"), index=False)
    net.load.to_csv(os.path.join(GRID_CASE_DIR, "loads.csv"), index=False)
    net.ext_grid.to_csv(os.path.join(GRID_CASE_DIR, "ext_grid.csv"), index=False)
    if len(net.trafo) > 0:
        net.trafo.to_csv(os.path.join(GRID_CASE_DIR, "transformers.csv"), index=False)

    with open(os.path.join(GRID_CASE_DIR, "powerflow_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"converged={net.converged}\n")
        f.write(f"buses={len(net.bus)}, lines={len(net.line)}, trafo={len(net.trafo)}\n")
        f.write(f"gen={len(net.gen)}, ext_grid={len(net.ext_grid)}, load={len(net.load)}\n")

    print(f"Wrote {GRID_CASE_DIR} (buses={len(net.bus)})")


# =============================================================================
# Grid domain: swing-equation simulation (deterministic, IEEE-14–style reduced model)
# =============================================================================

F0_HZ = 60.0
OMEGA_S = 2 * math.pi * F0_HZ
SBASE_MVA = 100.0

N_FINITE = 4
H_NOM_S = np.array([4.0, 4.0, 3.5, 3.5], dtype=float)
D_NOM_PU = np.array([0.8, 0.8, 0.7, 0.7], dtype=float)

J0 = np.array(
    [
        [18.0, -5.0, -5.0, -5.0],
        [-5.0, 17.0, -4.5, -4.5],
        [-5.0, -4.5, 15.0, -4.0],
        [-5.0, -4.5, -4.0, 14.5],
    ],
    dtype=float,
)

BUS_PARTICIPATION: Dict[int, np.ndarray] = {
    1: np.array([0.05, 0.1, 0.15, 0.7]),
    2: np.array([0.35, 0.3, 0.2, 0.15]),
    3: np.array([0.4, 0.35, 0.15, 0.1]),
    4: np.array([0.2, 0.35, 0.25, 0.2]),
    5: np.array([0.15, 0.2, 0.35, 0.3]),
    6: np.array([0.1, 0.15, 0.35, 0.4]),
    7: np.array([0.1, 0.2, 0.25, 0.45]),
    8: np.array([0.1, 0.15, 0.2, 0.55]),
    9: np.array([0.1, 0.2, 0.3, 0.4]),
    10: np.array([0.15, 0.25, 0.3, 0.3]),
    11: np.array([0.1, 0.2, 0.35, 0.35]),
    12: np.array([0.1, 0.2, 0.3, 0.4]),
    13: np.array([0.1, 0.2, 0.3, 0.4]),
    14: np.array([0.1, 0.2, 0.3, 0.4]),
}


def _ieee14_case_metadata() -> Dict[str, Any]:
    """Optional pandapower case14 summary (no matrices in prompts)."""
    meta: Dict[str, Any] = {
        "name": "IEEE 14 (pandapower case14)",
        "n_bus": 14,
        "n_branch": 0,
        "converged": False,
    }
    try:
        import pandapower as pp  # type: ignore
        import pandapower.networks as nw  # type: ignore

        net = nw.case14()
        pp.runpp(net)
        meta["n_bus"] = len(net.bus)
        meta["n_branch"] = len(net.line) + len(net.trafo)
        meta["converged"] = bool(net.converged)
        meta["finite_generators"] = int(len(net.gen))
        meta["slack_bus_pp"] = int(net.ext_grid.bus.iloc[0])
    except Exception as exc:  # pragma: no cover - optional dependency
        meta["note"] = f"pandapower unavailable or failed: {exc}"
    return meta


def load_ieee14_case_config() -> Dict[str, Any]:
    """Structured case description for context formatting (compact, no Y-bus)."""
    meta = _ieee14_case_metadata()
    cfg: Dict[str, Any] = {
        "case_name": "IEEE-14",
        "base_mva": SBASE_MVA,
        "frequency_hz": F0_HZ,
        "finite_machines": N_FINITE,
        "nominal_inertia_s": H_NOM_S.tolist(),
        "pandapower": meta,
    }
    try:
        csv_sum = load_ieee14_csv_summary()
        if csv_sum:
            cfg["csv_case"] = csv_sum
    except Exception:
        pass
    return cfg


def parse_probe_action(action_id: str) -> ProbeActionSpec:
    """Parse canonical probe id: probe_bus_{n}_amp_{float}_dur_{float}."""
    m = re.match(
        r"^probe_bus_(\d+)_amp_([0-9.]+)_dur_([0-9.]+)$",
        action_id.strip(),
    )
    if not m:
        raise ValueError(f"Unrecognized probe action id: {action_id}")
    bus = int(m.group(1))
    amp = float(m.group(2))
    dur = float(m.group(3))
    return ProbeActionSpec(
        action_id=action_id, bus=bus, amplitude_pu=amp, duration_s=dur
    )


def _participation_for_bus(bus: int) -> np.ndarray:
    if bus not in BUS_PARTICIPATION:
        keys = sorted(BUS_PARTICIPATION.keys())
        nearest = min(keys, key=lambda k: abs(k - bus))
        return BUS_PARTICIPATION[nearest].copy()
    return BUS_PARTICIPATION[bus].copy()


def _swing_rhs(
    t: float,
    y: np.ndarray,
    H: np.ndarray,
    D: np.ndarray,
    J: np.ndarray,
    Pm: np.ndarray,
    Pe0: np.ndarray,
    pulse: np.ndarray,
    t_pulse_end: float,
) -> np.ndarray:
    """State y = [delta(rad), omega_pu]."""
    n = H.shape[0]
    delta = y[:n]
    omega = y[n:]
    Pe = Pe0 + J @ delta
    P_inj = pulse if t <= t_pulse_end else np.zeros_like(pulse)
    P_acc = Pm - Pe - D * omega + P_inj
    domega_dt = P_acc / (2.0 * H)
    ddelta_dt = omega * OMEGA_S
    return np.concatenate([ddelta_dt, domega_dt])


def run_swing_simulation(
    case_config: Dict[str, Any],
    theta_true: Dict[str, float],
    action: ProbeActionSpec,
    sim_config: SimConfig,
) -> SwingSimResult:
    """
    Run classical linearized swing with homogeneous M, K scaling.

    theta_true keys:
      - M: inertia scale (multiplies nominal H)
      - K: coupling/damping scale (multiplies J0 and D_nom)
    """
    messages: List[str] = []
    theta_M = float(theta_true.get("M", 1.0))
    theta_K = float(theta_true.get("K", 1.0))
    if theta_M <= 0 or theta_K <= 0:
        return SwingSimResult(
            success=False,
            action_id=action.action_id,
            bus=action.bus,
            amplitude_pu=action.amplitude_pu,
            duration_s=action.duration_s,
            theta_M=theta_M,
            theta_K=theta_K,
            messages=["Invalid theta: M and K must be positive."],
        )

    spec = action
    H = theta_M * H_NOM_S
    D = theta_K * D_NOM_PU
    J = theta_K * J0

    n = N_FINITE
    Pm = np.zeros(n)
    Pe0 = np.zeros(n)
    delta0 = np.zeros(n)
    pulse = _participation_for_bus(spec.bus) * spec.amplitude_pu
    t_span = (0.0, float(sim_config.t_end))
    t_eval = np.arange(0.0, sim_config.t_end, sim_config.dt_max)
    y0 = np.concatenate([delta0, np.zeros(n)])

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return _swing_rhs(
            t, y, H, D, J, Pm, Pe0, pulse, float(spec.duration_s)
        )

    try:
        from scipy.integrate import solve_ivp

        sol = solve_ivp(
            rhs,
            t_span,
            y0,
            t_eval=t_eval,
            method="RK45",
            rtol=sim_config.rtol,
            atol=sim_config.atol,
            max_step=sim_config.dt_max,
        )
        if not sol.success:
            return SwingSimResult(
                success=False,
                action_id=spec.action_id,
                bus=spec.bus,
                amplitude_pu=spec.amplitude_pu,
                duration_s=spec.duration_s,
                theta_M=theta_M,
                theta_K=theta_K,
                messages=[sol.message or "solve_ivp failed"],
            )
        t = sol.t
        y = sol.y
        delta = y[:n, :]
        omega = y[n:, :]
        omega_coi = np.sum((2 * H[:, None]) * omega, axis=0) / np.sum(2 * H)
        freq_hz = F0_HZ * (1.0 + omega_coi)
        rocof = np.gradient(freq_hz, t, edge_order=2)
    except Exception as exc:
        return SwingSimResult(
            success=False,
            action_id=spec.action_id,
            bus=action.bus,
            amplitude_pu=action.amplitude_pu,
            duration_s=action.duration_s,
            theta_M=theta_M,
            theta_K=theta_K,
            messages=[str(exc)],
        )

    messages.append("Integration completed.")
    return SwingSimResult(
        success=True,
        action_id=spec.action_id,
        bus=spec.bus,
        amplitude_pu=spec.amplitude_pu,
        duration_s=spec.duration_s,
        theta_M=theta_M,
        theta_K=theta_K,
        t=t.tolist(),
        freq_hz=freq_hz.tolist(),
        rocof_hz_s=rocof.tolist(),
        gen_angles_rad=[delta[i, :].tolist() for i in range(n)],
        messages=messages,
        extra={"case_config_keys": list(case_config.keys())},
    )


def extract_frequency_features(sim: SwingSimResult) -> Dict[str, float]:
    if not sim.success or not sim.freq_hz:
        return {}
    f = np.array(sim.freq_hz)
    return {
        "min_hz": float(np.min(f)),
        "max_hz": float(np.max(f)),
        "mean_hz": float(np.mean(f)),
        "final_hz": float(f[-1]),
    }


def extract_rocof_features(sim: SwingSimResult) -> Dict[str, float]:
    if not sim.success or not sim.rocof_hz_s:
        return {}
    r = np.array(sim.rocof_hz_s)
    return {
        "min_rocof_hz_s": float(np.min(r)),
        "max_rocof_hz_s": float(np.max(r)),
        "max_abs_rocof_hz_s": float(np.max(np.abs(r))),
    }


def summarize_probe_features(sim: SwingSimResult) -> ProbeFeatureSummary:
    ff = extract_frequency_features(sim)
    rf = extract_rocof_features(sim)
    if not sim.success or not ff:
        return ProbeFeatureSummary(
            max_abs_freq_dev_hz=float("nan"),
            max_abs_rocof_hz_s=float("nan"),
            final_freq_hz=float("nan"),
            energy_like=float("nan"),
            valid=False,
            stability_flag="invalid",
            notes="; ".join(sim.messages),
        )
    fdev = max(abs(ff["max_hz"] - F0_HZ), abs(ff["min_hz"] - F0_HZ))
    max_abs_rocof = rf.get("max_abs_rocof_hz_s", float("nan"))
    final_hz = ff["final_hz"]
    if sim.freq_hz and sim.t and len(sim.t) > 2:
        farr = np.array(sim.freq_hz)
        tarr = np.array(sim.t)
        roc = np.gradient(farr, tarr, edge_order=2)
        energy_like = float(np.mean(roc ** 2))
    else:
        energy_like = float("nan")

    if fdev < 0.05 and max_abs_rocof < 0.5:
        stab = "stable"
    elif fdev < 0.2:
        stab = "marginal"
    else:
        stab = "oscillatory"

    return ProbeFeatureSummary(
        max_abs_freq_dev_hz=float(fdev),
        max_abs_rocof_hz_s=float(max_abs_rocof),
        final_freq_hz=float(final_hz),
        energy_like=float(energy_like),
        valid=True,
        stability_flag=stab,
        notes="",
    )


def check_simulation_validity(sim: SwingSimResult) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    if not sim.success:
        issues.append("simulation marked unsuccessful")
    if sim.freq_hz:
        if not np.all(np.isfinite(np.array(sim.freq_hz))):
            issues.append("non-finite frequency samples")
    if sim.rocof_hz_s:
        if not np.all(np.isfinite(np.array(sim.rocof_hz_s))):
            issues.append("non-finite ROCOF samples")
    return (len(issues) == 0), issues


def score_probe_action(sim: SwingSimResult, features: ProbeFeatureSummary) -> float:
    """Higher is better: informativeness minus safety penalty (code-only proxy)."""
    if not features.valid:
        return float("-inf")
    info = min(features.max_abs_freq_dev_hz, 0.5)
    safety_pen = 0.1 * max(0.0, features.max_abs_rocof_hz_s - 1.0)
    return float(info - safety_pen)


def run_probe_id(
    action_id: str,
    theta_true: Optional[Dict[str, float]] = None,
    sim_config: Optional[SimConfig] = None,
) -> Tuple[SwingSimResult, ProbeFeatureSummary]:
    """Convenience: parse action, load case, run swing, summarize."""
    theta_true = theta_true or {"M": 1.0, "K": 1.0}
    sim_config = sim_config or SimConfig()
    case = load_ieee14_case_config()
    action = parse_probe_action(action_id)
    sim = run_swing_simulation(case, theta_true, action, sim_config)
    feat = summarize_probe_features(sim)
    return sim, feat


# =============================================================================
# Grid domain: code-based metrics (belief entropy, safety-aware probe score)
# =============================================================================


def discrete_entropy_bits(probs: List[float]) -> float:
    """Shannon entropy in bits for a discrete distribution."""
    p = np.array(probs, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p + 1e-15)))


def posterior_sharpening(entropy_before: float, entropy_after: float) -> float:
    return float(entropy_before - entropy_after)


def variance_reduction(var_before: float, var_after: float) -> float:
    return float(var_before - var_after)


def estimation_error_scalar(estimate: float, truth: float) -> float:
    return float(abs(estimate - truth))


def safety_aware_informativeness(
    features: ProbeFeatureSummary,
    rocof_cap: float = 2.0,
    freq_cap_hz: float = 0.5,
) -> float:
    """
    Higher is better: reward informative response, penalize excessive ROCOF / deviation.
    """
    if not features.valid or not math.isfinite(features.max_abs_freq_dev_hz):
        return float("-inf")
    info = min(features.max_abs_freq_dev_hz, freq_cap_hz) / freq_cap_hz
    roc_excess = max(0.0, features.max_abs_rocof_hz_s - rocof_cap)
    safety = 1.0 / (1.0 + roc_excess)
    return float(0.7 * info + 0.3 * safety)


def mk_posterior_from_belief_dict(
    belief: Dict[str, Dict[str, str]],
    belief2score: Dict[str, float],
) -> Tuple[PosteriorSummary, float]:
    """Map first three global factors to PosteriorSummary + entropy (grid cache)."""
    entropies = []
    modes = []
    for key in ("inertia level", "coupling level", "response regime"):
        if key not in belief:
            continue
        vals = list(belief[key].keys())
        scores = np.array([belief2score[v] for v in belief[key].values()], dtype=float)
        probs = scores / scores.sum()
        entropies.append(discrete_entropy_bits(probs.tolist()))
        idx = int(np.argmax(probs))
        modes.append((key, vals[idx]))
    entropy_bits = float(sum(entropies)) if entropies else 0.0

    def _mode(name: str, default: str) -> str:
        for k, v in modes:
            if k == name:
                return v
        return default

    summary = PosteriorSummary(
        inertia_level=_mode("inertia level", "medium"),
        coupling_level=_mode("coupling level", "medium"),
        response_regime=_mode("response regime", "stable"),
        entropy_bits=entropy_bits,
    )
    return summary, entropy_bits


if __name__ == "__main__":
    export_ieee14_case_csvs()
