"""
utils/seir_sde.py

Consolidated SEIR-SDE simulation and information-gain utilities for DeLLMa-main.

This module is the computational backbone of the SEIRAgent:
  - Euler-Maruyama SEIR-SDE simulator (closed-population, transition-row diffusion)
  - Prior sampling over theta = (beta, sigma, gamma)
  - IG proxy: logdet(Cov(y_a) + lambda*I) for any observation schedule a

DeLLMa mapping:
  actions         = observation schedules  a = (t1, ..., tK)
  uncertain state = theta = (beta, sigma, gamma) ~ uniform prior
  context         = model config + simulated trajectory summaries
  utility         = IG_proxy(a) = logdet(Cov_{theta}(y_a) + lambda*I)
  decision        = argmax_a IG_proxy(a)
"""

import numpy as np

# ---------------------------------------------------------------------------
# Default model configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL_CONFIG = dict(
    N=10_000,
    S0=9_900,
    E0=50,
    I0=50,
    R0=0,
    T=60,
    dt=1.0,
)

DEFAULT_PRIOR_BOUNDS = dict(
    beta=(0.1, 0.8),
    sigma=(0.1, 0.4),
    gamma=(0.05, 0.3),
)


# ---------------------------------------------------------------------------
# SEIR-SDE simulator (Euler-Maruyama)
# ---------------------------------------------------------------------------

def _drift(S, E, I, R, beta, sigma, gamma, N):
    inf  = beta * S * I / N
    prog = sigma * E
    rec  = gamma * I
    return -inf, inf - prog, prog - rec, rec


def _diffusion_noise(S, E, I, R, beta, sigma, gamma, N, dW):
    """
    Stochastic increment dW @ g using the transition-row convention.
    Each dW_j drives one flow (S->E, E->I, I->R), entering as
    opposite signs into the connected compartments.
    """
    v_SE = np.sqrt(max(beta * S * I / N, 0.0))
    v_EI = np.sqrt(max(sigma * E, 0.0))
    v_IR = np.sqrt(max(gamma * I, 0.0))
    return (
        -v_SE * dW[0],
         v_SE * dW[0] - v_EI * dW[1],
         v_EI * dW[1] - v_IR * dW[2],
         v_IR * dW[2],
    )


def simulate_seir_sde(beta, sigma, gamma,
                      S0, E0, I0, R0, T, dt,
                      seed=None):
    """
    Simulate one SEIR-SDE trajectory using Euler-Maruyama.

    Returns
    -------
    times  : ndarray (n_steps+1,)
    S,E,I,R: ndarray (n_steps+1,)
    """
    rng = np.random.default_rng(seed)
    N = S0 + E0 + I0 + R0
    n_steps = int(round(T / dt))
    sqrt_dt = np.sqrt(dt)

    times = np.linspace(0, T, n_steps + 1)
    S_arr = np.empty(n_steps + 1)
    E_arr = np.empty(n_steps + 1)
    I_arr = np.empty(n_steps + 1)
    R_arr = np.empty(n_steps + 1)
    S_arr[0], E_arr[0], I_arr[0], R_arr[0] = S0, E0, I0, R0

    S, E, I, R = float(S0), float(E0), float(I0), float(R0)
    for k in range(n_steps):
        fS, fE, fI, fR = _drift(S, E, I, R, beta, sigma, gamma, N)
        dW = rng.standard_normal(4) * sqrt_dt
        nS, nE, nI, nR = _diffusion_noise(S, E, I, R, beta, sigma, gamma, N, dW)
        S = max(S + fS * dt + nS, 0.0)
        E = max(E + fE * dt + nE, 0.0)
        I = max(I + fI * dt + nI, 0.0)
        R = max(R + fR * dt + nR, 0.0)
        total = S + E + I + R
        if total > 0:
            s = N / total
            S, E, I, R = S * s, E * s, I * s, R * s
        S_arr[k+1], E_arr[k+1], I_arr[k+1], R_arr[k+1] = S, E, I, R

    return times, S_arr, E_arr, I_arr, R_arr


# ---------------------------------------------------------------------------
# Prior sampling
# ---------------------------------------------------------------------------

def sample_theta(n_samples, prior_bounds=None, seed=None):
    """
    Sample (beta, sigma, gamma) i.i.d. from the uniform prior.

    Returns
    -------
    theta : ndarray (n_samples, 3)  columns: [beta, sigma, gamma]
    """
    rng = np.random.default_rng(seed)
    bounds = prior_bounds or DEFAULT_PRIOR_BOUNDS
    beta  = rng.uniform(*bounds["beta"],  size=n_samples)
    sigma = rng.uniform(*bounds["sigma"], size=n_samples)
    gamma = rng.uniform(*bounds["gamma"], size=n_samples)
    return np.column_stack([beta, sigma, gamma])


# ---------------------------------------------------------------------------
# Observation extraction
# ---------------------------------------------------------------------------

def extract_obs(times, I_traj, N, schedule, obs_noise_std=0.0, rng=None):
    """
    Extract infected-rate observations at scheduled days.

    Parameters
    ----------
    schedule : tuple of int  (t1, t2, ..., tK) in days

    Returns
    -------
    y : ndarray (K,)
    """
    if rng is None:
        rng = np.random.default_rng()
    dt = times[1] - times[0]
    y = np.empty(len(schedule))
    for j, t in enumerate(schedule):
        idx = int(np.clip(round(t / dt), 0, len(times) - 1))
        y[j] = I_traj[idx] / N
        if obs_noise_std > 0:
            y[j] += rng.normal(0.0, obs_noise_std)
    return y


# ---------------------------------------------------------------------------
# IG proxy evaluation
# ---------------------------------------------------------------------------

def evaluate_ig_proxy(schedule, theta_samples, model_config=None,
                      n_paths_per_theta=5, lambda_reg=1e-6,
                      obs_noise_std=0.0, seed=None):
    """
    Compute IG_proxy(a) = logdet( Cov_{theta}(y_a) + lambda*I ).

    For each theta, simulate n_paths_per_theta SDE paths and average
    the observation vectors to suppress process noise, then take the
    covariance across theta samples.

    Parameters
    ----------
    schedule      : tuple of int
    theta_samples : ndarray (n_theta, 3)
    model_config  : dict (see DEFAULT_MODEL_CONFIG)
    n_paths_per_theta : int
    lambda_reg    : float
    obs_noise_std : float
    seed          : int or None

    Returns
    -------
    ig_proxy : float
    Y_bar    : ndarray (n_theta, K)  path-averaged observations per theta
    """
    cfg = model_config or DEFAULT_MODEL_CONFIG
    rng = np.random.default_rng(seed)
    n_theta = theta_samples.shape[0]
    K = len(schedule)
    Y_bar = np.zeros((n_theta, K))

    for i in range(n_theta):
        beta, sigma, gamma = theta_samples[i]
        Y_paths = np.zeros((n_paths_per_theta, K))
        for p in range(n_paths_per_theta):
            path_seed = int(rng.integers(0, 2**31))
            times, _, _, I_traj, _ = simulate_seir_sde(
                beta, sigma, gamma,
                cfg["S0"], cfg["E0"], cfg["I0"], cfg["R0"],
                cfg["T"], cfg["dt"],
                seed=path_seed,
            )
            obs_rng = np.random.default_rng(int(rng.integers(0, 2**31)))
            Y_paths[p] = extract_obs(times, I_traj, cfg["N"], schedule,
                                     obs_noise_std=obs_noise_std, rng=obs_rng)
        Y_bar[i] = Y_paths.mean(axis=0)

    cov_Y = np.cov(Y_bar, rowvar=False)
    if cov_Y.ndim == 0:
        cov_Y = np.array([[float(cov_Y)]])
    reg = cov_Y + lambda_reg * np.eye(K)
    sign, logdet = np.linalg.slogdet(reg)
    return (logdet if sign > 0 else -np.inf), Y_bar


def evaluate_all_schedules_ig(schedule_dict, theta_samples, model_config=None,
                               n_paths_per_theta=5, lambda_reg=1e-6,
                               obs_noise_std=0.0, seed=None, verbose=False):
    """
    Compute IG proxy for every schedule in schedule_dict.

    Parameters
    ----------
    schedule_dict : dict  {label: (t1,...,tK)}

    Returns
    -------
    ig_scores : dict {label: float}
    """
    rng = np.random.default_rng(seed)
    ig_scores = {}
    for label, schedule in schedule_dict.items():
        s_seed = int(rng.integers(0, 2**31))
        ig, _ = evaluate_ig_proxy(schedule, theta_samples,
                                  model_config=model_config,
                                  n_paths_per_theta=n_paths_per_theta,
                                  lambda_reg=lambda_reg,
                                  obs_noise_std=obs_noise_std,
                                  seed=s_seed)
        ig_scores[label] = ig
        if verbose:
            print(f"  {label} {schedule}  IG={ig:.4f}")
    return ig_scores


# ---------------------------------------------------------------------------
# Trajectory summary (used by SEIRAgent for context building)
# ---------------------------------------------------------------------------

def trajectory_summary(schedule, theta_samples, model_config=None,
                        n_paths_per_theta=3, seed=None):
    """
    For each observation date in schedule, return:
      - mean infected rate across (theta, path) combinations
      - std of infected rate across theta-averaged paths

    Returns
    -------
    means : ndarray (K,)
    stds  : ndarray (K,)
    """
    cfg = model_config or DEFAULT_MODEL_CONFIG
    rng = np.random.default_rng(seed)
    K = len(schedule)
    Y_bar = np.zeros((len(theta_samples), K))

    for i, (beta, sigma, gamma) in enumerate(theta_samples):
        paths = np.zeros((n_paths_per_theta, K))
        for p in range(n_paths_per_theta):
            times, _, _, I_traj, _ = simulate_seir_sde(
                beta, sigma, gamma,
                cfg["S0"], cfg["E0"], cfg["I0"], cfg["R0"],
                cfg["T"], cfg["dt"],
                seed=int(rng.integers(0, 2**31)),
            )
            paths[p] = extract_obs(times, I_traj, cfg["N"], schedule)
        Y_bar[i] = paths.mean(axis=0)

    return Y_bar.mean(axis=0), Y_bar.std(axis=0)
