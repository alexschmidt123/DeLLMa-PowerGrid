"""
GridAgent: parallel domain for IEEE-14 swing-equation probing under uncertainty.

GridMind-style: numeric context lives in utils/data_utils.py; LLM-facing strings
in utils/prompt_utils.py. DeLLMa core (belief → sample → rank) is unchanged.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from agent.agent import (
    PROJECT_ROOT,
    DeLLMaAgent,
    StateConfig,
    ActionConfig,
    PreferenceConfig,
)

sys.path.insert(0, PROJECT_ROOT)

from utils.data_utils import (  # noqa: E402
    load_ieee14_case_config,
    run_probe_id,
    check_simulation_validity,
    summarize_probe_features,
    mk_posterior_from_belief_dict,
    safety_aware_informativeness,
    get_grid_states,
)
from utils.prompt_utils import build_grid_context  # noqa: E402


def _theta_from_mode(inertia: str, coupling: str) -> Dict[str, float]:
    m_map = {"low": 0.85, "medium": 1.0, "high": 1.15}
    k_map = {"low": 0.9, "medium": 1.0, "high": 1.1}
    return {"M": m_map.get(inertia, 1.0), "K": k_map.get(coupling, 1.0)}


def _modes_from_belief(agent: DeLLMaAgent) -> Tuple[str, str]:
    """Argmax per factor from loaded belief_dist."""
    if not hasattr(agent, "belief_dist"):
        agent.load_state_beliefs()
    inertia_vals, inertia_p = agent.belief_dist["inertia level"]
    coupling_vals, coupling_p = agent.belief_dist["coupling level"]
    i = int(np.argmax(inertia_p))
    j = int(np.argmax(coupling_p))
    return inertia_vals[i], coupling_vals[j]


class GridAgent(DeLLMaAgent):
    """Probe-selection agent for homogeneous M,K identification on IEEE-14 reduced model."""

    system_content = (
        "You are a power-system analysis assistant helping choose informative probing "
        "actions under uncertainty for identifying swing-equation parameter scales (homogeneous "
        "M and K) in an IEEE-14 reduced model. Never invent numerical results. Use only the "
        "solver-grounded context below."
    )
    period = "experiment"
    unit: str = "probe"
    product: str = "grid"

    def __init__(
        self,
        choices: List[str],
        path: str = os.path.join(
            PROJECT_ROOT, "data", "powergrid", "simulation", "pandapower"
        ),
        raw_context_fname: str = "grid_baseline.txt",
        temperature: float = 0.0,
        state_config: Optional[dataclass] = None,
        action_config: Optional[dataclass] = None,
        preference_config: Optional[dataclass] = None,
        agent_name: str = "grid",
    ):
        utility_prompt = (
            "Goal: choose the next discrete probing action that is most informative for "
            "reducing uncertainty about homogeneous inertia (M) and coupling/damping (K) scales, "
            "while remaining reasonably safe (avoid probes that excite excessive ROCOF or "
            "frequency deviation). Base your qualitative reasoning only on the tabulated "
            "solver outputs."
        )
        super().__init__(
            path,
            raw_context_fname,
            temperature,
            utility_prompt,
            state_config,
            action_config,
            preference_config,
            agent_name,
            data_layout="flat",
        )
        self.choices = sorted(set(choices))
        # Belief cache keys must match state_config.states for load_state_beliefs(),
        # including in zero-shot (base) mode where the state block is still empty.
        if self.state_config is not None and len(self.state_config.states) == 0:
            self.state_config.states = get_grid_states()

        self.probe_history: List[Dict[str, Any]] = []
        self.last_tool_result: Optional[List[Dict[str, Any]]] = None
        self.context_cache = self.cache_context(
            self.raw_context_fname,
            self.cache_context_fname,
        )

    def cache_context(
        self,
        raw_fname: str,
        cache_fname: str,
        **kwargs,
    ) -> Dict[str, str]:
        """Persist a tiny marker file; full numbers always come from fresh tool calls."""
        note = {
            "domain": "powergrid",
            "note": "Context text is built deterministically in prepare_context(); see results logs for traces.",
        }
        if cache_fname:
            os.makedirs(os.path.dirname(cache_fname), exist_ok=True)
            with open(cache_fname, "w", encoding="utf-8") as f:
                json.dump(note, f, indent=2)
        return note

    def _belief_posterior_summary(self) -> Any:
        """PosteriorSummary + entropy from verbal belief cache."""
        fname = os.path.join(PROJECT_ROOT, "cache", f"{self.agent_name}_states.json")
        belief = json.load(open(fname, "r", encoding="utf-8"))
        summary, _ = mk_posterior_from_belief_dict(belief, self.belief2score)
        return summary

    def prepare_context(self) -> str:
        """
        Invoke deterministic swing simulations for each candidate probe at the posterior-mode
        theta(M,K) implied by the discrete belief cache.
        """
        case_config = load_ieee14_case_config()
        self.load_state_beliefs()
        inertia_m, coupling_m = _modes_from_belief(self)
        theta = _theta_from_mode(inertia_m, coupling_m)
        post = self._belief_posterior_summary()

        rows: List[Dict[str, Any]] = []
        for aid in self.choices:
            sim, feat = run_probe_id(aid, theta)
            ok, issues = check_simulation_validity(sim)
            row = {
                "action_id": aid,
                "success": sim.success and ok,
                "max_abs_freq_dev_hz": feat.max_abs_freq_dev_hz,
                "max_abs_rocof_hz_s": feat.max_abs_rocof_hz_s,
                "stability_flag": feat.stability_flag,
                "utility_proxy": safety_aware_informativeness(feat),
                "issues": issues,
            }
            rows.append(row)

        self.last_tool_result = rows
        table_lines = [
            "Candidate probes (solver at posterior-mode θ): columns are code-computed.",
            "action_id | max|Δf| (Hz) | max|ROCOF| (Hz/s) | stability | utility_proxy",
        ]
        for r in rows:
            table_lines.append(
                f"{r['action_id']} | {r['max_abs_freq_dev_hz']:.5f} | "
                f"{r['max_abs_rocof_hz_s']:.5f} | {r['stability_flag']} | {r['utility_proxy']:.5f}"
            )

        header = build_grid_context(
            case_config=case_config,
            posterior=post,
            probe_history=self.probe_history,
        )
        theta_line = (
            f"Posterior-mode homogeneous scales used for simulation: M={theta['M']:.3f}, "
            f"K={theta['K']:.3f} (mapped from inertia={inertia_m}, coupling={coupling_m})."
        )
        return (
            f"{header}\n\n{theta_line}\n\n"
            + "\n".join(table_lines)
            + "\n\nAll tabulated numbers were produced by utils/data_utils.py (swing solver)."
        )
