"""
agent/seiragent.py

DeLLMa agent for vaccine-allocation decision from observed SEIR windows.

Important data-access boundary:
- DeLLMa-visible folder only: data/seir/observed_window/
- Evaluation-only folders are intentionally NOT read by this agent.
"""

import os
import sys
from dataclasses import dataclass
from itertools import product as iproduct
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from agent.agent import (
    PROJECT_ROOT,
    DeLLMaAgent,
    StateConfig,
    ActionConfig,
    PreferenceConfig,
)

sys.path.append(PROJECT_ROOT)
from utils.data_utils import SEIR_SCHEDULES, SEIR_AGNOSTIC_STATES


class SEIRAgent(DeLLMaAgent):
    """
    SEIR vaccine-allocation agent with observed-window city files.

    choices : list of city IDs (city1 ... city10), where each action means
              allocating the first vaccine supply to that city.
    """

    system_content = (
        "You are an expert epidemiologist helping a public health agency choose "
        "which city should receive the first limited vaccine supply. "
        "Use only observed-window epidemic data (days 1-30) to reason about "
        "future burden reduction potential."
    )
    period = "epidemic response window"
    unit = "city allocation slot"
    product = "city_allocation"
    _dataset_cache: Dict[str, Dict[str, object]] = {}

    # Uncertain state dimensions for sequential / rank modes
    states: Dict[str, Dict[str, str]] = {
        "agnostic": {
            "epidemic peak timing": (
                "when the infected level appears to approach a local peak "
                "within the observed 30-day window"
            ),
            "transmission rate level": (
                "the apparent force of transmission implied by I(t) growth "
                "during days 1-30"
            ),
            "recovery rate level": (
                "the apparent recovery pressure implied by R(t) accumulation "
                "during days 1-30"
            ),
            "epidemic growth rate": (
                "how rapidly infection burden increases within the observed window"
            ),
        },
        "specific": {
            "early-window infection acceleration for": (
                "how quickly infection count rises early in the observed window"
            ),
            "late-window burden level for": (
                "how high infection count remains near the end of observed days"
            ),
        },
    }

    def __init__(
        self,
        choices: List[str],
        path: str = os.path.join(PROJECT_ROOT, "data/seir/"),
        temperature: float = 0.0,
        state_config: Optional[dataclass] = None,
        action_config: Optional[ActionConfig] = None,
        preference_config: Optional[PreferenceConfig] = None,
        agent_name: str = "seir",
        model_config: Optional[dict] = None,
        prior_bounds: Optional[dict] = None,
        obs_noise_std: float = 0.0,
    ):
        assert set(choices).issubset(set(SEIR_SCHEDULES)), (
            f"choices must be a subset of {list(SEIR_SCHEDULES)}, got {choices}"
        )
        self.choices = sorted(choices)
        self.path = path
        self.model_config = model_config or {}
        self.prior_bounds  = prior_bounds  or {}
        self.obs_noise_std = obs_noise_std

        window_days = self.model_config.get("T", "the configured")
        utility_prompt = (
            f"I am a public health epidemiologist deciding vaccine allocation across cities. "
            f"I have a budget of exactly {action_config.budget} initial vaccine allocation "
            f"slot(s) and must choose one city from the candidate list using only "
            f"the observed window. The epidemic horizon is {window_days} days."
        )

        super().__init__(
            path,
            None,
            temperature,
            utility_prompt,
            state_config,
            action_config,
            preference_config,
            agent_name,
        )

        if (
            self.state_config.state_enum_mode != "base"
            and len(self.state_config.states) == 0
        ):
            self.state_config.states = self._format_state_dict()

        self.dataset = self._load_dataset_cache()

    def _load_dataset_cache(self) -> Dict[str, object]:
        data_root = os.path.abspath(self.path)
        observed_root = os.path.join(data_root, "observed_window")
        if not os.path.isdir(observed_root):
            raise FileNotFoundError(
                f"Missing observed-window directory: {observed_root}. "
                "Run seir_sde_data_generation/generate_seir_sde_data.py first."
            )

        missing_design_files = [
            os.path.join(observed_root, f"{city_id}.csv")
            for city_id in self.choices
            if not os.path.isfile(os.path.join(observed_root, f"{city_id}.csv"))
        ]
        if missing_design_files:
            raise FileNotFoundError(
                "SEIRAgent requires observed city CSV files under data/seir/observed_window/. "
                "Missing:\n- " + "\n- ".join(missing_design_files) + "\n"
                "Generate them with seir_sde_data_generation/generate_seir_sde_data.py."
            )

        cached = self._dataset_cache.get(data_root)
        if cached is not None:
            missing = [c for c in self.choices if c not in cached["city_summaries"]]
            if missing:
                cached["city_summaries"].update(
                    self._compute_city_stats(observed_root, missing)
                )
            return cached

        all_city_ids = sorted(
            os.path.splitext(name)[0]
            for name in os.listdir(observed_root)
            if name.endswith(".csv")
        )
        if not all_city_ids:
            raise FileNotFoundError(
                f"No observed-window CSV files found under {observed_root}."
            )
        city_summaries = self._compute_city_stats(observed_root, all_city_ids)
        sample_size = int(next(iter(city_summaries.values()))["n"])

        cached = {
            "num_samples": sample_size,
            "context": {
                "N": self.model_config.get("N", "unknown"),
                "T": int(self.model_config.get("T", 60)),
                "dt": float(self.model_config.get("dt", 0.1)),
            },
            "city_summaries": city_summaries,
        }
        self._dataset_cache[data_root] = cached
        return cached

    def _compute_city_stats(
        self, observed_root: str, city_ids: List[str]
    ) -> Dict[str, Dict[str, object]]:
        summaries: Dict[str, Dict[str, object]] = {}
        for city_id in city_ids:
            fp = os.path.join(observed_root, f"{city_id}.csv")
            df = pd.read_csv(fp).sort_values("day")
            if not {"day", "infected_population", "recovered_population"}.issubset(df.columns):
                raise ValueError(
                    f"Observed file for '{city_id}' must contain columns: "
                    "day, infected_population, recovered_population."
                )
            infected = df["infected_population"].to_numpy(dtype=float)
            recovered = df["recovered_population"].to_numpy(dtype=float)
            summaries[city_id] = {
                "days": df["day"].astype(int).tolist(),
                "infected": infected.tolist(),
                "recovered": recovered.tolist(),
                "infected_mean": float(np.mean(infected)),
                "infected_std": float(np.std(infected)),
                "recovered_mean": float(np.mean(recovered)),
                "recovered_std": float(np.std(recovered)),
                "infected_last": float(infected[-1]),
                "recovered_last": float(recovered[-1]),
                "n": int(len(infected)),
            }
        return summaries

    # ------------------------------------------------------------------
    # State dict for sequential / rank mode
    # ------------------------------------------------------------------

    def _format_state_dict(self) -> Dict[str, str]:
        state2desc = self.states["agnostic"].copy()
        for choice, variable in iproduct(self.choices, sorted(self.states["specific"].keys())):
            key = f"{variable} city {choice}".lower()
            state2desc[key] = self.states["specific"][variable]
        return state2desc

    # ------------------------------------------------------------------
    # Context preparation
    # ------------------------------------------------------------------

    def prepare_context(self) -> str:
        """
        Build context from observed-window data only.
        """
        context_cfg = self.dataset["context"]
        num_samples = self.dataset["num_samples"]
        choice_labels = ", ".join(self.choices)
        header = (
            f"Below are the {len(self.choices)} candidate cities I can allocate "
            f"the first vaccine supply to: {choice_labels}.\n"
            f"Use only observed-window data (days 1-30). Future-window files are not visible.\n"
            f"Each city file contains {num_samples} observed days with infected and recovered counts "
            f"(N={context_cfg['N']}, T={int(context_cfg['T'])}, dt={context_cfg['dt']}).\n"
            f"For each city, I summarize observed infected/recovered statistics and end-of-window levels.\n\n"
        )

        parts = [header]
        for city_id in self.choices:
            stats = self.dataset["city_summaries"][city_id]
            block = self._format_city_block(city_id, stats)
            parts.append(block)

        return "".join(parts)

    def _format_city_block(self, city_id: str, stats: Dict[str, object]) -> str:
        lines = [f"=== {city_id}  —  Observed window days 1-30 ===\n"]
        lines.append(
            f"  Infected mean={stats['infected_mean']:.2f}, std={stats['infected_std']:.2f}, "
            f"last_day={stats['infected_last']:.2f}\n"
        )
        lines.append(
            f"  Recovered mean={stats['recovered_mean']:.2f}, std={stats['recovered_std']:.2f}, "
            f"last_day={stats['recovered_last']:.2f}\n"
        )
        lines.append("\n")
        return "".join(lines)

    # ------------------------------------------------------------------
    # Action formatting (override base to show human-readable schedules)
    # ------------------------------------------------------------------

    def prepare_actions(self) -> str:
        self.actions = [[(c, self.action_config.budget)] for c in self.choices]
        self.action_strs = []
        for i, city_id in enumerate(self.choices):
            self.action_strs.append(
                f"Action {i+1}. {city_id}: allocate the first vaccine supply to {city_id} "
                f"(budget = {self.action_config.budget})"
            )
        merged = "\n".join(self.action_strs)
        return f"Below are the actions I can take:\n{merged}"


if __name__ == "__main__":
    agent = SEIRAgent(
        choices=["city1", "city4", "city10"],
        state_config=StateConfig("sequential"),
        action_config=ActionConfig(budget=3),
        preference_config=PreferenceConfig(),
    )
    print(agent.prepare_context()[:3000])
    print("\n---\n")
    print(agent.prepare_actions())
