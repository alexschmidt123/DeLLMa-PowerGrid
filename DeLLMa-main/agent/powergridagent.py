import os
import sys
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional

import pandas as pd

from agent.agent import (
    PROJECT_ROOT,
    DeLLMaAgent,
    StateConfig,
    ActionConfig,
    PreferenceConfig,
)

sys.path.append(PROJECT_ROOT)
from utils.data_utils import POWERGRID_CLUSTERS

CLUSTER_ID_TO_LABEL: Dict[str, str] = {
    cid: f"microgrid regime {int(cid.split('_')[1])} ({cid})"
    for cid in POWERGRID_CLUSTERS
}


class PowerGridAgent(DeLLMaAgent):
    """
    Grid operator agent: choices are k-means regimes, each backed only by
    data/powergrid/cluster_XX.csv (same information interface style as TradeAgent
    reading one CSV per stock).
    """

    system_content = (
        "You are an expert power systems engineer helping a grid operator "
        "choose which microgrid operating regime to prioritize under uncertainty."
    )
    period = "operating interval"
    unit: str = "budget units"
    product: str = "microgrid_regime"

    states: Dict[str, Dict[str, str]] = {
        "agnostic": {
            "aggregate load forecast error": "discrepancy between forecast and realized net load across the 4-node system",
            "renewable generation uncertainty": "variability and forecast error of distributed or renewable-like injections represented in the scenario sample",
            "contingency and equipment stress": "probability of branch or unit outages or protection events not fully captured in the static features",
            "market and reserve pricing": "cost and availability of balancing reserves and ancillary services relevant to stability",
            "regulatory / operational limit": "constraints on voltage, thermal limits, or curtailment policies affecting feasible actions",
        },
        "specific": {
            f"local overload risk for": f"risk of thermal or stability stress localized to the subsystem summarized by this regime's scenarios",
            f"response margin for": f"headroom in participant reaction times (tau) and power balance (p, g) implied by this regime's sample",
            f"stability outlook for": f"expected stability margin and stable/unstable mix in this regime's historical sample",
        },
    }

    def __init__(
        self,
        choices: List[str],
        path: str = os.path.join(PROJECT_ROOT, "data/powergrid/"),
        raw_context_fname: str | None = None,
        temperature: float = 0.0,
        state_config: Optional[dataclass] = None,
        action_config: Optional[ActionConfig] = None,
        preference_config: Optional[PreferenceConfig] = None,
        agent_name: str = "powergrid",
        summary_sample_rows: int = 5,
        reveal_stability_stats: bool = False,
    ):
        assert set(choices).issubset(set(POWERGRID_CLUSTERS)), (
            f"choices must be subsets of {POWERGRID_CLUSTERS}, got {choices}"
        )
        self.choices = sorted(set(choices))
        self.path = path
        self.summary_sample_rows = max(0, int(summary_sample_rows))
        self.reveal_stability_stats = bool(reveal_stability_stats)

        if self.reveal_stability_stats:
            utility_prompt = (
                f"I'm a grid operator planning the next operating move. I would like to "
                f"prioritize risk mitigation and secure operation using '{action_config.budget}' "
                f"{self.unit}. In this dataset, **stab < 0** indicates stable operation and "
                f"**stab > 0** indicates unstable; **stabf** is the binary stability label. "
                f"For protection-first triage, prefer regimes with **higher (more positive) "
                f"mean stab** and a **larger fraction of `unstable`** rows in the sample."
            )
        else:
            # Outcome columns (evaluation-only) are omitted from the prompt entirely.
            utility_prompt = (
                f"I'm a grid operator planning the next operating move. I would like to "
                f"prioritize **operational stability** and secure operation using '{action_config.budget}' "
                f"{self.unit}. Use **only** the **τ / p / g** summaries and scenario rows below; "
                f"**outcome / label columns are withheld** from this prompt (they exist only for offline evaluation)."
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
        )

        if (
            self.state_config.state_enum_mode != "base"
            and len(self.state_config.states) == 0
        ):
            self.state_config.states = self._format_state_dict()

    def _format_state_dict(self) -> Dict[str, str]:
        state2desc = self.states["agnostic"].copy()
        for choice, variable in product(
            self.choices, sorted(self.states["specific"].keys())
        ):
            label = CLUSTER_ID_TO_LABEL[choice]
            key = f"{variable} {label}".lower()
            state2desc[key] = self.states["specific"][variable]
        return state2desc

    def _read_cluster_csv(self, cluster_id: str) -> pd.DataFrame:
        fp = os.path.join(self.path, f"{cluster_id}.csv")
        if not os.path.isfile(fp):
            raise FileNotFoundError(
                f"PowerGridAgent only loads per-regime files under {self.path}; "
                f"missing {fp}"
            )
        return pd.read_csv(fp)

    def _format_cluster_context(self, cluster_id: str) -> str:
        df = self._read_cluster_csv(cluster_id)
        label = CLUSTER_ID_TO_LABEL[cluster_id]
        feat_tau = [f"tau{i}" for i in range(1, 5)]
        feat_p = [f"p{i}" for i in range(1, 5)]
        feat_g = [f"g{i}" for i in range(1, 5)]

        def mean_std(cols):
            lines = []
            for c in cols:
                lines.append(f"    mean {c}: {df[c].mean():.6f} (std {df[c].std():.6f})")
            return "\n".join(lines)

        block = f"""=== {cluster_id} — {label} ===
Number of scenarios in this regime sample: {len(df)}
"""
        if self.reveal_stability_stats:
            frac_unstable = (df["stabf"].astype(str).str.lower() == "unstable").mean()
            block += f"""Mean stability margin stab: {df['stab'].mean():.6f} (std {df['stab'].std():.6f})
Fraction of scenarios labeled unstable (stabf): {frac_unstable:.4f}

"""

        block += f"""Aggregated features (tau — participant reaction times):
{mean_std(feat_tau)}
Aggregated features (p — power-related node values):
{mean_std(feat_p)}
Aggregated features (g — pricing / elasticity coefficients):
{mean_std(feat_g)}
"""
        if self.summary_sample_rows > 0 and len(df) > 0:
            cols = feat_tau + feat_p + feat_g
            if self.reveal_stability_stats:
                cols = cols + ["stab", "stabf"]
            tail = df.tail(min(self.summary_sample_rows, len(df)))[cols]
            block += "\nExample scenarios (most recent rows in this regime file; not time-series):\n"
            if not self.reveal_stability_stats:
                block += "(outcome / label columns are omitted)\n"
            block += tail.to_string(index=False)
            block += "\n"
        block += "\n"
        return block

    def prepare_context(self) -> str:
        choice_str = ", ".join(self.choices)
        common = (
            f"Below are the microgrid regimes I am considering: {choice_str}. "
            f"Each regime is summarized using **only** its dedicated file "
            f"`cluster_XX.csv` under the powergrid data folder (one file per regime). "
            f"I would like to know which single regime I should commit to for the next "
            f"{self.period}, given a budget of {self.action_config.budget} {self.unit}. "
        )
        if self.reveal_stability_stats:
            header = (
                f"{common}"
                f"The objective is to identify the highest-risk regime first for protection "
                f"under the dataset's definition (`stab`>0 unstable, `stab`<0 stable).\n\n"
            )
        else:
            header = (
                f"{common}"
                f"The objective is to improve expected **operational stability** using "
                f"only the information shown (outcome labels are **muted** in this prompt; "
                f"used only for offline evaluation).\n\n"
            )
        parts = [header]
        for cid in self.choices:
            parts.append(self._format_cluster_context(cid))
        return "".join(parts)


if __name__ == "__main__":
    agent = PowerGridAgent(
        choices=["cluster_00", "cluster_01", "cluster_02"],
        state_config=StateConfig("sequential"),
        action_config=ActionConfig(budget=10000),
        preference_config=PreferenceConfig(),
    )
    print(agent.prepare_context()[:2000])
