# Practical Modification Plan: DeLLMa + GridMind Patterns

This is an **implementation order** for modifying this repo so DeLLMa uses **solver-grounded context** (GridMind-style) before running belief elicitation and ranking.

---

## PI constraint: extend farm/stock → power grid with **minimal change** to DeLLMa core

**Goal:** Treat power grid as a **third application domain**, like agriculture and stocks—not a rewrite of the decision pipeline.

| **Leave unchanged (DeLLMa “core”)** | **Extend (same pattern as `FarmAgent` / `TradeAgent`)** |
|-------------------------------------|---------------------------------------------------------|
| `belief2score`, `load_state_beliefs`, `sample_state`, `sample_state_action_pairs_batch`, ranking → choix ILSR, softmax, `prepare_dellma_prompt` structure | New **`GridAgent`** in **`agent/gridagent.py`** that subclasses `DeLLMaAgent` and implements `prepare_context`, `cache_context`, and grid-specific fields—**mirroring** how `farmagent.py` / `tradeagent.py` work |
| Prompt utilities in `utils/prompt_utils.py` (unless you hit a real bug) | New **`data/powergrid/`** layout + **`cache/grid_states.json`** (same JSON belief format as farmer) |
| Math / ranking logic | Optional **`tools/`** for PandaPower summaries (numbers stay outside the LLM; GridMind-style) |

**Smallest possible edits to *existing* files (typical PR):**

1. **`agent/agent.py`** — allow `agent_name` to include `"grid"` (today only `"farmer"` / `"trader"`). *One line change to the validation list.*
2. **`utils/data_utils.py`** — add grid product list / state descriptors and a `get_combinations(..., "grid")` branch. *Additive only.*
3. **`main.py`** — add `elif args.agent_name == "grid":` with `partial(GridAgent, ...)`, `domain`, `budget`, result paths. *Additive only.*
4. **`evaluate_dellma.py`** — optional `grid` branch for your metric. *Additive only.*

**No need** to refactor `DeLLMaAgent` methods, change Bradley–Terry, or touch ranking minibatch logic—unless your PI explicitly wants algorithm research.

**Reuse base behavior:** `DeLLMaAgent.__init__` expects `path/reports/summary/`; you can satisfy that by using e.g. `data/powergrid/` with `reports/summary/` under it (same as farmer) so you **do not** have to change directory logic in `agent.py` beyond the agent name.

---

## Design principle (one line)

**All numbers come from Python tools (e.g. PandaPower). The LLM only sees text built from validated tool outputs + your goal; DeLLMa’s pipeline (belief → sample → rank → Bradley–Terry) stays the same.**

---

## Phase 0 — Tool layer only (no LLM, no DeLLMa changes yet)

**Goal:** Reproduce one IEEE case end-to-end in code: load network → run power flow (and optionally ACOPF / one N-1) → return a **dict of numbers** → convert to a **short string** (this string will become part of \(C\)).

| Task | Deliverable |
|------|-------------|
| Pick case | e.g. IEEE 14 from PandaPower / MATPOWER import |
| Implement `run_base_case()` | Returns structured fields: converged, min V, max line loading %, total gen cost (if OPF), etc. |
| Implement `format_case_summary(d: dict) -> str` | 10–40 lines of prose + optional tiny markdown table |
| Optional | `filter_feasible_actions(actions, case)` — drop probes/topologies that fail PF or limits |

**New files (suggested):**

- `tools/grid_solver.py` (or `powergrid/pandapower_tools.py`) — all numerical calls here.
- `tools/grid_context_format.py` — dict → text for prompts.

**Dependency:** Add to `requirements.txt` when ready, e.g. `pandapower` (GridMind uses PandaPower).

**Exit criterion:** Running one script prints a summary string; you can paste that string into ChatGPT and it reads like a human grid report (no Y-bus matrices).

---

## Phase 1 — Data layout and cache (mirror farmer/trader)

**Goal:** A stable place for grid-specific assets and belief cache.

| Path | Purpose |
|------|---------|
| `data/powergrid/` | Optional: case files, small CSVs, or notes (PandaPower can also build from code) |
| `cache/grid_states.json` | Belief over discrete state factors (same format as `cache/farmer_2021_states.json`: verbal labels per factor) |

**Exit criterion:** You can hand-edit `cache/grid_states.json` with 3–5 state variables (e.g. load level, wind level, outage scenario) and discrete values each, matching keys you will put in `StateConfig.states`.

---

## Phase 2 — Minimal touch on `agent/agent.py` (PI-friendly)

Today `DeLLMaAgent` only allows `agent_name in ["farmer", "trader"]`.

**Preferred minimal change:** add `"grid"` to that list so `GridAgent(..., agent_name="grid")` works. **Do not** refactor the rest of the base class.

**Directory layout:** `GridAgent` uses `path=data/powergrid/simulation/pandapower/` with `data_layout="flat"`. Under `data/powergrid/` only **`authoritative/`** and **`simulation/`**; under **`simulation/`** only **`pandapower/`** and **`matlab_simulink/`**. Farmer still uses `path/reports/...` (default `data_layout="reports"`).

**Optional:** other agents unchanged; only grid passes the flat layout flag.

**Belief cache:** `cache_state_beliefs` / `load_state_beliefs` already use `cache/{agent_name}_states.json` when there is no `source_year` — use `agent_name="grid"` → `cache/grid_states.json`.

**Exit criterion:** Instantiating `GridAgent` does not raise `ValueError` on agent name.

---

## Phase 3 — New `GridAgent` (`agent/gridagent.py`)

**Goal:** Same interface as `FarmAgent` / `TradeAgent` for anything that calls `prepare_dellma_prompt()`, `prepare_context()`, etc.

| Method | GridMind-aligned behavior |
|--------|---------------------------|
| `__init__` | `choices` = list of **discrete action strings** (e.g. probe IDs, “reinforce line 6”, “topology A”). `utility_prompt` = your goal in plain English. |
| `prepare_context()` | **Call Phase 0 tools** → get `dict` → `format_case_summary(dict)` → prepend goal + date + case name. This is the main \(C\). |
| `cache_context` | Either **no-op** returning cached string, or call LLM once to compress tool output to JSON summary (farmer pattern) — optional. |
| `system_content` | “You are a power system operator / analyst assisting with decisions under uncertainty…” |

**Important:** Do **not** put raw Jacobian or full branch matrices in the prompt. Only summaries and key scalars (as in GridMind’s narrative + structured fields).

**Exit criterion:** `prepare_context()` returns a string that includes tool-derived numbers only from your Python functions.

---

## Phase 4 — `utils/data_utils.py`

**Add:**

- `GRID_ACTIONS` or a function listing default discrete actions for experiments.
- `GRID_STATES` structure analogous to `FRUIT_STATES` / `TradeAgent.states` — agnostic + optional per-element descriptors.
- `get_combinations`: branch `elif agent_name == "grid":` — return combinations of actions you want to sweep (or a single tuple for a fixed experiment).

---

## Phase 5 — Wire `main.py`

**Add:**

- `parser.add_argument("--agent_name", ..., choices=["farmer", "trader", "grid"])` (or separate flag).
- Branch `elif args.agent_name == "grid":` → `domain = "powergrid"`, `agent_init_fct = partial(GridAgent, ...)`, set `budget` / units appropriately (e.g. dollars for investment, or `1` for “one probe”).
- Set `result_folder` to something like `results/powergrid/...` so runs do not overwrite agriculture/stocks.

**Exit criterion:** `python main.py --agent_name grid ...` creates prompts under `results/powergrid/...` without touching farmer code paths.

---

## Phase 6 — `evaluate_dellma.py` (your metric)

**Goal:** After DeLLMa outputs an action index, score it with **code**, not the LLM.

| Step | Example |
|------|---------|
| Parse chosen action | Map index → probe / mitigation id |
| Simulator | Run PandaPower (or your dynamic model) for that action under ground-truth or sampled scenario |
| Metric | Cost, violation count, variance of \(\hat M\), etc. |

Add a branch `if agent_name == "grid":` parallel to farmer/stock evaluation.

**Exit criterion:** You can report numeric performance vs baseline (e.g. random action, always smallest probe).

---

## Phase 7 — GridMind-style safeguards (recommended)

| Practice | Where |
|----------|--------|
| Validate solver convergence | Inside `tools/grid_solver.py` before formatting \(C\) |
| Every number in an explanation traceable | Prefer storing last tool `dict` on the agent for logging |
| Optional Pydantic models | `tools/schemas.py` for `ACOPFResult`, `PowerFlowResult` — serialize to JSON for logs (GridMind uses typed context) |
| Do not trust LLM for feasibility | Filter `choices` before DeLLMa, or re-run PF after decision and discard if infeasible |

---

## File checklist (summary)

| File | Change |
|------|--------|
| `tools/grid_solver.py` | **New** — PandaPower / PF / OPF |
| `tools/grid_context_format.py` | **New** — dict → prompt text |
| `agent/gridagent.py` | **New** — `GridAgent` |
| `agent/agent.py` | Allow `grid` agent name; optional path handling |
| `utils/data_utils.py` | Grid states, actions, `get_combinations` |
| `main.py` | `--agent_name grid`, paths, init |
| `evaluate_dellma.py` | Grid evaluation branch |
| `requirements.txt` | `pandapower` (when you add tools) |

---

## Order of implementation (strict)

1. Phase 0 (tools + string \(C\))  
2. Phase 1 (cache + hand-written `grid_states.json`)  
3. Phase 2 + 3 (`grid` agent + `prepare_context`)  
4. Phase 4 + 5 (combinations + `main.py`)  
5. Phase 6 (evaluation)  
6. Phase 7 (hardening)

---

## What you are *not* changing in DeLLMa core

- Belief scoring map `belief2score`, `load_state_beliefs`, `sample_state`, `sample_state_action_pairs_batch`, ranking modes, choix ILSR — **keep as-is** unless you have a research reason to change them.

---

## Implementation plan (step-by-step)

Use this as a **ticket list**. Order matters; each milestone has a clear **done** criterion.

### Milestone A — Tool layer only (physics firewall)

| # | Task | Done when |
|---|------|-----------|
| A1 | Create `tools/` package (`__init__.py` if needed). | Folder importable. |
| A2 | Implement `grid_solver.py`: load IEEE case (e.g. 14-bus via PandaPower), run `runpp` (and optionally one N-1 or OPF later). | Returns a **dict** with `converged`, `min_vm_pu`, `max_line_loading_percent`, etc. |
| A3 | Implement `format_case_summary(d) -> str` in `grid_context_format.py` (or same file). | String is human-readable; **no** raw Y-bus / Jacobian. |
| A4 | Add `pandapower` to `requirements.txt`. | `pip install -r requirements.txt` succeeds. |
| A5 | Small script or `if __name__ == "__main__"` that prints one summary. | You can paste output into an LLM as fake \(C\). |

**Hybrid logic:** This milestone = **GridMind-style** “absolute physical baseline” from solvers only.

---

### Milestone B — Data + belief cache (same format as farmer)

| # | Task | Done when |
|---|------|-----------|
| B1 | Create `data/powergrid/simulation/pandapower/summary/` (grid `path` is `simulation/pandapower/`). | Dirs exist. |
| B2 | Add `data/powergrid/simulation/pandapower/grid_baseline.txt` stub **or** generate text only from tools. | `raw_context_fname` resolves under `path/`. |
| B3 | Author `cache/grid_states.json`: 3–5 discrete state factors, verbal labels per value (same schema as `cache/farmer_2021_states.json`). | Keys match `StateConfig.states` you will set in code. |

---

### Milestone C — `GridAgent` (subjective side + context)

| # | Task | Done when |
|---|------|-----------|
| C1 | Add `agent/gridagent.py`: class `GridAgent(DeLLMaAgent)`, `system_content`, `unit`, `product`. | Imports work. |
| C2 | Implement `cache_context`: return string or dict like `FarmAgent` (can cache tool summary JSON under `reports/summary/`). | No crash on init. |
| C3 | Implement `prepare_context()`: prepend goal + call **A2–A3** so \(C\) = text grounded in solver output. | Prompt string contains only tool-derived numbers (via formatter). |
| C4 | Set `choices` from discrete actions (probe IDs / mitigations); match `data_utils` list. | `prepare_actions()` builds `action_strs` like farmer. |

**Hybrid logic:** DeLLMa still does **belief + ranking**; `GridAgent` only supplies **grid-specific \(C\)** and actions.

---

### Milestone D — Minimal edits to existing core files

| # | File | Edit | Done when |
|---|------|------|-----------|
| D1 | `agent/agent.py` | Allow `agent_name in [..., "grid"]`. | No `ValueError` for grid. |
| D2 | `utils/data_utils.py` | `GRID_*` or lists; `get_combinations(..., "grid")`. | Returns list of choice tuples. |
| D3 | `main.py` | `choices=["farmer","trader","grid"]`; `elif agent_name == "grid":` → `GridAgent`, `domain="powergrid"`, `budget`, `result_folder`. | `python main.py --agent_name grid ...` runs to prompt generation. |

---

### Milestone E — End-to-end DeLLMa run (no new algorithm)

| # | Task | Done when |
|---|------|-----------|
| E1 | Run with `dellma_mode=rank-minibatch` (or `rank`) like stocks. | Prompts saved under `results/powergrid/...`. |
| E2 | Confirm prompts include: context \(C\) from tools + state–action pair lines. | Manual inspection of one `prompt_0.txt`. |
| E3 | (Optional) Run inference API if keys configured; else stop at prompts. | You have a reproducible experiment folder. |

---

### Milestone F — Evaluation (physics check after decision)

| # | Task | Done when |
|---|------|-----------|
| F1 | `evaluate_dellma.py`: branch for `agent_name == "grid"`. | Parses chosen action index. |
| F2 | Map index → probe/action; re-run solver or metric in code. | Numeric score logged (cost, violation, proxy for information). |

**Hybrid logic:** Evaluation uses **tools**, not LLM, for “what happened if we applied \(\xi^\star\).”

---

### Milestone G — Hardening (recommended)

| # | Task | Done when |
|---|------|-----------|
| G1 | If PF fails, `format_case_summary` states failure; optional filter removes infeasible actions from `choices` before DeLLMa. | No silent success on diverged PF. |
| G2 | Log tool dict + prompt hash next to each run. | Auditable trail (GridMind-style provenance). |

---

### Workflow mapping (your PI’s hybrid synthesis)

1. **Tools (`grid_solver`)** compute what happens physically for each candidate probe / scenario (deterministic).  
2. **LLM** (via DeLLMa prompts) ranks **(state, action)** pairs using operator goals in language.  
3. **Bradley–Terry / existing DeLLMa code** aggregates rankings → \(\bar{U}(a)\) → choose \(\xi^\star\).  
4. **Evaluate** with tools again on \(\xi^\star\) — **never** ask the LLM for ROCOF/PF numbers as ground truth.

---

### Estimated effort (rough)

| Milestone | Effort |
|-----------|--------|
| A | 0.5–2 days (solver familiarity) |
| B–D | 0.5–1 day |
| E | few hours |
| F | 0.5–1 day |
| G | ongoing |

---

## References in repo

- `docs/HOW_TO_USE_LLM_IN_POWER_GRID.md` — roles of LLM vs solvers  
- `docs/NEW_IDEAS_DeLLMa_GridMind.md` — combining both papers conceptually  
- `reproduction_visualization/DeLLMa_math.tex` — math for the unchanged DeLLMa core  
