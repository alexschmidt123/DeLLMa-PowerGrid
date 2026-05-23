from collections import deque
from typing import List, Tuple, Optional
import pandas as pd
from itertools import combinations

# ---------------------------------------------------------------------------
# SEIR city-action pool
# ---------------------------------------------------------------------------
# Keys align with generated files under:
# data/seir/observed_window/city1.csv ... city10.csv
SEIR_SCHEDULES = [f"city{i}" for i in range(1, 11)]

# State variables known to every schedule (used in rank / rank-minibatch parsing)
SEIR_AGNOSTIC_STATES = [
    "epidemic peak timing",
    "transmission rate level",
    "recovery rate level",
    "epidemic growth rate",
]

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

# Must match keys in PowerGridAgent.states["agnostic"] for rank parsing
POWERGRID_AGNOSTIC_STATES = [
    "aggregate load forecast error",
    "renewable generation uncertainty",
    "contingency and equipment stress",
    "market and reserve pricing",
    "regulatory / operational limit",
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

# K-means regimes; files are data/powergrid/cluster_00.csv ... (k-1); k=7 matches
# farmer/trader combinatorics: 2^7 - 1 - 7 = 120 choice-set runs.
POWERGRID_CLUSTERS = [f"cluster_{i:02d}" for i in range(7)]

# All symbols with CSVs under data/stocks/. First 5 = legacy stocks_5 (--max_choices 5).
STOCKS = ["AMD", "BILI", "DIS", "GE", "GME", "GOOGL", "META", "NVDA", "SPY"]
STOCKS_5 = STOCKS[:5]
STOCKS_SYMBOL_TO_NAME_MAP = {
    "AMD": "Advanced Micro Devices",
    "BILI": "Bilibili Inc.",
    "DIS": "The Walt Disney Company",
    "GE": "General Electric",
    "GME": "GameStop Corp",
    "GOOGL": "Alphabet, i.e. Google",
    "META": "Meta Platforms, i.e. Facebook",
    "NVDA": "NVIDIA",
    "SPY": "S&P 500",
}


def get_product_pool(
    agent_name: str,
    source_year: Optional[str] = None,
    max_choices: Optional[int] = None,
) -> List[str]:
    """
  Return the ordered list of choice items (fruits, stocks, clusters, cities).

  If max_choices is set, keep only the first max_choices entries in that list.
  """
    if agent_name == "farmer":
        if source_year is None:
            raise ValueError("source_year is required for farmer.")
        products = list(FRUITS[source_year])
    elif agent_name == "trader":
        products = list(STOCKS)
    elif agent_name == "powergrid":
        products = list(POWERGRID_CLUSTERS)
    elif agent_name == "seir":
        products = list(SEIR_SCHEDULES)
    else:
        raise ValueError(
            "agent_name must be one of 'farmer', 'trader', 'powergrid', or 'seir'"
        )

    if max_choices is None:
        return products
    if max_choices < 2:
        raise ValueError(
            f"max_choices must be at least 2 (got {max_choices}); "
            "choice sets combine 2 or more items."
        )
    if max_choices > len(products):
        raise ValueError(
            f"max_choices={max_choices} exceeds pool size {len(products)} for {agent_name}."
        )
    return products[:max_choices]


def get_combinations(
    agent_name: str,
    source_year: Optional[str] = None,
    max_choices: Optional[int] = None,
) -> List[Tuple[str, ...]]:
    combs = []
    products = get_product_pool(agent_name, source_year=source_year, max_choices=max_choices)

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
