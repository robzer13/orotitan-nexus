"""Universe helpers and defaults."""
from __future__ import annotations

from typing import List

from .config import DEFAULT_CAC40_TICKERS, UniverseSettings


def load_universe(settings: UniverseSettings) -> List[str]:
    """Return the list of tickers to screen, falling back to CAC40."""

    tickers = settings.tickers or DEFAULT_CAC40_TICKERS
    # Preserve order while removing duplicates
    seen = set()
    ordered = []
    for ticker in tickers:
        if ticker not in seen:
            ordered.append(ticker)
            seen.add(ticker)
    return ordered
