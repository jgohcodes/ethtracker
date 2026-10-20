"""Edge-detection helpers for the watch loop, so alerts fire once per event, not once per poll."""

from __future__ import annotations


def crossed(prev: float | None, price: float, level: float) -> bool:
    """True when price moved from one side of ``level`` to the other (or onto it) since ``prev``."""
    if prev is None:
        return False
    return (prev < level <= price) or (prev > level >= price)


def new_flags(seen: set[tuple[int, str]], current: set[tuple[int, str]]) -> set[tuple[int, str]]:
    """Flags in ``current`` not already reported. Mutates ``seen`` to the current set so cleared flags can re-fire."""
    fresh = current - seen
    seen.clear()
    seen |= current
    return fresh
