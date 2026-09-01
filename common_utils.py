"""Small compatibility helpers for the nominal FJSP data generator."""


def str_to_suffix(value: str) -> str:
    """Convert an optional data suffix to the ``+suffix`` form."""

    return "" if value == "" else f"+{value}"
