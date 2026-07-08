"""Board size limits and helpers for the dashboard."""

from typing import Tuple

MIN_BOARD_SIZE = 4
MAX_BOARD_SIZE = 30
DEFAULT_BOARD_SIZE = 12


def clamp_board_size(value: int) -> int:
    return max(MIN_BOARD_SIZE, min(MAX_BOARD_SIZE, value))


def hamiltonian_compatible(width: int, height: int) -> bool:
    """Hamiltonian agents need at least one even dimension."""
    return width % 2 == 0 or height % 2 == 0


def normalize_for_hamiltonian(width: int, height: int) -> Tuple[int, int]:
    """Bump one dimension to even when both are odd."""
    if hamiltonian_compatible(width, height):
        return width, height
    if width < MAX_BOARD_SIZE:
        return width + 1, height
    if height < MAX_BOARD_SIZE:
        return width, height + 1
    return width - 1, height


def format_board_label(width: int, height: int) -> str:
    cells = width * height
    return f"{width} × {height}  ({cells} cells)"
