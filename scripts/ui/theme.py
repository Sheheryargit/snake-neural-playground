"""Visual theme and typography for the Snake RL dashboard."""

from dataclasses import dataclass
from typing import Dict, Tuple

import pygame

Color = Tuple[int, int, int]


@dataclass(frozen=True)
class Theme:
    # Surfaces
    bg: Color = (6, 9, 15)
    bg_gradient_top: Color = (10, 16, 28)
    bg_elevated: Color = (12, 18, 30)
    panel: Color = (16, 22, 34)
    panel_hover: Color = (22, 30, 46)
    panel_active: Color = (26, 38, 58)
    border: Color = (38, 52, 76)
    border_soft: Color = (26, 36, 54)

    # Text
    text: Color = (244, 247, 252)
    text_secondary: Color = (148, 162, 188)
    text_muted: Color = (88, 102, 126)

    # Accents
    accent: Color = (64, 196, 255)
    accent_soft: Color = (24, 72, 108)
    success: Color = (46, 212, 158)
    warning: Color = (245, 188, 66)
    danger: Color = (255, 108, 108)
    purple: Color = (172, 148, 255)
    manual: Color = (255, 180, 72)

    # Game
    board_bg: Color = (10, 15, 24)
    board_grid: Color = (14, 20, 32)
    board_grid_line: Color = (22, 30, 46)
    snake_body: Color = (30, 128, 84)
    snake_body_dark: Color = (20, 96, 62)
    snake_head: Color = (80, 230, 140)
    snake_head_glow: Color = (36, 150, 92)
    food: Color = (245, 72, 72)
    food_glow: Color = (170, 36, 48)
    food_highlight: Color = (255, 130, 130)

    # Agent colors
    agent_colors: Dict[int, Color] = None

    def __post_init__(self):
        object.__setattr__(
            self,
            "agent_colors",
            {
                1: (148, 163, 184),
                2: (245, 188, 66),
                3: (64, 196, 255),
                4: (172, 148, 255),
                5: (46, 212, 158),
                6: (248, 128, 188),
                7: (255, 180, 72),
            },
        )


THEME = Theme()

FONT_NAMES = ("SF Pro Display", "Segoe UI", "Helvetica Neue", "Arial", "sans-serif")


def font(size: int, bold: bool = False) -> pygame.font.Font:
    return pygame.font.SysFont(FONT_NAMES, size, bold=bold)


def draw_background(screen: pygame.Surface) -> None:
    """Subtle vertical gradient background."""
    width, height = screen.get_size()
    for y in range(height):
        t = y / max(height - 1, 1)
        r = int(THEME.bg_gradient_top[0] * (1 - t) + THEME.bg[0] * t)
        g = int(THEME.bg_gradient_top[1] * (1 - t) + THEME.bg[1] * t)
        b = int(THEME.bg_gradient_top[2] * (1 - t) + THEME.bg[2] * t)
        pygame.draw.line(screen, (r, g, b), (0, y), (width, y))
