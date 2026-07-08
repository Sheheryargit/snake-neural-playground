"""Layout constants and zone helpers — prevents UI overlap."""

from dataclasses import dataclass
from typing import Dict

import pygame

TOPBAR_H = 56
PAD = 24
SIDEBAR_W = 320
BATTLE_SIDEBAR_W = 300
CONTROL_BAR_H = 44
NARRATION_H = 40
MODEL_INFO_H = 148
HUD_H = 52
SECTION_GAP = 10
SECTION_HEADER_H = 18
AGENT_CHIP_ROW_H = 34
AGENT_CHIP_H = 30
AGENT_GRID_COLS = 2
BATTLE_AGENT_COUNT = 7


def agent_grid_height(num_agents: int = BATTLE_AGENT_COUNT) -> int:
    rows = (num_agents + AGENT_GRID_COLS - 1) // AGENT_GRID_COLS
    return rows * AGENT_CHIP_ROW_H


@dataclass
class SidebarLayout:
    """Pre-computed vertical slots for sidebar sections."""
    x: int
    y: int
    width: int
    play_y: int
    presets_y: int
    agents_y: int
    info_y: int
    board_y: int
    view_y: int
    speed_y: int
    tooltip_y: int


def sidebar_layout(screen_w: int, screen_h: int) -> SidebarLayout:
    x = screen_w - SIDEBAR_W - PAD
    y = TOPBAR_H + PAD
    w = SIDEBAR_W
    cy = y
    play_y = cy
    cy += 50 + SECTION_GAP
    presets_y = cy
    cy += 50 + SECTION_GAP
    agents_y = cy
    cy += SECTION_HEADER_H + agent_grid_height(6) + SECTION_GAP
    info_y = cy
    cy += 168 + SECTION_GAP
    board_y = cy
    cy += 100 + SECTION_GAP
    view_y = cy
    cy += 72 + SECTION_GAP
    speed_y = max(cy + SECTION_GAP, screen_h - PAD - 80)
    tooltip_y = screen_h - PAD - 26
    return SidebarLayout(x, y, w, play_y, presets_y, agents_y, info_y, board_y, view_y, speed_y, tooltip_y)


def battle_sidebar_layout(screen_w: int, screen_h: int) -> SidebarLayout:
    """Stacked slots — each section gets explicit vertical space (no overlap)."""
    x = screen_w - BATTLE_SIDEBAR_W - PAD
    y = TOPBAR_H + PAD
    w = BATTLE_SIDEBAR_W
    grid_h = agent_grid_height(BATTLE_AGENT_COUNT)
    cy = y
    play_y = cy
    cy += 44 + SECTION_GAP
    presets_y = cy
    cy += SECTION_HEADER_H + grid_h + SECTION_GAP
    agents_y = cy
    cy += SECTION_HEADER_H + grid_h + SECTION_GAP
    board_y = cy
    cy += 44 + SECTION_GAP
    view_y = cy
    cy += SECTION_HEADER_H + 36 + SECTION_GAP
    speed_y = cy
    tooltip_y = screen_h - PAD - 28
    return SidebarLayout(
        x=x, y=y, width=w,
        play_y=play_y,
        presets_y=presets_y,
        agents_y=agents_y,
        info_y=play_y,
        board_y=board_y,
        view_y=view_y,
        speed_y=speed_y,
        tooltip_y=tooltip_y,
    )


def main_content_rect(screen_w: int, screen_h: int, sidebar_w: int = SIDEBAR_W) -> pygame.Rect:
    return pygame.Rect(
        PAD,
        TOPBAR_H + PAD,
        screen_w - sidebar_w - PAD * 3,
        screen_h - TOPBAR_H - PAD * 2,
    )


def board_zone(
    main: pygame.Rect,
    reserve_bottom: int = CONTROL_BAR_H + NARRATION_H + 24,
    *,
    show_model_info: bool = False,
) -> Dict[str, pygame.Rect]:
    """Allocate board, hud, model info, narration, control bar without overlap."""
    model_reserve = MODEL_INFO_H + 12 if show_model_info else 0
    usable_h = main.height - reserve_bottom - model_reserve - HUD_H - 8
    board_size = min(main.width - 40, usable_h)
    board_y = main.y + HUD_H + 8
    board = pygame.Rect(main.centerx - board_size // 2, board_y, board_size, board_size)
    model_info = pygame.Rect(main.x + 8, board.bottom + 10, main.width - 16, MODEL_INFO_H)
    narration_y = model_info.bottom + 10 if show_model_info else board.bottom + 12
    return {
        "hud": pygame.Rect(board.x, main.y, board.width, HUD_H),
        "board": board,
        "model_info": model_info,
        "narration": pygame.Rect(main.x, narration_y, main.width, NARRATION_H),
        "control_bar": pygame.Rect(main.x, main.bottom - CONTROL_BAR_H, main.width, CONTROL_BAR_H),
    }


def battle_zone(main: pygame.Rect, board_w: int, board_h: int) -> Dict[str, pygame.Rect]:
    """Two boards side-by-side inside main area."""
    reserve_bottom = CONTROL_BAR_H + NARRATION_H + 20
    header_h = 28
    usable = pygame.Rect(main.x, main.y + header_h, main.width, main.height - header_h - reserve_bottom)
    gap = 20
    col_w = (usable.width - gap) // 2
    size = min(col_w - 16, usable.height - HUD_H - 16, 380)
    left = pygame.Rect(usable.x + (col_w - size) // 2, usable.y + HUD_H + 8, size, size)
    right = pygame.Rect(usable.x + col_w + gap + (col_w - size) // 2, usable.y + HUD_H + 8, size, size)
    return {
        "header": pygame.Rect(main.x, main.y, main.width, header_h),
        "hud_a": pygame.Rect(left.x, left.y - HUD_H, left.width, HUD_H - 4),
        "hud_b": pygame.Rect(right.x, right.y - HUD_H, right.width, HUD_H - 4),
        "board_a": left,
        "board_b": right,
        "narration": pygame.Rect(main.x, usable.bottom + 8, main.width, NARRATION_H),
        "control_bar": pygame.Rect(main.x, main.bottom - CONTROL_BAR_H, main.width, CONTROL_BAR_H),
    }
