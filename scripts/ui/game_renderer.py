"""Beautiful game board rendering for Simple and Advanced modes."""

import math
from typing import Dict, List, Optional, Tuple

import pygame

from snake_rl.envs.snake_env import Action, Direction, SnakeEnv
from ui.demo_features import AGENT_ICONS
from ui.components import draw_epsilon_bar, draw_mode_pill, draw_panel, draw_text
from ui.model_info import build_rl_mode_state
from ui.theme import THEME, font

Point = Tuple[int, int]


def board_geometry(env: SnakeEnv, rect: pygame.Rect) -> Tuple[int, int, int, int, int, pygame.Rect]:
    """Return cell, ox, oy, board_w, board_h, frame."""
    cell = min(rect.width // env.width, rect.height // env.height)
    board_w = cell * env.width
    board_h = cell * env.height
    ox = rect.x + (rect.width - board_w) // 2
    oy = rect.y + (rect.height - board_h) // 2
    frame_pad = 12
    frame = pygame.Rect(ox - frame_pad, oy - frame_pad, board_w + frame_pad * 2, board_h + frame_pad * 2)
    return cell, ox, oy, board_w, board_h, frame


def draw_fill_ring(screen: pygame.Surface, frame: pygame.Rect, fill_frac: float, color: Tuple[int, int, int]) -> None:
    if fill_frac <= 0:
        return
    cx, cy = frame.centerx, frame.centery
    radius = max(frame.width, frame.height) // 2 + 6
    start = -math.pi / 2
    end = start + 2 * math.pi * min(1.0, fill_frac)
    pygame.draw.arc(screen, color, (cx - radius, cy - radius, radius * 2, radius * 2), start, end, 3)


def draw_ghost_move(screen, env, ox, oy, cell, ghost_point: Optional[Point], accent) -> None:
    if ghost_point is None:
        return
    gx, gy = ghost_point
    pad = max(3, cell // 6)
    gr = pygame.Rect(ox + gx * cell + pad, oy + gy * cell + pad, cell - 2 * pad, cell - 2 * pad)
    s = pygame.Surface((gr.width, gr.height), pygame.SRCALPHA)
    s.fill((*accent[:3], 60))
    screen.blit(s, gr.topleft)
    pygame.draw.rect(screen, accent, gr, 1, border_radius=max(4, cell // 5))


def draw_decision_overlay(screen, env, ox, oy, cell, action_analysis: Dict, chosen: Action) -> None:
    if not env.snake:
        return
    hx, hy = env.snake[0]
    cx = ox + hx * cell + cell // 2
    cy = oy + hy * cell + cell // 2
    offsets = {
        Action.STRAIGHT: (0, -cell // 2 - 8),
        Action.RIGHT: (cell // 2 + 8, 0),
        Action.LEFT: (-cell // 2 - 8, 0),
    }
    labels = {Action.STRAIGHT: "↑", Action.RIGHT: "→", Action.LEFT: "←"}
    for action in [Action.STRAIGHT, Action.RIGHT, Action.LEFT]:
        analysis = action_analysis[action]
        safe = not analysis["danger"]
        color = THEME.success if safe else THEME.danger
        if action == chosen:
            color = THEME.accent
        dx, dy = offsets[action]
        draw_text(screen, labels[action], cx + dx, cy + dy, 14, color, bold=True, center=True)


def draw_agent_badge(screen, frame: pygame.Rect, icon: str, name: str, tagline: str, accent) -> None:
    badge = pygame.Rect(frame.x, frame.y - 36, frame.width, 30)
    pygame.draw.rect(screen, THEME.panel, badge, border_radius=8)
    pygame.draw.rect(screen, accent, badge, 1, border_radius=8)
    draw_text(screen, f"{icon}  {name}", badge.x + 12, badge.centery, 13, accent, bold=True)
    tag_surf = font(10).render(tagline, True, THEME.text_muted)
    screen.blit(tag_surf, (badge.right - tag_surf.get_width() - 10, badge.centery - 5))


def draw_game_board(
    screen: pygame.Surface,
    env: SnakeEnv,
    rect: pygame.Rect,
    *,
    simple_mode: bool = True,
    paused: bool = False,
    game_over: bool = False,
    game_over_reason: str = "",
    pulse: float = 0.0,
    manual_mode: bool = False,
    shake: Tuple[int, int] = (0, 0),
    ghost_point: Optional[Point] = None,
    action_analysis: Optional[Dict] = None,
    chosen_action: Optional[Action] = None,
    show_fill_ring: bool = True,
    agent_icon: str = "",
    agent_name: str = "",
    agent_tagline: str = "",
    agent_accent: Optional[Tuple[int, int, int]] = None,
    replay_frame=None,
    effects_manager=None,
    label_mode: str = "badge",
    show_manual_controls: bool = True,
) -> pygame.Rect:
    """Render polished Snake board inside rect. Returns frame rect."""
    t = THEME
    draw_rect = rect.move(shake)
    cell, ox, oy, board_w, board_h, frame = board_geometry(env, draw_rect)
    accent = agent_accent or (t.manual if manual_mode else t.accent)

    if show_fill_ring:
        total = env.width * env.height
        fill_frac = len(env.snake) / max(total, 1)
        draw_fill_ring(screen, frame, fill_frac, t.success)

    pygame.draw.rect(screen, t.bg_elevated, frame, border_radius=16)
    border_color = t.manual if manual_mode else accent
    pygame.draw.rect(screen, border_color, frame, 2, border_radius=16)

    if agent_name and label_mode == "badge":
        draw_agent_badge(screen, frame, agent_icon or "🐍", agent_name, agent_tagline, accent)
    elif agent_name and label_mode == "corner":
        draw_text(screen, f"{agent_icon} {agent_name}", frame.x + 10, frame.y + 8, 11, accent, bold=True)

    board_rect = pygame.Rect(ox, oy, board_w, board_h)
    pygame.draw.rect(screen, t.board_bg, board_rect, border_radius=10)

    snake = replay_frame.snake if replay_frame else env.snake
    food = replay_frame.food if replay_frame else env.food
    display_score = replay_frame.score if replay_frame else env.score

    if simple_mode:
        for y in range(env.height):
            for x in range(env.width):
                if (x + y) % 2 == 0:
                    cr = pygame.Rect(ox + x * cell, oy + y * cell, cell, cell)
                    pygame.draw.rect(screen, t.board_grid, cr)
    else:
        for y in range(env.height):
            for x in range(env.width):
                cr = pygame.Rect(ox + x * cell, oy + y * cell, cell, cell)
                pygame.draw.rect(screen, t.board_grid, cr)
                pygame.draw.rect(screen, t.board_grid_line, cr, 1)

    if ghost_point and not replay_frame:
        draw_ghost_move(screen, env, ox, oy, cell, ghost_point, accent)

    if food is not None:
        fx, fy = food
        pad = max(4, cell // 5)
        food_rect = pygame.Rect(ox + fx * cell + pad, oy + fy * cell + pad, cell - 2 * pad, cell - 2 * pad)
        glow_size = int(4 + 3 * math.sin(pulse * 3))
        glow_rect = food_rect.inflate(glow_size * 2, glow_size * 2)
        pygame.draw.ellipse(screen, t.food_glow, glow_rect)
        pygame.draw.rect(screen, t.food, food_rect, border_radius=max(6, cell // 4))
        highlight = pygame.Rect(food_rect.x + 4, food_rect.y + 4, cell // 4, cell // 4)
        pygame.draw.ellipse(screen, t.food_highlight, highlight)

    for i in range(len(snake) - 1, 0, -1):
        sx, sy = snake[i]
        pad = max(2, cell // 8)
        body_rect = pygame.Rect(ox + sx * cell + pad, oy + sy * cell + pad, cell - 2 * pad, cell - 2 * pad)
        shade = t.snake_body if i % 2 == 0 else t.snake_body_dark
        pygame.draw.rect(screen, shade, body_rect, border_radius=max(5, cell // 5))

    if snake:
        hx, hy = snake[0]
        pad = max(1, cell // 10)
        head_rect = pygame.Rect(ox + hx * cell + pad, oy + hy * cell + pad, cell - 2 * pad, cell - 2 * pad)
        head_color = t.manual if manual_mode else t.snake_head
        glow_color = (200, 140, 50) if manual_mode else t.snake_head_glow
        pygame.draw.rect(screen, glow_color, head_rect.inflate(4, 4), border_radius=max(7, cell // 4))
        pygame.draw.rect(screen, head_color, head_rect, border_radius=max(7, cell // 4))
        eye_r = max(2, cell // 10)
        ex1 = head_rect.centerx - cell // 6
        ex2 = head_rect.centerx + cell // 6
        ey = head_rect.centery - cell // 8
        pygame.draw.circle(screen, (10, 20, 14), (ex1, ey), eye_r)
        pygame.draw.circle(screen, (10, 20, 14), (ex2, ey), eye_r)

    if action_analysis and chosen_action is not None and not replay_frame:
        draw_decision_overlay(screen, env, ox, oy, cell, action_analysis, chosen_action)

    if effects_manager and not replay_frame:
        effects_manager.draw_overlays(screen, rect, env, cell, ox, oy)

    if manual_mode and show_manual_controls and not paused and not game_over and not replay_frame:
        ctrl_x = min(frame.right + 12, screen.get_width() - 120)
        draw_manual_controls(screen, ctrl_x, frame.centery - 60)

    if paused:
        overlay = pygame.Surface((board_w, board_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        screen.blit(overlay, (ox, oy))
        draw_text(screen, "PAUSED", ox + board_w // 2, oy + board_h // 2, 32, t.accent, bold=True, center=True)

    if replay_frame:
        overlay = pygame.Surface((board_w, board_h), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 80))
        screen.blit(overlay, (ox, oy))
        draw_text(screen, "REPLAY", ox + board_w // 2, oy + 18, 14, t.purple, bold=True, center=True)
        draw_text(screen, f"Score {display_score}", ox + board_w // 2, oy + board_h - 18, 12, t.text_muted, center=True)

    if game_over and not replay_frame:
        overlay = pygame.Surface((board_w, board_h), pygame.SRCALPHA)
        overlay.fill((20, 0, 0, 150))
        screen.blit(overlay, (ox, oy))
        draw_text(screen, "GAME OVER", ox + board_w // 2, oy + board_h // 2 - 20, 26, t.danger, bold=True, center=True)
        if game_over_reason:
            draw_text(screen, game_over_reason.replace("_", " ").title(), ox + board_w // 2, oy + board_h // 2 + 6, 13, t.text_secondary, center=True)
        hint = "Press R to play again" if manual_mode else "Restarting…"
        draw_text(screen, hint, ox + board_w // 2, oy + board_h // 2 + 30, 12, t.text_muted, center=True)

    return frame


def draw_manual_controls(screen: pygame.Surface, x: int, y: int) -> None:
    t = THEME
    panel = pygame.Rect(x, y, 108, 120)
    pygame.draw.rect(screen, t.panel, panel, border_radius=10)
    pygame.draw.rect(screen, t.manual, panel, 1, border_radius=10)
    draw_text(screen, "Controls", panel.centerx, panel.y + 14, 11, t.manual, bold=True, center=True)
    key_size = 28
    cx = panel.centerx
    keys = [
        (cx - key_size // 2, panel.y + 30, "↑"),
        (cx - key_size - 4, panel.y + 62, "←"),
        (cx - key_size // 2, panel.y + 62, "↓"),
        (cx + 4, panel.y + 62, "→"),
    ]
    for kx, ky, label in keys:
        kr = pygame.Rect(kx, ky, key_size, key_size)
        pygame.draw.rect(screen, t.bg_elevated, kr, border_radius=6)
        pygame.draw.rect(screen, t.border_soft, kr, 1, border_radius=6)
        draw_text(screen, label, kr.centerx, kr.centery, 14, t.text_secondary, center=True)
    draw_text(screen, "or WASD", panel.centerx, panel.bottom - 12, 10, t.text_muted, center=True)


def draw_score_hud(
    screen: pygame.Surface,
    rect: pygame.Rect,
    score: int,
    best: int,
    episode: int,
    steps: int,
    *,
    manual_mode: bool = False,
    fill_pct: float = 0.0,
) -> None:
    t = THEME
    hud = pygame.Rect(rect.x, rect.y, rect.width, 56)
    pygame.draw.rect(screen, t.bg_elevated, hud, border_radius=12)
    accent = t.manual if manual_mode else t.border_soft
    pygame.draw.rect(screen, accent, hud, 1, border_radius=12)
    draw_text(screen, "Score", rect.x + 20, rect.y + 10, 10, t.text_muted, bold=True)
    draw_text(screen, str(score), rect.x + 20, rect.y + 24, 28, t.success, bold=True)
    draw_text(screen, "Best", rect.x + 120, rect.y + 10, 10, t.text_muted, bold=True)
    draw_text(screen, str(best), rect.x + 120, rect.y + 24, 22, t.accent, bold=True)
    right_label = "Your game" if manual_mode else f"Game #{episode}"
    draw_text(screen, right_label, rect.right - 130, rect.y + 14, 12, t.text_secondary)
    draw_text(screen, f"{steps} steps · {fill_pct:.0f}% fill", rect.right - 130, rect.y + 32, 11, t.text_muted)


def draw_narration_bar(screen: pygame.Surface, rect: pygame.Rect, text: str) -> None:
    pygame.draw.rect(screen, THEME.panel, rect, border_radius=10)
    pygame.draw.rect(screen, THEME.accent_soft, rect, 1, border_radius=10)
    draw_text(screen, text, rect.x + 16, rect.centery - 7, 14, THEME.text_secondary)


def draw_rl_learning_mode_strip(
    screen: pygame.Surface,
    rect: pygame.Rect,
    agent,
    learning_mode: str,
    *,
    accent: Optional[Tuple[int, int, int]] = None,
    compact: bool = False,
) -> None:
    """Prominent training/autonomous indicator with epsilon visualization."""
    t = THEME
    mode = build_rl_mode_state(agent, learning_mode)
    mode_color = t.warning if mode["is_training"] else t.success

    pygame.draw.rect(screen, t.panel, rect, border_radius=10)
    pygame.draw.rect(screen, mode_color, rect, 2, border_radius=10)

    pill_w = 108 if compact else 120
    pill = pygame.Rect(rect.x + 10, rect.y + (10 if compact else 12), pill_w, 24)
    draw_mode_pill(screen, pill, mode["mode_label"], active=mode["is_training"], color=mode_color)

    text_x = pill.right + 12
    draw_text(screen, mode["subtitle"], text_x, rect.y + (10 if compact else 12), 12, t.text, bold=True)
    if not compact:
        draw_text(screen, mode["detail"], text_x, rect.y + 30, 10, t.text_muted)

    bar_rect = pygame.Rect(
        rect.right - (140 if compact else 180) - 10,
        rect.y + (14 if compact else 38),
        140 if compact else 180,
        12 if compact else 14,
    )
    bar_label = "Exploration" if mode["is_training"] else "Exploration (off)"
    bar_color = mode_color if mode["is_training"] else t.border_soft
    draw_epsilon_bar(screen, bar_rect, mode["epsilon"], fill_color=bar_color, label=bar_label)
    if compact:
        draw_text(screen, f"ε={mode['epsilon_display']}", bar_rect.x, bar_rect.bottom + 4, 9, t.text_muted)


def draw_model_results_panel(
    screen: pygame.Surface,
    rect: pygame.Rect,
    results: dict,
    *,
    accent: Optional[Tuple[int, int, int]] = None,
    agent=None,
    learning_mode: str = "training",
) -> None:
    """Show loaded model file, training benchmarks, and live session stats."""
    accent = accent or THEME.accent
    draw_panel(screen, rect, f"{results['agent_name']} — Model & Results", accent=accent)

    if agent is not None:
        mode_rect = pygame.Rect(rect.x + 10, rect.y + 30, rect.width - 20, 52)
        draw_rl_learning_mode_strip(screen, mode_rect, agent, learning_mode, accent=accent)

    model_y = rect.y + (88 if agent is not None else 32)
    model_line = f"Model: {results['model_file']}"
    if results["training_steps"] != "—":
        model_line += f"  ·  {results['training_steps']} steps"
    draw_text(screen, model_line, rect.x + 14, model_y, 11, THEME.text_secondary)

    cols = [
        ("Trained best", results["trained_best"], THEME.success),
        ("Trained avg", results["trained_avg"], THEME.accent),
        (f"Max ({results['board_label']})", results["max_on_board"], THEME.warning),
        ("Session best", results["session_best"], THEME.success),
        ("Session avg", results["session_avg"], THEME.purple),
    ]
    gap = 8
    per_row = 3 if rect.width < 520 else 5
    col_w = (rect.width - 28 - gap * (per_row - 1)) // per_row
    x = rect.x + 14
    y = model_y + 20
    for i, (label, value, color) in enumerate(cols):
        if i > 0 and i % per_row == 0:
            x = rect.x + 14
            y += 34
        pygame.draw.rect(screen, THEME.bg_elevated, pygame.Rect(x, y, col_w, 30), border_radius=8)
        draw_text(screen, label, x + 8, y + 4, 9, THEME.text_muted, bold=True)
        draw_text(screen, value, x + 8, y + 16, 12, color, bold=True)
        x += col_w + gap
