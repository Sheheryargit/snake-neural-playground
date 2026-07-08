"""Simple and Advanced dashboard layouts with fixed zones (no overlap)."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygame

from snake_rl.core.metrics import AgentMetrics
from snake_rl.envs.snake_env import Action, SnakeEnv
from ui.agent_data import AGENT_PROFILES, COMPARISON_LABELS, COMPARISON_ROWS, MANUAL_AGENT_KEY, agent_key_from_pygame
from ui.components import (
    RectButton,
    draw_bar_chart,
    draw_divider,
    draw_line_chart,
    draw_mode_pill,
    draw_panel,
    draw_section_header,
    draw_stat_card,
    draw_tab_bar,
    draw_text,
    draw_wrapped_text,
)
from ui.board_config import MAX_BOARD_SIZE, MIN_BOARD_SIZE, format_board_label
from ui.demo_features import AGENT_ICONS, AGENT_TOOLTIPS, BATTLE_AGENT_OPTIONS, DEMO_PRESETS, DemoSession, SessionLeaderboard
from ui.game_renderer import draw_game_board, draw_model_results_panel, draw_narration_bar, draw_rl_learning_mode_strip, draw_score_hud
from ui.model_info import build_model_results, build_rl_mode_state, is_rl_model_agent
from ui.layout import (
    AGENT_CHIP_H,
    AGENT_CHIP_ROW_H,
    AGENT_GRID_COLS,
    BATTLE_SIDEBAR_W,
    SIDEBAR_W,
    agent_grid_height,
    battle_sidebar_layout,
    battle_zone,
    board_zone,
    main_content_rect,
    sidebar_layout,
)
from ui.theme import THEME, font

ADVANCED_TABS = ["Overview", "Brain", "Compare", "State", "Leaderboard"]
AI_AGENT_KEYS = [1, 2, 3, 4, 5, 6]
STATE_LABELS = [
    "Danger straight", "Danger right", "Danger left",
    "Moving left", "Moving right", "Moving up", "Moving down",
    "Food left", "Food right", "Food up", "Food down",
]


class DashboardUI:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.ui_mode = "simple"
        self.advanced_tab = 0
        self.agent_buttons: List[RectButton] = []
        self.battle_buttons_a: List[RectButton] = []
        self.battle_buttons_b: List[RectButton] = []
        self.manual_button: Optional[RectButton] = None
        self.mode_button: Optional[RectButton] = None
        self.tab_buttons: List[RectButton] = []
        self.speed_buttons: List[RectButton] = []
        self.view_buttons: Dict[str, RectButton] = {}
        self.board_size_buttons: Dict[str, RectButton] = {}
        self.preset_buttons: List[Tuple[RectButton, str]] = []
        self.control_buttons: Dict[str, RectButton] = {}
        self.neural_sidebar_button: Optional[RectButton] = None
        self.brain_neural_button: Optional[RectButton] = None
        self.net_topbar_button: Optional[RectButton] = None
        self.hovered_tooltip = ""

    # ── Builders ─────────────────────────────────────────────────────────────

    def build_manual_button(self, rect: pygame.Rect, selected_key: int) -> None:
        self.manual_button = RectButton(rect, "Play Yourself", key=MANUAL_AGENT_KEY, subtitle="Arrow / WASD")
        self.manual_button.active = selected_key == MANUAL_AGENT_KEY

    def build_agent_grid(self, buttons: List[RectButton], rect: pygame.Rect, selected_key: int, keys: List[int]) -> None:
        buttons.clear()
        chip_w = (rect.width - 8) // AGENT_GRID_COLS
        for i, pygame_key in enumerate(keys):
            num = agent_key_from_pygame(pygame_key) or (pygame_key - 48)
            profile = AGENT_PROFILES.get(num)
            if not profile:
                continue
            col = i % AGENT_GRID_COLS
            row = i // AGENT_GRID_COLS
            r = pygame.Rect(
                rect.x + col * (chip_w + 8),
                rect.y + row * AGENT_CHIP_ROW_H,
                chip_w,
                AGENT_CHIP_H,
            )
            btn = RectButton(r, f"{AGENT_ICONS[num]} {profile.short_name}", key=pygame_key)
            btn.active = pygame_key == selected_key
            buttons.append(btn)

    def build_battle_pickers(self, sl, battle_key_a: int, battle_key_b: int) -> None:
        grid_h = agent_grid_height()
        rect_a = pygame.Rect(sl.x, sl.presets_y + 18, sl.width, grid_h)
        rect_b = pygame.Rect(sl.x, sl.agents_y + 18, sl.width, grid_h)
        keys = [k for k, _ in BATTLE_AGENT_OPTIONS]
        self.build_agent_grid(self.battle_buttons_a, rect_a, battle_key_a, keys)
        self.build_agent_grid(self.battle_buttons_b, rect_b, battle_key_b, keys)

    def build_preset_buttons(self, rect: pygame.Rect) -> None:
        self.preset_buttons = []
        gap = 6
        w = (rect.width - gap * 3) // 4
        x = rect.x
        for preset in DEMO_PRESETS.values():
            self.preset_buttons.append((RectButton(pygame.Rect(x, rect.y, w, 28), preset.label), preset.name))
            x += w + gap

    def build_board_size_buttons(self, rect: pygame.Rect) -> None:
        self.board_size_buttons = {
            "w_minus": RectButton(pygame.Rect(rect.x + 72, rect.y + 20, 26, 26), "−"),
            "w_plus": RectButton(pygame.Rect(rect.x + 130, rect.y + 20, 26, 26), "+"),
            "h_minus": RectButton(pygame.Rect(rect.x + 72, rect.y + 54, 26, 26), "−"),
            "h_plus": RectButton(pygame.Rect(rect.x + 130, rect.y + 54, 26, 26), "+"),
        }

    def build_speed_buttons(self, rect: pygame.Rect, fps: int) -> None:
        self.speed_buttons = []
        gap = 6
        w = (rect.width - gap * 3) // 4
        x = rect.x
        for label, speed in [("1×", 1), ("8×", 8), ("50×", 50), ("MAX", 2000)]:
            btn = RectButton(pygame.Rect(x, rect.y, w, 28), label)
            btn.active = fps == speed
            self.speed_buttons.append((btn, speed))
            x += w + gap

    def build_view_buttons(self, rect: pygame.Rect, demo: DemoSession) -> None:
        gap = 6
        w = (rect.width - gap * 2) // 3
        self.view_buttons = {
            "ghost": RectButton(pygame.Rect(rect.x, rect.y, w, 28), "Ghost"),
            "overlay": RectButton(pygame.Rect(rect.x + w + gap, rect.y, w, 28), "Arrows"),
            "sound": RectButton(pygame.Rect(rect.x + 2 * (w + gap), rect.y, w, 28), "Sound"),
        }
        self.view_buttons["ghost"].active = demo.ghost_enabled
        self.view_buttons["overlay"].active = demo.decision_overlay
        self.view_buttons["sound"].active = demo.sound_enabled

    def build_control_bar(self, rect: pygame.Rect, paused: bool, battle_active: bool = False, neural_live: bool = False) -> None:
        labels = [
            ("pause", "Pause" if not paused else "Play"),
            ("reset", "Reset"),
            ("replay", "Replay"),
            ("demo", "Demo"),
            ("battle", "Battle"),
            ("neural", "Neural Net"),
            ("export", "PNG"),
        ]
        self.control_buttons = {}
        gap = 6
        w = (rect.width - gap * (len(labels) - 1)) // len(labels)
        x = rect.x
        for key, label in labels:
            btn = RectButton(pygame.Rect(x, rect.y, w, rect.height), label)
            if key == "battle" and battle_active:
                btn.active = True
            if key == "neural" and neural_live:
                btn.active = True
            self.control_buttons[key] = btn
            x += w + gap

    def build_topbar_buttons(self, ui_mode: str) -> None:
        self.mode_button = RectButton(pygame.Rect(self.width - 168, 14, 148, 28), "Advanced" if ui_mode == "simple" else "Simple")

    def _hover_tooltips(self, pos: Tuple[int, int], buttons: List[RectButton]) -> None:
        for btn in buttons:
            if btn.contains(pos) and btn.key:
                num = agent_key_from_pygame(btn.key)
                if num:
                    self.hovered_tooltip = AGENT_TOOLTIPS.get(num, "")
                    btn.hovered = True
                    return
            btn.hovered = False

    def handle_click(self, pos: Tuple[int, int], selected_key: int, battle_mode: bool = False) -> Optional[Dict[str, Any]]:
        self.hovered_tooltip = ""
        if self.mode_button and self.mode_button.contains(pos):
            return {"action": "toggle_mode"}
        if self.net_topbar_button and self.net_topbar_button.contains(pos):
            return {"action": "neural"}
        if self.brain_neural_button and self.brain_neural_button.contains(pos):
            return {"action": "neural"}
        if self.neural_sidebar_button and self.neural_sidebar_button.contains(pos):
            return {"action": "neural"}
        for key, btn in self.control_buttons.items():
            if btn.contains(pos):
                return {"action": key}
        for key, btn in self.view_buttons.items():
            if btn.contains(pos):
                return {"action": f"toggle_{key}"}
        if battle_mode:
            for btn in self.battle_buttons_a:
                if btn.contains(pos) and btn.key:
                    return {"action": "select_battle_a", "key": btn.key}
            for btn in self.battle_buttons_b:
                if btn.contains(pos) and btn.key:
                    return {"action": "select_battle_b", "key": btn.key}
        if self.manual_button and self.manual_button.contains(pos):
            return {"action": "select_agent", "key": MANUAL_AGENT_KEY}
        for btn in self.agent_buttons:
            if btn.contains(pos) and btn.key:
                return {"action": "select_agent", "key": btn.key}
        for btn, preset_name in self.preset_buttons:
            if btn.contains(pos):
                return {"action": "demo_preset", "preset": preset_name}
        for btn in self.tab_buttons:
            if btn.contains(pos):
                return {"action": "select_tab", "tab": self.tab_buttons.index(btn)}
        for btn, speed in self.speed_buttons:
            if btn.contains(pos):
                return {"action": "set_fps", "fps": speed}
        for k, delta in [("w_minus", (-1, 0)), ("w_plus", (1, 0)), ("h_minus", (0, -1)), ("h_plus", (0, 1))]:
            if self.board_size_buttons.get(k) and self.board_size_buttons[k].contains(pos):
                return {"action": "resize_board", "dw": delta[0], "dh": delta[1]}
        return None

    def handle_motion(self, pos: Tuple[int, int]) -> None:
        self.hovered_tooltip = ""
        self._hover_tooltips(pos, self.agent_buttons)
        if not self.hovered_tooltip:
            self._hover_tooltips(pos, self.battle_buttons_a + self.battle_buttons_b)
        if self.manual_button and self.manual_button.contains(pos):
            self.hovered_tooltip = AGENT_TOOLTIPS[7]

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def draw_topbar(self, screen, *, ui_mode, fps, paused, agent_name, manual_mode=False, board_width=12, board_height=12, demo: Optional[DemoSession] = None, agent=None, learning_mode: str = "training", agent_num: int = 1):
        t = THEME
        pygame.draw.rect(screen, t.bg_elevated, pygame.Rect(0, 0, self.width, 56))
        pygame.draw.line(screen, t.border_soft, (0, 56), (self.width, 56))
        draw_text(screen, "Snake RL Arena", 24, 18, 18, t.text, bold=True)
        x = 200
        if demo and demo.presentation_mode:
            draw_text(screen, "PRESENTATION", x, 20, 11, t.warning, bold=True)
            x += 130
        elif demo and demo.battle_mode:
            num_a = agent_key_from_pygame(demo.battle_key_a) or 1
            num_b = agent_key_from_pygame(demo.battle_key_b) or 2
            battle_label = f"{AGENT_PROFILES[num_a].short_name} vs {AGENT_PROFILES[num_b].short_name}"
            draw_text(screen, "BATTLE", x, 20, 11, t.purple, bold=True)
            x += 80
            draw_text(screen, battle_label, x, 20, 12, t.text_secondary)
            x += min(220, len(battle_label) * 7)
        else:
            lbl = "Simple" if ui_mode == "simple" else "Advanced"
            draw_text(screen, lbl, x, 20, 11, t.success if ui_mode == "simple" else t.purple, bold=True)
            x += 90
        status = "YOUR TURN" if manual_mode else ("Paused" if paused else "Live")
        sc = t.manual if manual_mode else (t.warning if paused else t.success)
        pygame.draw.circle(screen, sc, (x + 6, 28), 4)
        draw_text(screen, status, x + 16, 20, 11, sc, bold=True)
        x += 72
        if agent is not None and is_rl_model_agent(agent_num):
            mode = build_rl_mode_state(agent, learning_mode)
            mode_color = t.warning if mode["is_training"] else t.success
            pill = pygame.Rect(x, 16, 108, 24)
            draw_mode_pill(screen, pill, mode["mode_label"], active=mode["is_training"], color=mode_color)
            draw_text(screen, f"ε={mode['epsilon_display']}", pill.right + 10, 20, 11, mode_color, bold=True)
            self.net_topbar_button = RectButton(pygame.Rect(pill.right + 52, 16, 44, 24), "Net")
            self.net_topbar_button.active = bool(demo and demo.nn_live)
            self.net_topbar_button.draw(screen, THEME.purple, compact=True)
            x = pill.right + 104
        else:
            self.net_topbar_button = None
        if not (demo and demo.battle_mode):
            draw_text(screen, agent_name, x, 20, 12, t.text_secondary)
        draw_text(screen, format_board_label(board_width, board_height), self.width - 320, 20, 11, t.text_muted)
        draw_text(screen, f"{fps} fps", self.width - 200, 20, 11, t.text_muted)
        if self.mode_button:
            self.mode_button.draw(screen, t.accent, compact=True)

    def _board_kwargs(self, env, episode, profile, agent_num, manual_mode, demo, action_analysis, replay_frame=None, compact=False):
        accent = THEME.manual if manual_mode else THEME.agent_colors.get(agent_num, THEME.accent)
        ghost = None
        if demo.ghost_enabled and action_analysis and episode.get("last_action") is not None:
            ghost = action_analysis[episode["last_action"]]["next_point"]
        return dict(
            simple_mode=True,
            paused=False,
            game_over=episode["done"] and not replay_frame,
            game_over_reason=episode.get("reason", ""),
            manual_mode=manual_mode,
            shake=demo.effects.shake_offset(),
            ghost_point=ghost,
            action_analysis=action_analysis if demo.decision_overlay else None,
            chosen_action=episode.get("last_action"),
            agent_icon=AGENT_ICONS.get(agent_num, "🐍"),
            agent_name=profile.short_name if compact else profile.full_name,
            agent_tagline="" if compact else profile.tagline,
            agent_accent=accent,
            replay_frame=replay_frame,
            effects_manager=demo.effects,
            label_mode="corner" if compact else "badge",
            show_manual_controls=not compact and manual_mode,
        )

    def _draw_sidebar_normal(self, screen, sl, selected_key, env, metrics, profile, agent_num, manual_mode, fps, is_learning, learning_mode, demo, agent=None):
        accent = THEME.manual if manual_mode else THEME.agent_colors.get(agent_num, THEME.accent)
        self.build_manual_button(pygame.Rect(sl.x, sl.play_y, sl.width, 44), selected_key)
        if self.manual_button:
            self.manual_button.draw(screen, THEME.manual)
        draw_section_header(screen, sl.x, sl.presets_y, "Demo Presets")
        self.build_preset_buttons(pygame.Rect(sl.x, sl.presets_y + 18, sl.width, 28))
        for btn, _ in self.preset_buttons:
            btn.draw(screen, THEME.warning, compact=True)
        draw_section_header(screen, sl.x, sl.agents_y, "AI Agents")
        self.build_agent_grid(
            self.agent_buttons,
            pygame.Rect(sl.x, sl.agents_y + 18, sl.width, agent_grid_height(6)),
            selected_key,
            [k + 48 for k in AI_AGENT_KEYS],
        )
        for btn in self.agent_buttons:
            n = agent_key_from_pygame(btn.key) if btn.key else 0
            btn.draw(screen, THEME.agent_colors.get(n, THEME.accent), compact=True)
        info_h = 168 if is_learning and agent is not None else 96
        draw_panel(screen, pygame.Rect(sl.x, sl.info_y, sl.width, info_h), "Now Playing")
        draw_text(screen, profile.full_name, sl.x + 12, sl.info_y + 30, 14, accent, bold=True)
        draw_text(screen, profile.tagline, sl.x + 12, sl.info_y + 48, 10, THEME.text_muted)
        if is_learning and agent is not None:
            mode = build_rl_mode_state(agent, learning_mode)
            mode_color = THEME.warning if mode["is_training"] else THEME.success
            pill = pygame.Rect(sl.x + 12, sl.info_y + 64, 100, 22)
            draw_mode_pill(screen, pill, mode["mode_label"], active=mode["is_training"], color=mode_color)
            draw_text(screen, f"ε={mode['epsilon_display']} ({mode['epsilon_pct']}%)", pill.right + 8, sl.info_y + 68, 10, mode_color, bold=True)
            draw_text(screen, "Press M to toggle mode", sl.x + 12, sl.info_y + 88, 9, THEME.text_muted)
            stat_y = sl.info_y + 96
        else:
            self.neural_sidebar_button = None
            stat_y = sl.info_y + 68
        cw = (sl.width - 8) // 2
        draw_stat_card(screen, pygame.Rect(sl.x, stat_y, cw, 32), "Avg", f"{metrics.average_score():.1f}", THEME.accent)
        draw_stat_card(screen, pygame.Rect(sl.x + cw + 8, stat_y, cw, 32), "Best", str(metrics.best_score()), THEME.success)
        if is_learning and agent is not None:
            self.neural_sidebar_button = RectButton(pygame.Rect(sl.x + 12, stat_y + 38, sl.width - 24, 26), "Open Neural Net (N)")
            self.neural_sidebar_button.active = demo.nn_live
            self.neural_sidebar_button.draw(screen, THEME.purple, compact=True)
        draw_section_header(screen, sl.x, sl.board_y, "Board Size")
        panel = pygame.Rect(sl.x, sl.board_y + 16, sl.width, 82)
        pygame.draw.rect(screen, THEME.panel, panel, border_radius=8)
        draw_text(screen, f"Width: {env.width}", sl.x + 12, sl.board_y + 26, 11, THEME.text)
        draw_text(screen, f"Height: {env.height}", sl.x + 12, sl.board_y + 58, 11, THEME.text)
        self.build_board_size_buttons(pygame.Rect(sl.x, sl.board_y + 16, sl.width, 82))
        for btn in self.board_size_buttons.values():
            btn.draw(screen, THEME.accent, compact=True)
        draw_section_header(screen, sl.x, sl.view_y, "View")
        self.build_view_buttons(pygame.Rect(sl.x, sl.view_y + 18, sl.width, 28), demo)
        for btn in self.view_buttons.values():
            btn.draw(screen, THEME.accent, compact=True)
        draw_section_header(screen, sl.x, sl.speed_y, "Speed")
        self.build_speed_buttons(pygame.Rect(sl.x, sl.speed_y + 18, sl.width, 28), fps)
        for btn, _ in self.speed_buttons:
            btn.draw(screen, THEME.accent, compact=True)
        if self.hovered_tooltip:
            tip = pygame.Rect(sl.x, sl.tooltip_y, sl.width, 24)
            pygame.draw.rect(screen, THEME.panel_active, tip, border_radius=6)
            draw_text(screen, self.hovered_tooltip, sl.x + 8, sl.tooltip_y + 5, 10, THEME.text_secondary)

    def _draw_sidebar_battle(self, screen, sl, demo, fps, env):
        sidebar_panel = pygame.Rect(sl.x - 8, sl.play_y - 8, sl.width + 16, sl.speed_y + 56 - sl.play_y)
        pygame.draw.rect(screen, THEME.bg_elevated, sidebar_panel, border_radius=12)
        pygame.draw.rect(screen, THEME.border_soft, sidebar_panel, 1, border_radius=12)

        draw_panel(screen, pygame.Rect(sl.x, sl.play_y, sl.width, 44), "Agent Battle")
        draw_text(screen, "Pick Fighter A and Fighter B", sl.x + 12, sl.play_y + 24, 10, THEME.text_muted)
        draw_section_header(screen, sl.x, sl.presets_y, "Fighter A (left)")
        self.build_battle_pickers(sl, demo.battle_key_a, demo.battle_key_b)
        for btn in self.battle_buttons_a:
            n = agent_key_from_pygame(btn.key) if btn.key else 0
            btn.draw(screen, THEME.agent_colors.get(n, THEME.accent), compact=True)
        draw_section_header(screen, sl.x, sl.agents_y, "Fighter B (right)")
        for btn in self.battle_buttons_b:
            n = agent_key_from_pygame(btn.key) if btn.key else 0
            btn.draw(screen, THEME.agent_colors.get(n, THEME.accent), compact=True)
        num_a = agent_key_from_pygame(demo.battle_key_a) or 1
        num_b = agent_key_from_pygame(demo.battle_key_b) or 2
        draw_text(screen, f"{AGENT_PROFILES[num_a].short_name}  vs  {AGENT_PROFILES[num_b].short_name}", sl.x + 8, sl.board_y, 12, THEME.accent, bold=True)
        draw_text(screen, "Keys 1–6/H = left · Shift+key = right", sl.x + 8, sl.board_y + 20, 10, THEME.text_muted)
        draw_section_header(screen, sl.x, sl.view_y, "Speed")
        self.build_speed_buttons(pygame.Rect(sl.x, sl.view_y + 18, sl.width, 28), fps)
        for btn, _ in self.speed_buttons:
            btn.draw(screen, THEME.accent, compact=True)
        draw_section_header(screen, sl.x, sl.speed_y, "Board")
        draw_text(screen, f"{env.width} × {env.height}", sl.x + 8, sl.speed_y + 20, 12, THEME.text_secondary)
        if self.hovered_tooltip:
            tip = pygame.Rect(sl.x, sl.tooltip_y, sl.width, 24)
            pygame.draw.rect(screen, THEME.panel_active, tip, border_radius=6)
            draw_text(screen, self.hovered_tooltip, sl.x + 8, sl.tooltip_y + 5, 10, THEME.text_secondary)

    def draw_simple(self, screen, *, env, agent, metrics, episode, selected_key, fps, paused, pulse, learning_mode, is_learning, board_warning="", demo: DemoSession, action_analysis=None):
        replay_frame = demo.replay.current_frame() if demo.replay.playing else None
        fill_pct = 100 * len(env.snake) / max(env.width * env.height, 1)
        sidebar_w = BATTLE_SIDEBAR_W if demo.battle_mode else (0 if demo.presentation_mode else SIDEBAR_W)
        main = main_content_rect(self.width, self.height, sidebar_w)

        if demo.battle_mode:
            zones = battle_zone(main, env.width, env.height)
            key_a, key_b = demo.battle_key_a, demo.battle_key_b
            num_a = agent_key_from_pygame(key_a) or 1
            num_b = agent_key_from_pygame(key_b) or 2
            prof_a, prof_b = AGENT_PROFILES[num_a], AGENT_PROFILES[num_b]
            env_b, ep_b, met_b = demo.battle_env, demo.battle_episode, demo.battle_metrics
            draw_text(screen, "Agent Battle", zones["header"].centerx, zones["header"].y + 4, 14, THEME.text, bold=True, center=True)
            draw_text(screen, prof_a.short_name, zones["hud_a"].x + 8, zones["hud_a"].y + 4, 11, THEME.agent_colors[num_a], bold=True)
            draw_text(screen, f"Score {env.score}  ·  Best {metrics.best_score()}", zones["hud_a"].x + 8, zones["hud_a"].y + 20, 10, THEME.text_muted)
            draw_text(screen, prof_b.short_name, zones["hud_b"].x + 8, zones["hud_b"].y + 4, 11, THEME.agent_colors[num_b], bold=True)
            draw_text(screen, f"Score {env_b.score}  ·  Best {met_b.best_score()}", zones["hud_b"].x + 8, zones["hud_b"].y + 20, 10, THEME.text_muted)
            draw_game_board(screen, env, zones["board_a"], pulse=pulse, **self._board_kwargs(env, episode, prof_a, num_a, False, demo, action_analysis, replay_frame, compact=True))
            draw_game_board(screen, env_b, zones["board_b"], pulse=pulse, **self._board_kwargs(env_b, ep_b, prof_b, num_b, False, demo, None, None, compact=True))
            if demo.narration:
                draw_narration_bar(screen, zones["narration"], demo.narration)
            self.build_control_bar(zones["control_bar"], paused, battle_active=True, neural_live=demo.nn_live)
            for btn in self.control_buttons.values():
                btn.draw(screen, THEME.accent, compact=True)
            self._draw_sidebar_battle(screen, battle_sidebar_layout(self.width, self.height), demo, fps, env)
            return

        agent_num = agent_key_from_pygame(selected_key) or 1
        profile = AGENT_PROFILES[agent_num]
        manual_mode = selected_key == MANUAL_AGENT_KEY
        show_model_info = is_rl_model_agent(agent_num)
        zones = board_zone(main, show_model_info=show_model_info)
        draw_score_hud(screen, zones["hud"], env.score, metrics.best_score(), metrics.episode_number, env.steps, manual_mode=manual_mode, fill_pct=fill_pct)
        draw_game_board(screen, env, zones["board"], pulse=pulse, **self._board_kwargs(env, episode, profile, agent_num, manual_mode, demo, action_analysis, replay_frame))
        if show_model_info:
            results = build_model_results(agent, profile, metrics, env, learning_mode)
            draw_model_results_panel(
                screen,
                zones["model_info"],
                results,
                accent=THEME.agent_colors.get(agent_num, THEME.accent),
                agent=agent,
                learning_mode=learning_mode,
            )
        if demo.narration:
            draw_narration_bar(screen, zones["narration"], demo.narration)
        self.build_control_bar(zones["control_bar"], paused, neural_live=demo.nn_live)
        for btn in self.control_buttons.values():
            btn.draw(screen, THEME.accent, compact=True)
        if not demo.presentation_mode:
            self._draw_sidebar_normal(screen, sidebar_layout(self.width, self.height), selected_key, env, metrics, profile, agent_num, manual_mode, fps, is_learning, learning_mode, demo, agent=agent)

    def draw_advanced(self, screen, *, env, agent, metrics, episode, selected_key, fps, paused, pulse, learning_mode, all_metrics, state, action_analysis, board_warning="", demo: Optional[DemoSession] = None):
        size_factor = min(1.0, 12 / max(env.width, env.height))
        board_px = int(380 * (0.9 + 0.1 * size_factor))
        board_rect = pygame.Rect(24, 72, board_px, board_px)
        panel = pygame.Rect(board_rect.right + 20, 64, self.width - board_rect.right - 44, self.height - 80)
        agent_num = agent_key_from_pygame(selected_key) or 1
        profile = AGENT_PROFILES[agent_num]
        manual_mode = selected_key == MANUAL_AGENT_KEY
        replay_frame = demo.replay.current_frame() if demo and demo.replay.playing else None
        draw_game_board(screen, env, board_rect, simple_mode=False, paused=paused, game_over=episode["done"] and not replay_frame,
                        game_over_reason=episode.get("reason", ""), pulse=pulse, manual_mode=manual_mode,
                        shake=demo.effects.shake_offset() if demo else (0, 0), replay_frame=replay_frame,
                        effects_manager=demo.effects if demo else None, label_mode="corner",
                        agent_icon=AGENT_ICONS.get(agent_num, "🐍"), agent_name=profile.short_name,
                        agent_accent=THEME.agent_colors.get(agent_num, THEME.accent))
        by = board_rect.bottom + 12
        for i, (lbl, val, col) in enumerate([("Score", env.score, THEME.success), ("Best", metrics.best_score(), THEME.accent), ("Games", metrics.episode_number, THEME.purple), ("FPS", fps, THEME.warning)]):
            draw_stat_card(screen, pygame.Rect(24 + i * 108, by, 100, 44), lbl, str(val), col)
        if is_rl_model_agent(agent_num):
            model_h = 168 if board_px < 520 else 148
            model_rect = pygame.Rect(24, by + 56, board_px, model_h)
            results = build_model_results(agent, profile, metrics, env, learning_mode)
            draw_model_results_panel(
                screen,
                model_rect,
                results,
                accent=THEME.agent_colors.get(agent_num, THEME.accent),
                agent=agent,
                learning_mode=learning_mode,
            )
        self.tab_buttons = draw_tab_bar(screen, panel.x + 8, panel.y + 8, ADVANCED_TABS, self.advanced_tab, tab_width=100)
        content = pygame.Rect(panel.x + 8, panel.y + 48, panel.width - 16, panel.height - 56)
        tab = ADVANCED_TABS[self.advanced_tab]
        if tab == "Overview":
            self._draw_overview_tab(screen, content, metrics, profile)
        elif tab == "Brain":
            self._draw_brain_tab(screen, content, agent, episode, learning_mode, profile, demo)
        elif tab == "Compare":
            self._draw_compare_tab(screen, content)
        elif tab == "State":
            self._draw_state_tab(screen, content, state, action_analysis, episode)
        elif tab == "Leaderboard" and demo:
            self._draw_leaderboard_tab(screen, content, demo.leaderboard)

    def _draw_leaderboard_tab(self, screen, rect, leaderboard: SessionLeaderboard):
        draw_panel(screen, rect, "Session Leaderboard")
        y = rect.y + 36
        for i, entry in enumerate(leaderboard.entries[:14], 1):
            draw_text(screen, f"#{i}  {entry.agent_name}", rect.x + 14, y, 12, THEME.text)
            draw_text(screen, str(entry.score), rect.x + 180, y, 12, THEME.success, bold=True)
            draw_text(screen, entry.board, rect.x + 240, y, 11, THEME.text_muted)
            y += 18

    def _draw_overview_tab(self, screen, rect, metrics, profile):
        half_w = (rect.width - 12) // 2
        chart_h = min(140, (rect.height - 100) // 2)
        draw_line_chart(screen, pygame.Rect(rect.x, rect.y, half_w, chart_h), metrics.scores(), "Scores", THEME.success)
        draw_line_chart(screen, pygame.Rect(rect.x + half_w + 12, rect.y, half_w, chart_h), metrics.rewards(), "Rewards", THEME.purple)
        y2 = rect.y + chart_h + 12
        draw_panel(screen, pygame.Rect(rect.x, y2, rect.width, rect.height - chart_h - 12), "Summary")
        draw_wrapped_text(screen, profile.implementation, rect.x + 14, y2 + 32, rect.width - 28, 12)

    def _draw_brain_tab(self, screen, rect, agent, episode, learning_mode, profile, demo: Optional[DemoSession] = None):
        draw_panel(screen, rect, f"{profile.full_name}")
        y = rect.y + 36
        if is_rl_model_agent(profile.key):
            self.brain_neural_button = RectButton(pygame.Rect(rect.x + 12, y, 180, 28), "Open Neural Net Live")
            self.brain_neural_button.active = bool(demo and demo.nn_live)
            self.brain_neural_button.draw(screen, THEME.purple, compact=True)
            y += 36
            strip = pygame.Rect(rect.x + 10, y, rect.width - 20, 56)
            draw_rl_learning_mode_strip(screen, strip, agent, learning_mode, accent=THEME.agent_colors.get(profile.key, THEME.accent))
            y += 64
        else:
            self.brain_neural_button = None
        draw_wrapped_text(screen, profile.implementation, rect.x + 14, y, rect.width - 28, 12)
        y += 48
        if hasattr(agent, "get_q_values"):
            draw_text(screen, "Current Q-values", rect.x + 14, y, 12, THEME.accent, bold=True)
            y += 20
            for i, action in enumerate([Action.STRAIGHT, Action.RIGHT, Action.LEFT]):
                q = agent.get_q_values(episode["state"])[i]
                draw_text(screen, f"{action.name}: {q:+.3f}", rect.x + 14, y, 12)
                y += 18

    def _draw_compare_tab(self, screen, rect):
        draw_panel(screen, rect, "Benchmarks")
        draw_bar_chart(screen, pygame.Rect(rect.x + 12, rect.y + 40, rect.width - 24, 140), COMPARISON_LABELS, COMPARISON_ROWS[0][1], "Best", THEME.success)

    def _draw_state_tab(self, screen, rect, state, action_analysis, episode):
        half = (rect.width - 12) // 2
        draw_panel(screen, pygame.Rect(rect.x, rect.y, half, rect.height), "State")
        sy = rect.y + 32
        for i, label in enumerate(STATE_LABELS[:8]):
            draw_text(screen, f"{label}: {int(state[i])}", rect.x + 12, sy, 11)
            sy += 16
        draw_panel(screen, pygame.Rect(rect.x + half + 12, rect.y, half, rect.height), "Moves")
        ry = rect.y + 32
        for action in [Action.STRAIGHT, Action.RIGHT, Action.LEFT]:
            a = action_analysis[action]
            draw_text(screen, f"{action.name}: {'ok' if not a['danger'] else 'X'}", rect.x + half + 24, ry, 12)
            ry += 18
