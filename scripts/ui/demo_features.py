"""Demo/presentation features: effects, replay, sound, narration, leaderboard, presets."""

import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pygame

from snake_rl.envs.snake_env import Action, SnakeEnv
from ui.agent_data import AGENT_PROFILES, MANUAL_AGENT_KEY, agent_key_from_pygame

Point = Tuple[int, int]


# ── Demo presets ──────────────────────────────────────────────────────────────

@dataclass
class DemoPreset:
    name: str
    label: str
    agent_key: int
    width: int
    height: int
    fps: int
    learning_mode: str = "autonomous"


DEMO_PRESETS: Dict[str, DemoPreset] = {
    "quick": DemoPreset("quick", "Quick", pygame.K_2, 8, 8, 10),
    "rl": DemoPreset("rl", "RL Demo", pygame.K_3, 12, 12, 20, "training"),
    "perfect": DemoPreset("perfect", "Perfect", pygame.K_5, 12, 12, 80),
    "play": DemoPreset("play", "Play", MANUAL_AGENT_KEY, 10, 10, 10),
}


# All agents available for battle selection (pygame key -> profile key)
BATTLE_AGENT_OPTIONS = [
    (49, 1), (50, 2), (51, 3), (52, 4), (53, 5), (54, 6), (MANUAL_AGENT_KEY, 7),
]

AGENT_ICONS = {1: "🎲", 2: "🎯", 3: "📊", 4: "🧠", 5: "🔄", 6: "⚡", 7: "🎮"}
AGENT_TOOLTIPS = {
    1: "Random baseline — picks moves blindly",
    2: "Greedy — chases food, no long-term plan",
    3: "Q-Learning — tabular RL, 10M steps trained",
    4: "DQN — neural network Q-learner",
    5: "Hamiltonian — guaranteed full board solve",
    6: "Hybrid — Hamiltonian + smart shortcuts",
    7: "You — arrow keys or WASD to steer",
}


# ── Visual effects ────────────────────────────────────────────────────────────

@dataclass
class ScorePop:
    text: str
    x: float
    y: float
    life: float = 1.0
    color: Tuple[int, int, int] = (52, 211, 153)


@dataclass
class ConfettiParticle:
    x: float
    y: float
    vx: float
    vy: float
    color: Tuple[int, int, int]
    life: float = 1.0


class EffectsManager:
    def __init__(self):
        self.shake = 0.0
        self.score_pops: List[ScorePop] = []
        self.confetti: List[ConfettiParticle] = []
        self.trail: Deque[Point] = deque(maxlen=24)
        self.last_score = 0

    def update(self, dt: float = 0.05) -> None:
        if self.shake > 0:
            self.shake = max(0, self.shake - dt * 3)
        self.score_pops = [p for p in self.score_pops if p.life > 0]
        for p in self.score_pops:
            p.y -= 1.2
            p.life -= dt
        self.confetti = [c for c in self.confetti if c.life > 0]
        for c in self.confetti:
            c.x += c.vx
            c.y += c.vy
            c.vy += 0.15
            c.life -= dt * 0.4

    def shake_offset(self) -> Tuple[int, int]:
        if self.shake <= 0:
            return 0, 0
        mag = int(self.shake * 8)
        return random.randint(-mag, mag), random.randint(-mag, mag)

    def on_step(self, env: SnakeEnv, reward: float, done: bool, reason: str) -> None:
        if env.snake:
            self.trail.appendleft(env.snake[0])
        if env.score > self.last_score:
            self.score_pops.append(ScorePop("+1", float(env.snake[0][0]), float(env.snake[0][1])))
            self.last_score = env.score
        if done:
            if reason == "collision":
                self.shake = 1.0
            elif reason == "board_full":
                self._spawn_confetti(env.width, env.height)

    def on_reset(self, env: SnakeEnv) -> None:
        self.last_score = env.score
        self.trail.clear()

    def _spawn_confetti(self, w: int, h: int) -> None:
        colors = [(255, 100, 100), (100, 200, 255), (255, 220, 80), (180, 120, 255), (80, 230, 160)]
        for _ in range(80):
            self.confetti.append(ConfettiParticle(
                x=random.uniform(0, w),
                y=random.uniform(0, h),
                vx=random.uniform(-2, 2),
                vy=random.uniform(-4, -1),
                color=random.choice(colors),
            ))

    def draw_overlays(self, screen: pygame.Surface, rect: pygame.Rect, env: SnakeEnv, cell: int, ox: int, oy: int) -> None:
        from ui.components import draw_text
        from ui.theme import THEME

        for i, (tx, ty) in enumerate(list(self.trail)[1:8]):
            alpha = max(20, 80 - i * 12)
            tr = pygame.Rect(ox + tx * cell + cell // 4, oy + ty * cell + cell // 4, cell // 2, cell // 2)
            s = pygame.Surface((tr.width, tr.height), pygame.SRCALPHA)
            s.fill((*THEME.snake_body[:3], alpha))
            screen.blit(s, tr.topleft)

        for p in self.score_pops:
            px = ox + int(p.x * cell) + cell // 2
            py = oy + int(p.y * cell) - int((1 - p.life) * 30)
            alpha = int(255 * p.life)
            surf = pygame.Surface((60, 24), pygame.SRCALPHA)
            from ui.theme import font
            txt = font(18, bold=True).render(p.text, True, (*p.color, alpha))
            surf.blit(txt, (0, 0))
            screen.blit(surf, (px, py))

        for c in self.confetti:
            px = ox + int(c.x * cell)
            py = oy + int(c.y * cell)
            pygame.draw.circle(screen, c.color, (px, py), 3)


# ── Replay ────────────────────────────────────────────────────────────────────

@dataclass
class ReplayFrame:
    snake: List[Point]
    food: Optional[Point]
    score: int
    direction: int


class ReplayBuffer:
    def __init__(self, capacity: int = 600):
        self.frames: Deque[ReplayFrame] = deque(maxlen=capacity)
        self.best_frames: List[ReplayFrame] = []
        self.best_score = 0
        self.playing = False
        self.play_index = 0
        self.play_speed = 3

    def record(self, env: SnakeEnv) -> None:
        if self.playing:
            return
        self.frames.append(ReplayFrame(
            snake=list(env.snake),
            food=env.food,
            score=env.score,
            direction=int(env.direction),
        ))

    def save_best_if_better(self, env: SnakeEnv) -> bool:
        if env.score > self.best_score and len(self.frames) > 10:
            self.best_score = env.score
            self.best_frames = list(self.frames)
            return True
        return False

    def start_replay(self) -> bool:
        if not self.best_frames:
            return False
        self.playing = True
        self.play_index = 0
        return True

    def stop_replay(self) -> None:
        self.playing = False
        self.play_index = 0

    def current_frame(self) -> Optional[ReplayFrame]:
        if not self.playing or not self.best_frames:
            return None
        return self.best_frames[self.play_index]

    def advance(self) -> bool:
        if not self.playing:
            return False
        self.play_index += 1
        if self.play_index >= len(self.best_frames):
            self.stop_replay()
            return False
        return True


# ── Narration ───────────────────────────────────────────────────────────────

NARRATION_BY_AGENT = {
    1: "Random chaos — no plan, just vibes.",
    2: "Greedy eyes the food — short-term smart, long-term risky.",
    3: "Q-Learning weighs past experience for each move.",
    4: "DQN's neural net predicts the best action.",
    5: "Hamiltonian follows a safe cycle — food eaten on the route.",
    6: "Hybrid takes clever shortcuts without trapping itself.",
    7: "Your turn — steer with arrows or WASD!",
}

EVENT_NARRATION = {
    "ate_food": "Food eaten! Score climbing.",
    "collision": "Crash! Wall or self collision.",
    "board_full": "Perfect! Board completely filled!",
    "timeout": "Timed out — too long without food.",
    "paused": "Paused — press Space to resume.",
    "battle": "Agent battle — who scores higher?",
    "replay": "Replaying the best run…",
    "neural_viz": "Opening live neural network view…",
}


def get_narration(agent_key: int, event: str = "", learning_mode: str = "") -> str:
    if event and event in EVENT_NARRATION:
        return EVENT_NARRATION[event]
    num = agent_key_from_pygame(agent_key) or 1
    base = NARRATION_BY_AGENT.get(num, "")
    if num in (3, 4) and learning_mode == "training":
        return base + " (actively learning)"
    return base


# ── Leaderboard ───────────────────────────────────────────────────────────────

@dataclass
class LeaderboardEntry:
    agent_name: str
    score: int
    board: str
    reason: str


class SessionLeaderboard:
    def __init__(self, max_entries: int = 20):
        self.entries: List[LeaderboardEntry] = []
        self.max_entries = max_entries
        self.best_by_agent: Dict[str, int] = {}

    def record(self, agent_key: int, score: int, width: int, height: int, reason: str) -> None:
        num = agent_key_from_pygame(agent_key) or 1
        name = AGENT_PROFILES[num].short_name
        board = f"{width}×{height}"
        self.entries.append(LeaderboardEntry(name, score, board, reason))
        self.entries.sort(key=lambda e: e.score, reverse=True)
        self.entries = self.entries[: self.max_entries]
        self.best_by_agent[name] = max(self.best_by_agent.get(name, 0), score)


# ── Sound ─────────────────────────────────────────────────────────────────────

class SoundManager:
    def __init__(self):
        self.enabled = True
        self._sounds: Dict[str, pygame.mixer.Sound] = {}
        self._ready = False

    def init(self) -> None:
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self._sounds = {
                "eat": self._tone(660, 0.08),
                "death": self._tone(180, 0.25),
                "highscore": self._tone(880, 0.15),
                "win": self._chord(),
            }
            self._ready = True
        except pygame.error:
            self._ready = False
            self.enabled = False

    def _tone(self, freq: float, duration: float) -> pygame.mixer.Sound:
        rate = 22050
        n = int(rate * duration)
        t = np.arange(n) / rate
        wave = (np.sin(2 * np.pi * freq * t) * np.exp(-t * 8) * 32767 * 0.25).astype(np.int16)
        stereo = np.column_stack([wave, wave])
        return pygame.sndarray.make_sound(stereo.copy())

    def _chord(self) -> pygame.mixer.Sound:
        rate = 22050
        duration = 0.35
        n = int(rate * duration)
        t = np.arange(n) / rate
        wave = sum(np.sin(2 * np.pi * f * t) for f in (523, 659, 784)) / 3
        wave = (wave * np.exp(-t * 3) * 32767 * 0.2).astype(np.int16)
        stereo = np.column_stack([wave, wave])
        return pygame.sndarray.make_sound(stereo.copy())

    def play(self, name: str) -> None:
        if self.enabled and self._ready and name in self._sounds:
            self._sounds[name].play()


# ── Demo session state ────────────────────────────────────────────────────────

@dataclass
class DemoSession:
    presentation_mode: bool = False
    battle_mode: bool = False
    battle_key_a: int = 0  # set at runtime to pygame.K_2
    battle_key_b: int = 0  # set at runtime to pygame.K_1
    battle_env: Any = None
    battle_episode: Any = None
    battle_metrics: Any = None
    ghost_enabled: bool = True
    decision_overlay: bool = False
    sound_enabled: bool = True
    narration: str = ""
    narration_event_until: int = 0
    export_flash_until: int = 0
    nn_viz_opened: bool = False
    nn_live: bool = False
    nn_step: int = 0
    mouse_pos: Tuple[int, int] = (0, 0)
    hovered_tooltip: str = ""

    effects: EffectsManager = field(default_factory=EffectsManager)
    replay: ReplayBuffer = field(default_factory=ReplayBuffer)
    leaderboard: SessionLeaderboard = field(default_factory=SessionLeaderboard)
    sound: SoundManager = field(default_factory=SoundManager)

    def toggle_presentation(self) -> None:
        self.presentation_mode = not self.presentation_mode

    def toggle_battle(self) -> None:
        self.battle_mode = not self.battle_mode

    def set_narration_event(self, event: str, now: int, duration: int = 2000) -> None:
        self.narration = EVENT_NARRATION.get(event, event)
        self.narration_event_until = now + duration

    def update_narration(self, agent_key: int, learning_mode: str, now: int) -> None:
        if now < self.narration_event_until:
            return
        self.narration = get_narration(agent_key, learning_mode=learning_mode)
