"""Agent metadata and benchmark data from repo training logs."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class AgentProfile:
    key: int
    short_name: str
    full_name: str
    tagline: str
    type_label: str
    implementation: str
    simple_description: str
    benchmark_best: Optional[float]
    benchmark_avg: Optional[float]
    benchmark_note: str
    max_theoretical: float = 141.0


AGENT_PROFILES: Dict[int, AgentProfile] = {
    1: AgentProfile(
        key=1,
        short_name="Random",
        full_name="Random Agent",
        tagline="Baseline chaos",
        type_label="Baseline",
        implementation="Uniform random choice among STRAIGHT, RIGHT, LEFT. No memory, no planning.",
        simple_description="Picks moves at random. Useful as a baseline to prove learning works.",
        benchmark_best=15.0,
        benchmark_avg=3.0,
        benchmark_note="Typical random baseline on 12×12",
    ),
    2: AgentProfile(
        key=2,
        short_name="Greedy",
        full_name="Greedy Food Agent",
        tagline="One-step food chaser",
        type_label="Rule-based",
        implementation="Evaluates each safe move, picks the one that minimizes Manhattan distance to food.",
        simple_description="Always tries to get closer to food — smart short-term, but no long-term plan.",
        benchmark_best=68.0,
        benchmark_avg=20.6,
        benchmark_note="1M games benchmark from repo",
    ),
    3: AgentProfile(
        key=3,
        short_name="Q-Learn",
        full_name="Q-Learning Agent",
        tagline="Tabular RL",
        type_label="Reinforcement Learning",
        implementation=(
            "11-value state → Q-table lookup. Epsilon-greedy exploration. "
            "Model: q_learning_10m.pkl (10M steps). Alpha=0.1, gamma=0.9."
        ),
        simple_description="Learns from experience using a lookup table of state → action values.",
        benchmark_best=45.0,
        benchmark_avg=12.3,
        benchmark_note="10M-step training run",
    ),
    4: AgentProfile(
        key=4,
        short_name="DQN",
        full_name="DQN Agent",
        tagline="Neural Q-network",
        type_label="Deep RL",
        implementation=(
            "11→128→128→3 MLP. Replay memory 100k. Linear epsilon decay over 500k steps. "
            "SmoothL1 loss, Adam lr=0.0005. Model: dqn_11_state.pt."
        ),
        simple_description="Uses a neural network to predict the best move from the game state.",
        benchmark_best=44.0,
        benchmark_avg=13.1,
        benchmark_note="1M-step training run (fixed epsilon)",
    ),
    5: AgentProfile(
        key=5,
        short_name="Hamilton",
        full_name="Hamiltonian Cycle Agent",
        tagline="Pure full-solve",
        type_label="Planning / Solver",
        implementation=(
            "Pre-computed Hamiltonian cycle over all 144 cells. "
            "Always advances to next cycle cell. Never chases food directly."
        ),
        simple_description="Follows a safe route that visits every square — food is eaten when encountered.",
        benchmark_best=141.0,
        benchmark_avg=141.0,
        benchmark_note="10M-step run — 100% board fill, 5097 avg steps/ep",
    ),
    6: AgentProfile(
        key=6,
        short_name="Hybrid",
        full_name="Hybrid Hamiltonian Agent",
        tagline="Smart shortcuts",
        type_label="Planning / Solver",
        implementation=(
            "Hamiltonian backbone + safe shortcuts toward food. "
            "Shortcuts only if tail-gap preserved on cycle (min_gap=4). "
            "~10% shortcut moves in 10M benchmark."
        ),
        simple_description="Same safe route, but takes shortcuts to food when it won't trap itself.",
        benchmark_best=141.0,
        benchmark_avg=141.0,
        benchmark_note="10M-step run — 100% board fill, 4394 avg steps/ep (−14% vs pure)",
    ),
    7: AgentProfile(
        key=7,
        short_name="You",
        full_name="Manual Player",
        tagline="Play live",
        type_label="Human",
        implementation="Arrow keys or WASD to steer. One move per tick at the current game speed.",
        simple_description="Take the controls and play Snake yourself. Use arrow keys or WASD to steer.",
        benchmark_best=None,
        benchmark_avg=None,
        benchmark_note="Your scores are tracked locally in this session.",
    ),
}


COMPARISON_ROWS = [
    ("Best score (12×12)", [15, 68, 45, 44, 141, 141]),
    ("Avg score", [3, 20.6, 12.3, 13.1, 141, 141]),
    ("Board fill %", [2, 50, 32, 31, 100, 100]),
    ("Avg steps / game", [40, 80, 120, 118, 5097, 4394]),
    ("Learning?", [0, 0, 1, 1, 0, 0]),
]

COMPARISON_LABELS = ["Random", "Greedy", "Q", "DQN", "Ham.", "Hybrid"]


MANUAL_AGENT_KEY = 104  # pygame.K_h


def agent_key_from_pygame(key: int) -> Optional[int]:
    mapping = {49: 1, 50: 2, 51: 3, 52: 4, 53: 5, 54: 6, MANUAL_AGENT_KEY: 7}
    return mapping.get(key)
