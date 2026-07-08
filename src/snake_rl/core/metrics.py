from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Counter as CounterType, Deque, List

from snake_rl.envs.snake_env import Action


@dataclass
class EpisodeRecord:
    """
    Stores the final result of one full game.
    """
    episode: int
    score: int
    steps: int
    total_reward: float
    reason: str


@dataclass
class AgentMetrics:
    """
    Tracks performance for one agent.

    This helps us answer:
    - Is this agent improving?
    - What is its best score?
    - How long does it survive?
    - What actions does it use most?
    """

    max_history: int = 100

    episode_number: int = 0
    current_total_reward: float = 0.0
    current_steps: int = 0

    current_action_counts: CounterType[Action] = field(default_factory=Counter)
    lifetime_action_counts: CounterType[Action] = field(default_factory=Counter)

    history: Deque[EpisodeRecord] = field(init=False)

    def __post_init__(self):
        self.history = deque(maxlen=self.max_history)

    def start_episode(self) -> None:
        """
        Called when a new game starts.
        """
        self.current_total_reward = 0.0
        self.current_steps = 0
        self.current_action_counts.clear()

    def record_step(self, action: Action, reward: float) -> None:
        """
        Called after every move.
        """
        self.current_total_reward += reward
        self.current_steps += 1
        self.current_action_counts[action] += 1
        self.lifetime_action_counts[action] += 1

    def end_episode(self, score: int, steps: int, reason: str) -> None:
        """
        Called when the game ends.
        """
        self.episode_number += 1

        self.history.append(
            EpisodeRecord(
                episode=self.episode_number,
                score=score,
                steps=steps,
                total_reward=self.current_total_reward,
                reason=reason,
            )
        )

    def scores(self) -> List[int]:
        return [record.score for record in self.history]

    def rewards(self) -> List[float]:
        return [record.total_reward for record in self.history]

    def recent_records(self, n: int = 8) -> List[EpisodeRecord]:
        return list(self.history)[-n:]

    def best_score(self) -> int:
        if not self.history:
            return 0
        return max(record.score for record in self.history)

    def average_score(self) -> float:
        if not self.history:
            return 0.0
        return sum(record.score for record in self.history) / len(self.history)

    def rolling_average_score(self, n: int = 10) -> float:
        if not self.history:
            return 0.0

        recent = list(self.history)[-n:]
        return sum(record.score for record in recent) / len(recent)
