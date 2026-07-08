"""
Replay memory for DQN.

Stores past experiences so the agent can learn from random mini-batches
instead of only the most recent move.
"""

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np

from snake_rl.envs.snake_env import Action


@dataclass
class Experience:
    """One step of experience: state, action, reward, next_state, done."""

    state: np.ndarray
    action: Action
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayMemory:
    """
    Fixed-size buffer of past experiences.

    DQN samples random batches from here to break correlation between
    consecutive game steps.
    """

    def __init__(self, capacity: int = 100_000):
        self.memory: Deque[Experience] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: Action,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.memory.append(
            Experience(
                state=state.copy(),
                action=action,
                reward=reward,
                next_state=next_state.copy(),
                done=done,
            )
        )

    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)
