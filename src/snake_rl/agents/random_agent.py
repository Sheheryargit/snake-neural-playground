import random
from typing import Optional

import numpy as np

from snake_rl.agents.base_agent import BaseAgent
from snake_rl.envs.snake_env import Action, SnakeEnv


class RandomAgent(BaseAgent):
    """
    Dumb baseline agent.

    It does not learn.
    It just randomly chooses:
    - straight
    - right
    - left

    This is useful because we need to know how bad the system is
    before learning starts.
    """

    name = "Random Agent"

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def choose_action(self, state: np.ndarray, env: Optional[SnakeEnv] = None) -> Action:
        return self.rng.choice([
            Action.STRAIGHT,
            Action.RIGHT,
            Action.LEFT,
        ])
