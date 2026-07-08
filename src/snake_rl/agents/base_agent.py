from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from snake_rl.envs.snake_env import Action, SnakeEnv


class BaseAgent(ABC):
    """
    Base class for all agents.

    Every agent must know how to choose an action.
    Some agents also learn after each step.
    """

    name = "Base Agent"

    def reset(self) -> None:
        """
        Called when a new episode starts.
        """
        pass

    @abstractmethod
    def choose_action(self, state: np.ndarray, env: Optional[SnakeEnv] = None) -> Action:
        """
        Given the current state, return an action.
        """
        pass

    def learn(
        self,
        state: np.ndarray,
        action: Action,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Called after every move.

        Random and Greedy agents do not learn, so this does nothing by default.
        Q-learning and DQN agents will override this.
        """
        pass
