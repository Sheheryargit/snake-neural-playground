from typing import Optional

import numpy as np

from snake_rl.agents.base_agent import BaseAgent
from snake_rl.envs.snake_env import Action, Direction, SnakeEnv


def action_for_direction(current: Direction, desired: Direction) -> Optional[Action]:
    """Map an absolute heading to a relative turn action."""
    if current == desired:
        return Action.STRAIGHT
    if Direction((int(current) + 1) % 4) == desired:
        return Action.RIGHT
    if Direction((int(current) - 1) % 4) == desired:
        return Action.LEFT
    return None


class ManualPlayer(BaseAgent):
    """Human-controlled agent driven by arrow keys or WASD."""

    name = "You"

    def __init__(self):
        self.desired_direction: Optional[Direction] = None
        self.last_decision = "ready"

    def reset(self) -> None:
        self.desired_direction = None
        self.last_decision = "ready"

    def set_desired_direction(self, desired: Direction, current: Direction) -> None:
        if action_for_direction(current, desired) is not None:
            self.desired_direction = desired
            self.last_decision = desired.name.lower()

    def choose_action(self, state: np.ndarray, env: Optional[SnakeEnv] = None) -> Action:
        if env is None or self.desired_direction is None:
            return Action.STRAIGHT
        action = action_for_direction(env.direction, self.desired_direction)
        return action if action is not None else Action.STRAIGHT
