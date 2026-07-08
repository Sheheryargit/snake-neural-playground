from typing import Optional

import numpy as np

from snake_rl.agents.base_agent import BaseAgent
from snake_rl.envs.snake_env import Action, Direction, SnakeEnv


class GreedyFoodAgent(BaseAgent):
    """
    Simple rule-based agent.

    It tries to choose the safe move that gets closest to the food.

    Important:
    This is NOT learning.
    This is just a hand-coded strategy.
    We use it as a comparison point.
    """

    name = "Greedy Food Agent"

    def choose_action(self, state: np.ndarray, env: Optional[SnakeEnv] = None) -> Action:
        if env is None or env.food is None:
            return Action.STRAIGHT

        safe_actions = []

        for action in [Action.STRAIGHT, Action.RIGHT, Action.LEFT]:
            next_direction = self._direction_after_action(env.direction, action)
            next_point = env._point_in_direction(env.snake[0], next_direction)

            if not env._is_danger(next_point):
                safe_actions.append(action)

        if not safe_actions:
            return Action.STRAIGHT

        best_action = min(
            safe_actions,
            key=lambda action: self._distance_after_action(env, action),
        )

        return best_action

    def _direction_after_action(self, current_direction: Direction, action: Action) -> Direction:
        if action == Action.STRAIGHT:
            return current_direction

        if action == Action.RIGHT:
            return Direction((int(current_direction) + 1) % 4)

        if action == Action.LEFT:
            return Direction((int(current_direction) - 1) % 4)

        raise ValueError(f"Unknown action: {action}")

    def _distance_after_action(self, env: SnakeEnv, action: Action) -> int:
        next_direction = self._direction_after_action(env.direction, action)
        next_point = env._point_in_direction(env.snake[0], next_direction)

        food_x, food_y = env.food
        next_x, next_y = next_point

        return abs(next_x - food_x) + abs(next_y - food_y)
