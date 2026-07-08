from typing import List, Optional

import numpy as np

from snake_rl.agents.base_agent import BaseAgent
from snake_rl.core.pathfinding import (
    Point,
    action_to_reach_point,
    find_path,
    find_path_to_nearest_boundary,
    get_next_boundary_point,
    is_boundary_point,
    is_safe_point,
    next_point_after_action,
)
from snake_rl.envs.snake_env import Action, SnakeEnv


class BoundaryLoopAgent(BaseAgent):
    """
    Boundary-first Snake agent.

    Core rule:
    The boundary loop is the main road.

    Behaviour:
    1. If not on boundary, go to boundary.
    2. If on boundary, keep looping clockwise.
    3. Only leave boundary for food if:
       - food is reachable
       - after eating food, boundary is reachable again
       - tail is reachable after eating
    4. After eating, always return to boundary before chasing new food.

    This is NOT learning.
    This is a rule-based path-planning algorithm.
    """

    name = "Boundary Loop Agent"

    MODE_GET_TO_BOUNDARY = "GET_TO_BOUNDARY"
    MODE_LOOP_BOUNDARY = "LOOP_BOUNDARY"
    MODE_DETOUR_TO_FOOD = "DETOUR_TO_FOOD"
    MODE_RETURN_TO_BOUNDARY = "RETURN_TO_BOUNDARY"

    def __init__(
        self,
        max_detour_length: int = 10,
        require_tail_reachable: bool = True,
    ):
        self.max_detour_length = max_detour_length
        self.require_tail_reachable = require_tail_reachable

        self.mode = self.MODE_GET_TO_BOUNDARY
        self.target_food: Optional[Point] = None

        self.debug_info = self._debug(
            target="boundary",
            path_length=0,
            detour_safe=False,
            reason="starting: first get to boundary",
        )

    def reset(self) -> None:
        self.mode = self.MODE_GET_TO_BOUNDARY
        self.target_food = None

        self.debug_info = self._debug(
            target="boundary",
            path_length=0,
            detour_safe=False,
            reason="reset: get to boundary",
        )

    def choose_action(self, state: np.ndarray, env: Optional[SnakeEnv] = None) -> Action:
        if env is None:
            return Action.STRAIGHT

        head = env.snake[0]

        # If not on boundary and returning/getting there, path to boundary first.
        if self.mode in [self.MODE_GET_TO_BOUNDARY, self.MODE_RETURN_TO_BOUNDARY]:
            if is_boundary_point(head, env.width, env.height):
                self.mode = self.MODE_LOOP_BOUNDARY
                self.target_food = None
            else:
                return self._move_towards_boundary(env)

        # Food changed while detouring → ate previous food → return to boundary.
        if self.mode == self.MODE_DETOUR_TO_FOOD:
            if env.food != self.target_food:
                self.mode = self.MODE_RETURN_TO_BOUNDARY
                self.target_food = None
                return self._move_towards_boundary(env)

            action = self._move_towards_food(env)

            if action is not None:
                return action

            self.mode = self.MODE_RETURN_TO_BOUNDARY
            self.target_food = None
            return self._move_towards_boundary(env)

        # LOOP_BOUNDARY: default clockwise loop; optional safe food detour.
        if self.mode == self.MODE_LOOP_BOUNDARY:
            if is_boundary_point(head, env.width, env.height):
                if self._safe_detour_to_food_and_back_exists(env):
                    self.mode = self.MODE_DETOUR_TO_FOOD
                    self.target_food = env.food
                    return self._move_towards_food(env)

            return self._follow_boundary_loop(env)

        return self._safe_fallback_action(env)

    def _follow_boundary_loop(self, env: SnakeEnv) -> Action:
        head = env.snake[0]

        if not is_boundary_point(head, env.width, env.height):
            self.mode = self.MODE_RETURN_TO_BOUNDARY
            return self._move_towards_boundary(env)

        desired_next = get_next_boundary_point(
            point=head,
            width=env.width,
            height=env.height,
        )

        action = action_to_reach_point(
            head=head,
            direction=env.direction,
            target_next_point=desired_next,
        )

        if action is not None and self._is_action_safe(env, action):
            self.debug_info = self._debug(
                target="boundary loop",
                path_length=1,
                detour_safe=False,
                reason="following clockwise boundary loop",
            )
            return action

        self.debug_info = self._debug(
            target="boundary loop",
            path_length=0,
            detour_safe=False,
            reason="boundary next point unavailable, using safe fallback",
        )
        return self._safe_fallback_action(env)

    def _move_towards_boundary(self, env: SnakeEnv) -> Action:
        path = find_path_to_nearest_boundary(
            start=env.snake[0],
            snake=env.snake,
            width=env.width,
            height=env.height,
        )

        if path is None or len(path) < 2:
            self.debug_info = self._debug(
                target="boundary",
                path_length=0,
                detour_safe=False,
                reason="no path to boundary, using safe fallback",
            )
            return self._safe_fallback_action(env)

        action = self._action_from_path(env, path)

        if action is None:
            self.debug_info = self._debug(
                target="boundary",
                path_length=len(path),
                detour_safe=False,
                reason="path to boundary exists but action impossible",
            )
            return self._safe_fallback_action(env)

        self.debug_info = self._debug(
            target="nearest boundary",
            path_length=len(path),
            detour_safe=False,
            reason="returning to boundary",
        )

        return action

    def _move_towards_food(self, env: SnakeEnv) -> Optional[Action]:
        if env.food is None:
            return None

        path = find_path(
            start=env.snake[0],
            target=env.food,
            snake=env.snake,
            width=env.width,
            height=env.height,
            ignore_tail=True,
        )

        if path is None or len(path) < 2:
            return None

        action = self._action_from_path(env, path)

        if action is None:
            return None

        self.debug_info = self._debug(
            target="food",
            path_length=len(path),
            detour_safe=True,
            reason="detouring to food, return already checked",
        )

        return action

    def _safe_detour_to_food_and_back_exists(self, env: SnakeEnv) -> bool:
        if env.food is None:
            return False

        path_to_food = find_path(
            start=env.snake[0],
            target=env.food,
            snake=env.snake,
            width=env.width,
            height=env.height,
            ignore_tail=True,
        )

        if path_to_food is None:
            self.debug_info = self._debug(
                target="boundary loop",
                path_length=0,
                detour_safe=False,
                reason="no path to food, stay on boundary",
            )
            return False

        if len(path_to_food) > self.max_detour_length:
            self.debug_info = self._debug(
                target="boundary loop",
                path_length=len(path_to_food),
                detour_safe=False,
                reason="food too far, stay on boundary",
            )
            return False

        simulated_snake = self._simulate_snake_after_path(
            snake=env.snake,
            path=path_to_food,
            food=env.food,
        )

        if not self._can_return_to_boundary_after_simulation(
            snake=simulated_snake,
            width=env.width,
            height=env.height,
        ):
            self.debug_info = self._debug(
                target="boundary loop",
                path_length=len(path_to_food),
                detour_safe=False,
                reason="food reachable but cannot return to boundary",
            )
            return False

        if self.require_tail_reachable:
            if not self._can_reach_tail_after_simulation(
                snake=simulated_snake,
                width=env.width,
                height=env.height,
            ):
                self.debug_info = self._debug(
                    target="boundary loop",
                    path_length=len(path_to_food),
                    detour_safe=False,
                    reason="food reachable but tail not reachable after eating",
                )
                return False

        self.debug_info = self._debug(
            target="food",
            path_length=len(path_to_food),
            detour_safe=True,
            reason="safe detour found: food and return path are valid",
        )

        return True

    def _simulate_snake_after_path(
        self,
        snake: List[Point],
        path: List[Point],
        food: Point,
    ) -> List[Point]:
        simulated = list(snake)

        for next_point in path[1:]:
            simulated.insert(0, next_point)

            if next_point == food:
                pass
            else:
                simulated.pop()

        return simulated

    def _can_return_to_boundary_after_simulation(
        self,
        snake: List[Point],
        width: int,
        height: int,
    ) -> bool:
        path = find_path_to_nearest_boundary(
            start=snake[0],
            snake=snake,
            width=width,
            height=height,
        )

        return path is not None and len(path) >= 1

    def _can_reach_tail_after_simulation(
        self,
        snake: List[Point],
        width: int,
        height: int,
    ) -> bool:
        head = snake[0]
        tail = snake[-1]

        path_to_tail = find_path(
            start=head,
            target=tail,
            snake=snake,
            width=width,
            height=height,
            ignore_tail=True,
        )

        return path_to_tail is not None

    def _action_from_path(self, env: SnakeEnv, path: List[Point]) -> Optional[Action]:
        if len(path) < 2:
            return None

        next_point = path[1]

        action = action_to_reach_point(
            head=env.snake[0],
            direction=env.direction,
            target_next_point=next_point,
        )

        if action is None:
            return None

        if not self._is_action_safe(env, action):
            return None

        return action

    def _is_action_safe(self, env: SnakeEnv, action: Action) -> bool:
        next_point = next_point_after_action(
            head=env.snake[0],
            direction=env.direction,
            action=action,
        )

        return is_safe_point(
            point=next_point,
            snake=env.snake,
            width=env.width,
            height=env.height,
            ignore_tail=True,
        )

    def _safe_fallback_action(self, env: SnakeEnv) -> Action:
        for action in [Action.STRAIGHT, Action.RIGHT, Action.LEFT]:
            if self._is_action_safe(env, action):
                return action

        return Action.STRAIGHT

    def _debug(
        self,
        target: str,
        path_length: int,
        detour_safe: bool,
        reason: str,
    ) -> dict:
        return {
            "mode": self.mode,
            "target": target,
            "path_length": path_length,
            "detour_safe": detour_safe,
            "reason": reason,
        }
