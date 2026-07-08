from typing import Dict, List, Optional, Tuple

import numpy as np

from snake_rl.agents.base_agent import BaseAgent
from snake_rl.envs.snake_env import Action, Direction, SnakeEnv


Point = Tuple[int, int]


class HamiltonianCycleAgent(BaseAgent):
    """
    Full-solve Snake agent.

    Core idea:
    - Build a Hamiltonian cycle that visits every board cell exactly once.
    - Always move to the next cell in that cycle.
    - Food will eventually appear on the route.
    - This avoids greedy traps and can fill the board.

    This is NOT learning.
    This is a deterministic planning / solving agent.
    """

    name = "Hamiltonian Cycle Agent"

    def __init__(self):
        self.width: Optional[int] = None
        self.height: Optional[int] = None

        self.cycle: List[Point] = []
        self.point_to_index: Dict[Point, int] = {}

        self.debug_info = {
            "mode": "HAMILTONIAN",
            "head_index": None,
            "next_index": None,
            "target_next_point": None,
            "cycle_length": 0,
            "reason": "not initialized",
        }

    def reset(self) -> None:
        self.debug_info = {
            "mode": "HAMILTONIAN",
            "head_index": None,
            "next_index": None,
            "target_next_point": None,
            "cycle_length": len(self.cycle),
            "reason": "agent reset",
        }

    def reset_env(self, env: SnakeEnv) -> np.ndarray:
        """
        Hamiltonian-friendly reset.

        The snake must start already aligned on the cycle:
        tail -> body -> head

        For example:
        cycle[0] = tail
        cycle[1] = body
        cycle[2] = head
        """
        self._ensure_cycle(env.width, env.height)

        if len(self.cycle) < 3:
            raise ValueError("Hamiltonian cycle must have at least 3 cells.")

        tail = self.cycle[0]
        body = self.cycle[1]
        head = self.cycle[2]

        env.snake = [head, body, tail]
        env.direction = self._direction_from_to(body, head)

        env.score = 0
        env.steps = 0
        env.steps_since_food = 0

        # Spawn food using the environment's own food logic.
        spawned_food = env._spawn_food()

        # Some versions of _spawn_food return the food.
        # Some versions set env.food internally.
        if spawned_food is not None:
            env.food = spawned_food

        self.debug_info = {
            "mode": "HAMILTONIAN",
            "head_index": 2,
            "next_index": 3,
            "target_next_point": self.cycle[3],
            "cycle_length": len(self.cycle),
            "reason": "environment reset onto Hamiltonian cycle",
        }

        return env.get_state()

    def choose_action(self, state: np.ndarray, env: Optional[SnakeEnv] = None) -> Action:
        if env is None:
            return Action.STRAIGHT

        self._ensure_cycle(env.width, env.height)

        head = env.snake[0]

        if head not in self.point_to_index:
            self.debug_info = {
                "mode": "HAMILTONIAN",
                "head_index": None,
                "next_index": None,
                "target_next_point": None,
                "cycle_length": len(self.cycle),
                "reason": "head is not on cycle, using safe fallback",
            }
            return self._safe_fallback_action(env)

        head_index = self.point_to_index[head]
        next_index = (head_index + 1) % len(self.cycle)
        target_next_point = self.cycle[next_index]

        action = self._action_to_reach_point(
            current_direction=env.direction,
            head=head,
            target=target_next_point,
        )

        if action is None:
            self.debug_info = {
                "mode": "HAMILTONIAN",
                "head_index": head_index,
                "next_index": next_index,
                "target_next_point": target_next_point,
                "cycle_length": len(self.cycle),
                "reason": "target requires reverse/invalid move, using fallback",
            }
            return self._safe_fallback_action(env)

        if not self._is_action_safe(env, action):
            self.debug_info = {
                "mode": "HAMILTONIAN",
                "head_index": head_index,
                "next_index": next_index,
                "target_next_point": target_next_point,
                "cycle_length": len(self.cycle),
                "reason": "cycle move unsafe, using fallback",
            }
            return self._safe_fallback_action(env)

        self.debug_info = {
            "mode": "HAMILTONIAN",
            "head_index": head_index,
            "next_index": next_index,
            "target_next_point": target_next_point,
            "cycle_length": len(self.cycle),
            "reason": "following Hamiltonian cycle",
        }

        return action

    # --------------------------------------------------
    # Hamiltonian cycle generation
    # --------------------------------------------------

    def _ensure_cycle(self, width: int, height: int) -> None:
        if self.width == width and self.height == height and self.cycle:
            return

        self.width = width
        self.height = height

        self.cycle = self._generate_cycle(width, height)
        self.point_to_index = {
            point: index for index, point in enumerate(self.cycle)
        }

        self._validate_cycle(width, height)

    def _generate_cycle(self, width: int, height: int) -> List[Point]:
        """
        Generate a Hamiltonian cycle for rectangular boards.

        A rectangular grid has a Hamiltonian cycle when at least one dimension is even.

        Supports:
        - 12x12
        - 20x20
        - any board where width or height is even
        """
        if width < 2 or height < 2:
            raise ValueError("Board must be at least 2x2.")

        if width % 2 == 0:
            return self._generate_even_width_cycle(width, height)

        if height % 2 == 0:
            # Generate cycle on transposed board, then swap coordinates back.
            transposed_cycle = self._generate_even_width_cycle(height, width)
            return [(y, x) for (x, y) in transposed_cycle]

        raise ValueError(
            f"Cannot create Hamiltonian cycle for {width}x{height}. "
            "At least one dimension must be even."
        )

    def _generate_even_width_cycle(self, width: int, height: int) -> List[Point]:
        """
        Hamiltonian cycle for even-width board.

        Example 4x4 cycle indices:

        00 01 02 03
        15 06 05 04
        14 07 08 09
        13 12 11 10

        The route is:
        00 -> 01 -> 02 -> 03 -> 04 -> ... -> 15 -> 00
        """
        cycle: List[Point] = []

        # Top row: left to right.
        for x in range(width):
            cycle.append((x, 0))

        # Interior/right columns: snake down/up from right to left, excluding x=0.
        going_down = True

        for x in range(width - 1, 0, -1):
            if going_down:
                for y in range(1, height):
                    cycle.append((x, y))
            else:
                for y in range(height - 1, 0, -1):
                    cycle.append((x, y))

            going_down = not going_down

        # Left wall: bottom to top, excluding top-left already included.
        for y in range(height - 1, 0, -1):
            cycle.append((0, y))

        return cycle

    def _validate_cycle(self, width: int, height: int) -> None:
        expected_cells = width * height

        if len(self.cycle) != expected_cells:
            raise ValueError(
                f"Invalid Hamiltonian cycle length. "
                f"Expected {expected_cells}, got {len(self.cycle)}."
            )

        if len(set(self.cycle)) != expected_cells:
            raise ValueError("Invalid Hamiltonian cycle: duplicate cells found.")

        for point in self.cycle:
            x, y = point
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"Invalid point outside board: {point}")

        # Check every consecutive move is adjacent, including final -> first.
        for index in range(len(self.cycle)):
            current = self.cycle[index]
            nxt = self.cycle[(index + 1) % len(self.cycle)]

            distance = abs(current[0] - nxt[0]) + abs(current[1] - nxt[1])

            if distance != 1:
                raise ValueError(
                    f"Invalid Hamiltonian cycle: {current} -> {nxt} is not adjacent."
                )

    # --------------------------------------------------
    # Movement helpers
    # --------------------------------------------------

    def _direction_from_to(self, start: Point, end: Point) -> Direction:
        sx, sy = start
        ex, ey = end

        dx = ex - sx
        dy = ey - sy

        if dx == 1 and dy == 0:
            return Direction.RIGHT

        if dx == -1 and dy == 0:
            return Direction.LEFT

        if dx == 0 and dy == 1:
            return Direction.DOWN

        if dx == 0 and dy == -1:
            return Direction.UP

        raise ValueError(f"Points are not adjacent: {start} -> {end}")

    def _action_to_reach_point(
        self,
        current_direction: Direction,
        head: Point,
        target: Point,
    ) -> Optional[Action]:
        desired_direction = self._direction_from_to(head, target)

        if desired_direction == current_direction:
            return Action.STRAIGHT

        if desired_direction == Direction((int(current_direction) + 1) % 4):
            return Action.RIGHT

        if desired_direction == Direction((int(current_direction) - 1) % 4):
            return Action.LEFT

        # Opposite direction is not allowed in Snake.
        return None

    def _next_point_after_action(
        self,
        head: Point,
        direction: Direction,
        action: Action,
    ) -> Point:
        if action == Action.STRAIGHT:
            next_direction = direction
        elif action == Action.RIGHT:
            next_direction = Direction((int(direction) + 1) % 4)
        elif action == Action.LEFT:
            next_direction = Direction((int(direction) - 1) % 4)
        else:
            raise ValueError(f"Unknown action: {action}")

        x, y = head

        if next_direction == Direction.RIGHT:
            return x + 1, y

        if next_direction == Direction.DOWN:
            return x, y + 1

        if next_direction == Direction.LEFT:
            return x - 1, y

        if next_direction == Direction.UP:
            return x, y - 1

        raise ValueError(f"Unknown direction: {next_direction}")

    def _is_action_safe(self, env: SnakeEnv, action: Action) -> bool:
        next_point = self._next_point_after_action(
            head=env.snake[0],
            direction=env.direction,
            action=action,
        )

        x, y = next_point

        if x < 0 or x >= env.width or y < 0 or y >= env.height:
            return False

        blocked = set(env.snake)

        # The tail usually moves away, so allow moving into current tail.
        if len(env.snake) > 0:
            blocked.discard(env.snake[-1])

        return next_point not in blocked

    def _safe_fallback_action(self, env: SnakeEnv) -> Action:
        for action in [Action.STRAIGHT, Action.RIGHT, Action.LEFT]:
            if self._is_action_safe(env, action):
                return action

        return Action.STRAIGHT
