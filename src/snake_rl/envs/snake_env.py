from dataclasses import dataclass
from enum import IntEnum
import random
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np


Point = Tuple[int, int]

StateMode = Literal["simple", "advanced"]

SIMPLE_STATE_SIZE = 11
ADVANCED_STATE_SIZE = 24


class Action(IntEnum):
    STRAIGHT = 0
    RIGHT = 1
    LEFT = 2


class Direction(IntEnum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3


DIRECTION_VECTORS = {
    Direction.RIGHT: (1, 0),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
    Direction.UP: (0, -1),
}


@dataclass
class StepResult:
    state: np.ndarray
    reward: float
    done: bool
    info: Dict


class SnakeEnv:
    """
    A minimal Snake environment for reinforcement learning.

    This class does NOT render fancy graphics.
    It only manages the game rules:
    - snake movement
    - food spawning
    - collision detection
    - reward calculation
    - state representation
    """

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        seed: Optional[int] = None,
        state_mode: StateMode = "simple",
    ):
        self.width = width
        self.height = height
        self.state_mode = state_mode
        self.rng = random.Random(seed)

        self.snake: List[Point] = []
        self.direction: Direction = Direction.RIGHT
        self.food: Optional[Point] = None
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0

        self.reset()

    def reset(self) -> np.ndarray:
        """
        Starts a new game and returns the first state.
        """
        center_x = self.width // 2
        center_y = self.height // 2

        self.direction = Direction.RIGHT

        self.snake = [
            (center_x, center_y),
            (center_x - 1, center_y),
            (center_x - 2, center_y),
        ]

        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self.food = self._spawn_food()

        return self.get_state()

    def step(self, action: Action) -> StepResult:
        """
        Moves the game forward by one step.

        Input:
            action: STRAIGHT, RIGHT, or LEFT

        Output:
            state: what the agent sees after the move
            reward: feedback for the action
            done: whether the game ended
            info: debug information
        """
        self.steps += 1
        self.steps_since_food += 1

        old_head = self.snake[0]
        old_distance = self._distance_to_food(old_head)

        self._apply_action(action)
        new_head = self._next_head()

        ate_food = self.food is not None and new_head == self.food

        # If the snake is not eating, the tail moves away.
        # So moving into the current tail position is allowed.
        body_to_check = self.snake if ate_food else self.snake[:-1]

        if self._hits_wall(new_head) or new_head in body_to_check:
            return StepResult(
                state=self.get_state(),
                reward=-10.0,
                done=True,
                info={
                    "score": self.score,
                    "reason": "collision",
                    "steps": self.steps,
                },
            )

        self.snake.insert(0, new_head)

        reward = -0.01  # tiny penalty so the snake does not waste time

        if ate_food:
            self.score += 1
            self.steps_since_food = 0
            reward = 10.0
            self.food = self._spawn_food()

            # Board is full: this is the true perfect finish.
            if self.food is None:
                return StepResult(
                    state=self.get_state(),
                    reward=100.0,
                    done=True,
                    info={
                        "score": self.score,
                        "reason": "board_full",
                        "steps": self.steps,
                    },
                )
        else:
            self.snake.pop()

            new_distance = self._distance_to_food(new_head)

            # Small reward shaping:
            # moving closer to food is slightly good,
            # moving away is slightly bad.
            if new_distance < old_distance:
                reward += 0.05
            else:
                reward -= 0.05

        # Prevent infinite looping without eating.
        timeout_limit = max(
            100 * len(self.snake),
            self.width * self.height * 2,
        )

        if self.steps_since_food > timeout_limit:
            return StepResult(
                state=self.get_state(),
                reward=-10.0,
                done=True,
                info={
                    "score": self.score,
                    "reason": "timeout",
                    "steps": self.steps,
                },
            )

        return StepResult(
            state=self.get_state(),
            reward=reward,
            done=False,
            info={
                "score": self.score,
                "reason": "running",
                "steps": self.steps,
            },
        )

    def get_state(self) -> np.ndarray:
        """
        Returns the state vector for the current state_mode.

        simple:   11 features (danger, direction, food direction)
        advanced: 24 features (simple + open space, reachability, traps)
        """
        if self.state_mode == "simple":
            return self.get_simple_state()
        if self.state_mode == "advanced":
            return self.get_advanced_state()
        raise ValueError(f"Unknown state_mode: {self.state_mode}")

    def get_simple_state(self) -> np.ndarray:
        """
        Original 11-value state: danger, direction, food direction.
        """
        head = self.snake[0]

        straight_direction = self.direction
        right_direction = Direction((int(self.direction) + 1) % 4)
        left_direction = Direction((int(self.direction) - 1) % 4)

        point_straight = self._point_in_direction(head, straight_direction)
        point_right = self._point_in_direction(head, right_direction)
        point_left = self._point_in_direction(head, left_direction)

        danger_straight = self._is_danger(point_straight)
        danger_right = self._is_danger(point_right)
        danger_left = self._is_danger(point_left)

        food_x, food_y = self.food if self.food is not None else (-1, -1)
        head_x, head_y = head

        state = [
            int(danger_straight),
            int(danger_right),
            int(danger_left),

            int(self.direction == Direction.LEFT),
            int(self.direction == Direction.RIGHT),
            int(self.direction == Direction.UP),
            int(self.direction == Direction.DOWN),

            int(food_x < head_x),
            int(food_x > head_x),
            int(food_y < head_y),
            int(food_y > head_y),
        ]

        return np.array(state, dtype=np.float32)

    def get_advanced_state(self) -> np.ndarray:
        """
        Richer state for Q-learning v2.

        Includes all 11 simple features plus:
        - danger within two steps (straight / right / left)
        - open-space bucket per move
        - food reachable after each move
        - tail reachable after each move
        - snake length bucket
        """
        simple = self.get_simple_state().tolist()
        head = self.snake[0]

        from snake_rl.core import board_analysis

        move_analysis = board_analysis.analyse_all_moves(
            snake=self.snake,
            food=self.food,
            current_direction=self.direction,
            width=self.width,
            height=self.height,
        )

        extra: List[int] = []

        for action in (Action.STRAIGHT, Action.RIGHT, Action.LEFT):
            extra.append(
                int(
                    board_analysis.is_danger_two_steps(
                        head=head,
                        current_direction=self.direction,
                        action=action,
                        snake=self.snake,
                        width=self.width,
                        height=self.height,
                    )
                )
            )

        for action in (Action.STRAIGHT, Action.RIGHT, Action.LEFT):
            extra.append(move_analysis[action]["open_space_bucket"])

        for action in (Action.STRAIGHT, Action.RIGHT, Action.LEFT):
            extra.append(int(move_analysis[action]["food_reachable"]))

        for action in (Action.STRAIGHT, Action.RIGHT, Action.LEFT):
            extra.append(int(move_analysis[action]["tail_reachable"]))

        extra.append(
            board_analysis.bucket_length(len(self.snake), self.width, self.height)
        )

        return np.array(simple + extra, dtype=np.float32)

    def get_move_analysis(self) -> Dict:
        """
        Per-move board analysis for UI / debugging.

        Returns analyse_all_moves() output for the current position.
        """
        from snake_rl.core import board_analysis

        return board_analysis.analyse_all_moves(
            snake=self.snake,
            food=self.food,
            current_direction=self.direction,
            width=self.width,
            height=self.height,
        )

    def render_ascii(self) -> None:
        """
        Simple terminal renderer for debugging.
        H = head
        S = snake body
        F = food
        . = empty space
        """
        snake_set = set(self.snake)

        for y in range(self.height):
            row = []
            for x in range(self.width):
                point = (x, y)

                if point == self.snake[0]:
                    row.append("H")
                elif point in snake_set:
                    row.append("S")
                elif point == self.food:
                    row.append("F")
                else:
                    row.append(".")
            print(" ".join(row))

        print(f"Score: {self.score} | Steps: {self.steps}")
        print()

    def _apply_action(self, action: Action) -> None:
        """
        Updates direction based on relative action.
        """
        if action == Action.STRAIGHT:
            return

        if action == Action.RIGHT:
            self.direction = Direction((int(self.direction) + 1) % 4)

        elif action == Action.LEFT:
            self.direction = Direction((int(self.direction) - 1) % 4)

        else:
            raise ValueError(f"Unknown action: {action}")

    def _next_head(self) -> Point:
        """
        Calculates where the head will move next.
        """
        head = self.snake[0]
        return self._point_in_direction(head, self.direction)

    def _point_in_direction(self, point: Point, direction: Direction) -> Point:
        dx, dy = DIRECTION_VECTORS[direction]
        return point[0] + dx, point[1] + dy

    def _hits_wall(self, point: Point) -> bool:
        x, y = point
        return x < 0 or x >= self.width or y < 0 or y >= self.height

    def _is_danger(self, point: Point) -> bool:
        """
        Checks if a move would be dangerous.

        We exclude the last tail cell because it usually moves away.
        """
        return self._hits_wall(point) or point in self.snake[:-1]

    def _spawn_food(self) -> Optional[Point]:
        """
        Places food on an empty cell.
        Returns None if the board is full.
        """
        snake_cells = set(self.snake)

        empty_cells = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in snake_cells
        ]

        if not empty_cells:
            return None

        return self.rng.choice(empty_cells)

    def _distance_to_food(self, point: Point) -> int:
        """
        Manhattan distance from a point to the food.
        """
        if self.food is None:
            return 0

        return abs(point[0] - self.food[0]) + abs(point[1] - self.food[1])
