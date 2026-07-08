from typing import Optional

import numpy as np

from snake_rl.agents.hamiltonian_agent import HamiltonianCycleAgent, Point
from snake_rl.envs.snake_env import Action, SnakeEnv


class HybridHamiltonianAgent(HamiltonianCycleAgent):
    """
    Hybrid full-solve Snake agent.

    Main idea:
    - Pure Hamiltonian is the safety backbone.
    - Shortcuts are allowed only if they preserve Hamiltonian body order.
    - If no shortcut is safe, follow the normal Hamiltonian cycle.

    This is NOT learning.
    This is deterministic planning with safety rules.
    """

    name = "Hybrid Hamiltonian Agent"

    def __init__(
        self,
        min_tail_gap_for_shortcut: int = 4,
    ):
        super().__init__()

        self.min_tail_gap_for_shortcut = min_tail_gap_for_shortcut

        self.shortcut_moves = 0
        self.fallback_moves = 0
        self.cycle_moves = 0

        self.debug_info = {
            "mode": "HYBRID_HAMILTONIAN",
            "head_index": None,
            "tail_index": None,
            "food_index": None,
            "chosen_index": None,
            "next_index": None,
            "target_next_point": None,
            "cycle_length": 0,
            "move_type": "none",
            "shortcut_moves": 0,
            "cycle_moves": 0,
            "fallback_moves": 0,
            "reason": "not initialized",
        }

    def reset(self) -> None:
        self.debug_info = {
            "mode": "HYBRID_HAMILTONIAN",
            "head_index": None,
            "tail_index": None,
            "food_index": None,
            "chosen_index": None,
            "next_index": None,
            "target_next_point": None,
            "cycle_length": len(self.cycle),
            "move_type": "none",
            "shortcut_moves": self.shortcut_moves,
            "cycle_moves": self.cycle_moves,
            "fallback_moves": self.fallback_moves,
            "reason": "agent reset",
        }

    def reset_env(self, env: SnakeEnv) -> np.ndarray:
        """
        Use the Hamiltonian-friendly reset from the parent class,
        then update debug mode to HYBRID.
        """
        state = super().reset_env(env)

        head = env.snake[0]
        tail = env.snake[-1]

        head_index = self.point_to_index.get(head)
        tail_index = self.point_to_index.get(tail)
        food_index = self.point_to_index.get(env.food) if env.food is not None else None

        self.debug_info = {
            "mode": "HYBRID_HAMILTONIAN",
            "head_index": head_index,
            "tail_index": tail_index,
            "food_index": food_index,
            "chosen_index": None,
            "next_index": None,
            "target_next_point": None,
            "cycle_length": len(self.cycle),
            "move_type": "reset",
            "shortcut_moves": self.shortcut_moves,
            "cycle_moves": self.cycle_moves,
            "fallback_moves": self.fallback_moves,
            "reason": "environment reset onto Hamiltonian cycle",
        }

        return state

    def choose_action(self, state: np.ndarray, env: Optional[SnakeEnv] = None) -> Action:
        if env is None:
            return Action.STRAIGHT

        self._ensure_cycle(env.width, env.height)

        head = env.snake[0]
        tail = env.snake[-1]

        if head not in self.point_to_index or tail not in self.point_to_index:
            self.fallback_moves += 1
            self._set_debug(
                env=env,
                chosen_point=None,
                move_type="fallback",
                reason="head or tail not on Hamiltonian cycle",
            )
            return self._safe_fallback_action(env)

        pure_action = self._pure_hamiltonian_action(env)

        if pure_action is None:
            self.fallback_moves += 1
            self._set_debug(
                env=env,
                chosen_point=None,
                move_type="fallback",
                reason="pure Hamiltonian action unavailable",
            )
            return self._safe_fallback_action(env)

        # If there is no food for some reason, stay on the cycle.
        if env.food is None or env.food not in self.point_to_index:
            self.cycle_moves += 1
            chosen_point = self._next_point_after_action(head, env.direction, pure_action)
            self._set_debug(
                env=env,
                chosen_point=chosen_point,
                move_type="cycle",
                reason="no food available, following cycle",
            )
            return pure_action

        best_action = pure_action
        best_point = self._next_point_after_action(head, env.direction, pure_action)
        best_score = self._cycle_distance(
            self.point_to_index[best_point],
            self.point_to_index[env.food],
        )
        best_move_type = "cycle"

        for action in [Action.STRAIGHT, Action.RIGHT, Action.LEFT]:
            if not self._is_action_safe(env, action):
                continue

            candidate_point = self._next_point_after_action(
                head=head,
                direction=env.direction,
                action=action,
            )

            if candidate_point not in self.point_to_index:
                continue

            if not self._preserves_hamiltonian_safety(env, candidate_point):
                continue

            candidate_index = self.point_to_index[candidate_point]
            food_index = self.point_to_index[env.food]

            candidate_score = self._cycle_distance(candidate_index, food_index)

            # Lower score means food is sooner on the Hamiltonian cycle.
            if candidate_score < best_score:
                best_action = action
                best_point = candidate_point
                best_score = candidate_score
                best_move_type = "shortcut"

        if best_move_type == "shortcut":
            self.shortcut_moves += 1
            self._set_debug(
                env=env,
                chosen_point=best_point,
                move_type="shortcut",
                reason="safe shortcut reduces cycle distance to food",
            )
            return best_action

        self.cycle_moves += 1
        self._set_debug(
            env=env,
            chosen_point=best_point,
            move_type="cycle",
            reason="no safe shortcut, following Hamiltonian cycle",
        )
        return best_action

    # --------------------------------------------------
    # Core hybrid logic
    # --------------------------------------------------

    def _pure_hamiltonian_action(self, env: SnakeEnv) -> Optional[Action]:
        head = env.snake[0]

        if head not in self.point_to_index:
            return None

        head_index = self.point_to_index[head]
        next_index = (head_index + 1) % len(self.cycle)
        target_next_point = self.cycle[next_index]

        action = self._action_to_reach_point(
            current_direction=env.direction,
            head=head,
            target=target_next_point,
        )

        if action is None:
            return None

        if not self._is_action_safe(env, action):
            return None

        return action

    def _preserves_hamiltonian_safety(
        self,
        env: SnakeEnv,
        candidate_point: Point,
    ) -> bool:
        """
        The shortcut is safe only if:
        - candidate is ahead of the head on the cycle
        - candidate does not catch/overtake the tail
        - if candidate eats food, it leaves extra space because the tail will not move
        """
        head = env.snake[0]
        tail = env.snake[-1]

        head_index = self.point_to_index[head]
        tail_index = self.point_to_index[tail]
        candidate_index = self.point_to_index[candidate_point]

        distance_to_candidate = self._cycle_distance(head_index, candidate_index)
        distance_to_tail = self._cycle_distance(head_index, tail_index)

        # Candidate must be ahead of the head.
        if distance_to_candidate <= 0:
            return False

        # Pure next-cycle move is always allowed if physically safe.
        if distance_to_candidate == 1:
            return True

        will_eat = env.food is not None and candidate_point == env.food
        will_fill_board = will_eat and len(env.snake) + 1 >= len(self.cycle)

        if will_fill_board:
            return True

        gap_after_candidate = distance_to_tail - distance_to_candidate

        # If eating, tail does not move, so we need extra room.
        required_gap = 2 if will_eat else 1

        # For shortcuts, be more conservative.
        required_gap = max(required_gap, self.min_tail_gap_for_shortcut)

        return gap_after_candidate >= required_gap

    def _cycle_distance(self, from_index: int, to_index: int) -> int:
        """
        Forward distance around the Hamiltonian cycle.
        """
        return (to_index - from_index) % len(self.cycle)

    # --------------------------------------------------
    # Debug
    # --------------------------------------------------

    def _set_debug(
        self,
        env: SnakeEnv,
        chosen_point: Optional[Point],
        move_type: str,
        reason: str,
    ) -> None:
        head = env.snake[0]
        tail = env.snake[-1]

        head_index = self.point_to_index.get(head)
        tail_index = self.point_to_index.get(tail)

        food_index = None
        if env.food is not None:
            food_index = self.point_to_index.get(env.food)

        chosen_index = None
        if chosen_point is not None:
            chosen_index = self.point_to_index.get(chosen_point)

        next_index = None
        target_next_point = None

        if head_index is not None:
            next_index = (head_index + 1) % len(self.cycle)
            target_next_point = self.cycle[next_index]

        self.debug_info = {
            "mode": "HYBRID_HAMILTONIAN",
            "head_index": head_index,
            "tail_index": tail_index,
            "food_index": food_index,
            "chosen_index": chosen_index,
            "next_index": next_index,
            "target_next_point": target_next_point,
            "cycle_length": len(self.cycle),
            "move_type": move_type,
            "shortcut_moves": self.shortcut_moves,
            "cycle_moves": self.cycle_moves,
            "fallback_moves": self.fallback_moves,
            "reason": reason,
        }
