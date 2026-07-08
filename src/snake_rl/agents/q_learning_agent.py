import pickle
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from snake_rl.agents.base_agent import BaseAgent
from snake_rl.envs.snake_env import Action, SnakeEnv


StateKey = Tuple[int, ...]


class QLearningAgent(BaseAgent):
    """
    First real learning agent.

    It learns a table:

        state -> [value for STRAIGHT, value for RIGHT, value for LEFT]

    Higher value means:
        "I believe this action is better in this state."
    """

    name = "Q-Learning Agent"

    def __init__(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.9,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.05,
        seed: Optional[int] = None,
    ):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.rng = random.Random(seed)

        # q_table maps state -> action values
        # Example:
        # (0, 1, 0, ...) -> [0.0, 0.0, 0.0]
        self.q_table: Dict[StateKey, List[float]] = defaultdict(
            lambda: [0.0, 0.0, 0.0]
        )

        self.last_decision: Dict = {
            "q_values": [0.0, 0.0, 0.0],
            "best_action": Action.STRAIGHT,
            "chosen_action": Action.STRAIGHT,
            "explored": False,
            "reason": "waiting for first move",
            "state_key": (),
        }

        self.model_info: Dict = {
            "path": None,
            "training_steps": None,
            "saved_at": None,
            "loaded": False,
        }
        self._loaded_state_keys: Set[StateKey] = set()

        # Epsilon restored when switching back from autonomous to training mode.
        self._training_epsilon: float = self.min_epsilon

    def choose_action(self, state: np.ndarray, env: Optional[SnakeEnv] = None) -> Action:
        state_key = self._state_to_key(state)
        q_values = list(self.q_table[state_key])
        best_action_index = int(np.argmax(q_values))
        best_action = Action(best_action_index)

        # Explore: try a random action
        if self.rng.random() < self.epsilon:
            action = self.rng.choice([
                Action.STRAIGHT,
                Action.RIGHT,
                Action.LEFT,
            ])
            self.last_decision = {
                "q_values": q_values,
                "best_action": best_action,
                "chosen_action": action,
                "explored": True,
                "reason": "random exploration (epsilon)",
                "state_key": state_key,
            }
            return action

        # Exploit: use the best learned action
        self.last_decision = {
            "q_values": q_values,
            "best_action": best_action,
            "chosen_action": best_action,
            "explored": False,
            "reason": "highest Q-value (greedy policy)",
            "state_key": state_key,
        }

        return best_action

    def learn(
        self,
        state: np.ndarray,
        action: Action,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        state_key = self._state_to_key(state)
        next_state_key = self._state_to_key(next_state)

        action_index = int(action)

        old_q = self.q_table[state_key][action_index]

        if done:
            target = reward
        else:
            best_future_q = max(self.q_table[next_state_key])
            target = reward + self.discount_factor * best_future_q

        new_q = old_q + self.learning_rate * (target - old_q)

        self.q_table[state_key][action_index] = new_q

    def reset(self) -> None:
        """
        Called at the start of a new episode.

        We decay epsilon here so the agent explores less over time.
        """
        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay,
        )

    def get_q_values(self, state: np.ndarray) -> List[float]:
        """
        Helper for UI/debugging.
        Shows current learned values for:
        [STRAIGHT, RIGHT, LEFT]
        """
        state_key = self._state_to_key(state)
        return self.q_table[state_key]

    def table_size(self) -> int:
        return len(self.q_table)

    def lookup_state(self, state: np.ndarray) -> List[float]:
        return list(self.get_q_values(state))

    def state_key_for(self, state: np.ndarray) -> StateKey:
        return self._state_to_key(state)

    def state_in_loaded_table(self, state_key: StateKey) -> bool:
        return state_key in self._loaded_state_keys

    def state_table_index(self, state_key: StateKey) -> Optional[int]:
        """1-based index of this state among keys saved in the loaded .pkl."""
        if state_key not in self._loaded_state_keys:
            return None
        return sorted(self._loaded_state_keys).index(state_key) + 1

    def loaded_table_size(self) -> int:
        return len(self._loaded_state_keys)

    def save(self, path: str, training_steps: Optional[int] = None) -> None:
        """
        Save the Q-learning agent memory to disk.
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        saved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        data = {
            "q_table": dict(self.q_table),
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon_decay": self.epsilon_decay,
            "min_epsilon": self.min_epsilon,
            "training_steps": training_steps,
            "saved_at": saved_at,
        }

        with open(save_path, "wb") as file:
            pickle.dump(data, file)

        self.model_info = {
            "path": save_path.name,
            "full_path": str(save_path),
            "training_steps": training_steps,
            "saved_at": saved_at,
            "loaded": True,
        }
        self._loaded_state_keys = set(self.q_table.keys())

    def load(self, path: str) -> None:
        """
        Load Q-learning agent memory from disk.
        """
        load_path = Path(path)

        with open(load_path, "rb") as file:
            data = pickle.load(file)

        self.q_table.clear()

        for state_key, q_values in data["q_table"].items():
            self.q_table[state_key] = q_values

        self.epsilon = data.get("epsilon", self.epsilon)
        self.learning_rate = data.get("learning_rate", self.learning_rate)
        self.discount_factor = data.get("discount_factor", self.discount_factor)
        self.epsilon_decay = data.get("epsilon_decay", self.epsilon_decay)
        self.min_epsilon = data.get("min_epsilon", self.min_epsilon)
        self._training_epsilon = self.epsilon

        self.model_info = {
            "path": load_path.name,
            "full_path": str(load_path),
            "training_steps": data.get("training_steps"),
            "saved_at": data.get("saved_at"),
            "loaded": True,
        }
        self._loaded_state_keys = set(data["q_table"].keys())

    def set_autonomous_mode(self) -> None:
        """Best-Q mode: no randomness, pure learned policy."""
        self.epsilon = 0.0

    def set_training_mode(self) -> None:
        """Training mode: small exploration, Q-table can keep updating."""
        self.epsilon = max(self.min_epsilon, self._training_epsilon)

    def _state_to_key(self, state: np.ndarray) -> StateKey:
        """
        Convert numpy array state into a tuple so it can be used as a dictionary key.

        Example:
        np.array([0, 1, 0]) -> (0, 1, 0)
        """
        return tuple(int(x) for x in state.tolist())
