"""
Deep Q-Network (DQN) agent for Snake.

Uses:
- policy_net  = current learner (updates every step)
- target_net  = stable teacher (copied from policy periodically)
- replay_memory = random batch of past experiences
"""

import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from snake_rl.agents.base_agent import BaseAgent
from snake_rl.agents.dqn_model import ACTION_SIZE, DQNModel, STATE_SIZE
from snake_rl.core.replay_memory import ReplayMemory
from snake_rl.envs.snake_env import Action, SnakeEnv


class DQNAgent(BaseAgent):
    name = "DQN Agent"

    def __init__(
        self,
        state_size: int = STATE_SIZE,
        action_size: int = ACTION_SIZE,
        learning_rate: float = 0.0005,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        min_epsilon: float = 0.05,
        epsilon_decay_steps: int = 500_000,
        batch_size: int = 128,
        memory_size: int = 100_000,
        target_update_every: int = 5_000,
        warmup_steps: int = 10_000,
        train_every: int = 4,
        seed: Optional[int] = None,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.min_epsilon = min_epsilon
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.target_update_every = target_update_every
        self.warmup_steps = warmup_steps
        self.train_every = train_every
        self.training_step_count = 0
        self.gradient_step_count = 0
        self.last_loss: Optional[float] = None

        self.rng = random.Random(seed)
        self.device = torch.device("cpu")
        self.training = True

        # Policy network = current learner
        self.policy_net = DQNModel(state_size, action_size).to(self.device)
        # Target network = stable teacher for Q-targets
        self.target_net = DQNModel(state_size, action_size).to(self.device)
        self.update_target_network()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        # Loss = how wrong the network prediction was (SmoothL1 is more stable than MSE)
        self.loss_fn = nn.SmoothL1Loss()

        # Replay memory = past experiences sampled randomly during training
        self.replay_memory = ReplayMemory(capacity=memory_size)

        self.last_decision: Dict = {
            "q_values": [0.0, 0.0, 0.0],
            "best_action": Action.STRAIGHT,
            "chosen_action": Action.STRAIGHT,
            "explored": False,
            "reason": "waiting for first move",
        }

        self.model_info: Dict = {
            "path": None,
            "training_steps": None,
            "saved_at": None,
            "loaded": False,
        }

    def update_epsilon(self) -> None:
        """Linear epsilon decay based on environment learn steps."""
        if not self.training:
            self.epsilon = 0.0
            return

        progress = min(1.0, self.training_step_count / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start - progress * (self.epsilon_start - self.min_epsilon)
        self.epsilon = max(self.min_epsilon, self.epsilon)

    def choose_action(self, state: np.ndarray, env: Optional[SnakeEnv] = None) -> Action:
        q_values = self.get_q_values(state)
        best_action_index = int(np.argmax(q_values))
        best_action = Action(best_action_index)

        # Epsilon = exploration randomness (only in training mode)
        if self.training and self.rng.random() < self.epsilon:
            action = self.rng.choice([Action.STRAIGHT, Action.RIGHT, Action.LEFT])
            self.last_decision = {
                "q_values": q_values,
                "best_action": best_action,
                "chosen_action": action,
                "explored": True,
                "reason": "random exploration (epsilon)",
            }
            return action

        self.last_decision = {
            "q_values": q_values,
            "best_action": best_action,
            "chosen_action": best_action,
            "explored": False,
            "reason": "highest Q-value (neural policy)",
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
        if not self.training:
            return

        self.replay_memory.push(state, action, reward, next_state, done)
        self.training_step_count += 1
        self.update_epsilon()

        if (
            self.training_step_count > self.warmup_steps
            and self.training_step_count % self.train_every == 0
            and len(self.replay_memory) >= self.batch_size
        ):
            self.train_step()

    def train_step(self) -> None:
        """Sample a mini-batch and update the policy network."""
        batch = self.replay_memory.sample(self.batch_size)

        states = torch.tensor(
            np.array([exp.state for exp in batch], dtype=np.float32),
            device=self.device,
        )
        actions = torch.tensor(
            [int(exp.action) for exp in batch],
            dtype=torch.long,
            device=self.device,
        )
        rewards = torch.tensor(
            [exp.reward for exp in batch],
            dtype=torch.float32,
            device=self.device,
        )
        next_states = torch.tensor(
            np.array([exp.next_state for exp in batch], dtype=np.float32),
            device=self.device,
        )
        dones = torch.tensor(
            [float(exp.done) for exp in batch],
            dtype=torch.float32,
            device=self.device,
        )

        # Current Q-values from policy network
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Next Q-values from target network (stable teacher)
            next_q = self.target_net(next_states).max(dim=1)[0]
            targets = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(current_q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.last_loss = float(loss.item())
        self.gradient_step_count += 1

        if self.gradient_step_count % self.target_update_every == 0:
            self.update_target_network()

    def update_target_network(self) -> None:
        """Copy policy weights into target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_q_values(self, state: np.ndarray) -> List[float]:
        self.policy_net.eval()
        with torch.no_grad():
            tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(tensor).squeeze(0).tolist()
        self.policy_net.train()
        return q_values

    def forward_with_activations(self, state: np.ndarray) -> Dict[str, List[float]]:
        """Run policy forward and return per-layer activations for visualization."""
        self.policy_net.eval()
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32, device=self.device)
            h1 = torch.relu(self.policy_net.network[0](x))
            h2 = torch.relu(self.policy_net.network[2](h1))
            out = self.policy_net.network[4](h2)
        self.policy_net.train()
        return {
            "input": [float(v) for v in x.tolist()],
            "hidden1": [float(v) for v in h1.tolist()],
            "hidden2": [float(v) for v in h2.tolist()],
            "output": [float(v) for v in out.tolist()],
        }

    def export_weights_json(self) -> Dict:
        """Serialize policy network weights for the live HTML viz."""
        sd = self.policy_net.state_dict()
        out = {}
        for key, tensor in sd.items():
            out[key] = tensor.detach().cpu().tolist()
        return {
            "architecture": [self.state_size, 128, 128, self.action_size],
            "state_dict": out,
        }

    def reset(self) -> None:
        """Called at the start of a new episode."""
        pass

    def set_autonomous_mode(self) -> None:
        """Evaluation mode: no exploration, no learning."""
        self.training = False
        self.epsilon = 0.0

    def set_training_mode(self) -> None:
        """Training mode: explore and learn from replay memory."""
        self.training = True
        self.update_epsilon()

    @property
    def learning_enabled(self) -> bool:
        """Backward-compatible alias used by the dashboard."""
        return self.training

    def save(self, path: str, training_steps: Optional[int] = None) -> None:
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        saved_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        data = {
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "epsilon_start": self.epsilon_start,
            "min_epsilon": self.min_epsilon,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "training_step_count": self.training_step_count,
            "gradient_step_count": self.gradient_step_count,
            "warmup_steps": self.warmup_steps,
            "train_every": self.train_every,
            "gamma": self.gamma,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "target_update_every": self.target_update_every,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "training_steps": training_steps or self.training_step_count,
            "saved_at": saved_at,
        }

        torch.save(data, save_path)

        self.model_info = {
            "path": save_path.name,
            "full_path": str(save_path),
            "training_steps": data["training_steps"],
            "saved_at": saved_at,
            "loaded": True,
        }

    def load(self, path: str) -> None:
        load_path = Path(path)
        data = torch.load(load_path, map_location=self.device)

        self.policy_net.load_state_dict(data["policy_net"])
        self.target_net.load_state_dict(data["target_net"])
        self.optimizer.load_state_dict(data["optimizer"])

        self.epsilon = data.get("epsilon", self.epsilon)
        self.epsilon_start = data.get("epsilon_start", self.epsilon_start)
        self.min_epsilon = data.get("min_epsilon", self.min_epsilon)
        self.epsilon_decay_steps = data.get("epsilon_decay_steps", self.epsilon_decay_steps)
        self.training_step_count = data.get("training_step_count", 0)
        self.gradient_step_count = data.get("gradient_step_count", 0)
        self.warmup_steps = data.get("warmup_steps", self.warmup_steps)
        self.train_every = data.get("train_every", self.train_every)
        self.gamma = data.get("gamma", self.gamma)
        self.batch_size = data.get("batch_size", self.batch_size)
        self.learning_rate = data.get("learning_rate", self.learning_rate)
        self.target_update_every = data.get("target_update_every", self.target_update_every)

        self.model_info = {
            "path": load_path.name,
            "full_path": str(load_path),
            "training_steps": data.get("training_steps"),
            "saved_at": data.get("saved_at"),
            "loaded": True,
        }
