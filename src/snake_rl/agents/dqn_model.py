"""
Small neural network for DQN.

Input:  11-value Snake state (same as tabular Q-learning)
Output: 3 Q-values for STRAIGHT, RIGHT, LEFT
"""

import torch
import torch.nn as nn


STATE_SIZE = 11
ACTION_SIZE = 3


class DQNModel(nn.Module):
    """
    Policy network: state -> predicted Q-value for each action.

    Architecture:
        11 -> 128 -> ReLU -> 128 -> ReLU -> 3
    """

    def __init__(self, state_size: int = STATE_SIZE, action_size: int = ACTION_SIZE):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)
