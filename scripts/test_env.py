#!/usr/bin/env python3
"""Smoke test for the Snake RL environment."""

import sys
from pathlib import Path

# Allow running without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from snake_rl.envs import SnakeEnv
from snake_rl.envs.snake_env import Action


def main() -> None:
    env = SnakeEnv(width=12, height=12, seed=42)
    state = env.reset()
    print("Initial state shape:", state.shape)
    env.render_ascii()

    total_reward = 0.0
    for step in range(20):
        action = Action(step % 3)
        result = env.step(action)
        total_reward += result.reward
        print(
            f"Step {step + 1}: action={action.name}, reward={result.reward:.2f}, "
            f"done={result.done}, info={result.info}"
        )
        if result.done:
            env.render_ascii()
            break

    print(f"\nTotal reward over episode: {total_reward:.2f}")
    print("Environment smoke test passed.")


if __name__ == "__main__":
    main()
