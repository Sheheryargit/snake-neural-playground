#!/usr/bin/env python3
"""Compare simple vs advanced state vectors from SnakeEnv."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from snake_rl.envs.snake_env import (
    ADVANCED_STATE_SIZE,
    SIMPLE_STATE_SIZE,
    SnakeEnv,
)


def main() -> None:
    simple_env = SnakeEnv(width=12, height=12, seed=42, state_mode="simple")
    advanced_env = SnakeEnv(width=12, height=12, seed=42, state_mode="advanced")

    simple_state = simple_env.reset()
    advanced_state = advanced_env.reset()

    print("State mode comparison")
    print("=" * 50)
    print(f"Simple state size:   {simple_state.shape[0]} (expected {SIMPLE_STATE_SIZE})")
    print(f"Advanced state size: {advanced_state.shape[0]} (expected {ADVANCED_STATE_SIZE})")
    print()
    print("Simple state (first 11 features):")
    print(simple_state)
    print()
    print("Advanced state (24 features):")
    print(advanced_state)
    print()

    analysis = advanced_env.get_move_analysis()
    print("Move analysis (advanced env):")
    for action, info in analysis.items():
        print(
            f"  {action.name:7} danger={info['dangerous']} "
            f"open_bucket={info['open_space_bucket']} "
            f"food={info['food_reachable']} tail={info['tail_reachable']}"
        )

    assert simple_state.shape[0] == SIMPLE_STATE_SIZE
    assert advanced_state.shape[0] == ADVANCED_STATE_SIZE
    assert list(advanced_state[:SIMPLE_STATE_SIZE]) == list(simple_state)

    print("\nAdvanced state test passed.")


if __name__ == "__main__":
    main()
