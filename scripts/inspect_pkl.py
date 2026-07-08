#!/usr/bin/env python3
"""Pretty-print a saved Q-learning .pkl file (safe read-only)."""

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a Q-learning pickle file")
    parser.add_argument(
        "path",
        nargs="?",
        default="models/q_learning_10m.pkl",
        help="Path to .pkl file",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    with open(path, "rb") as f:
        data = pickle.load(f)

    print("=" * 60)
    print(f"File: {path}")
    print("=" * 60)

    if not isinstance(data, dict):
        print(f"Type: {type(data)}")
        print(data)
        return

    q_table = data.get("q_table", {})
    print(f"training_steps: {data.get('training_steps')}")
    print(f"saved_at:       {data.get('saved_at')}")
    print(f"epsilon:        {data.get('epsilon')}")
    print(f"learning_rate:  {data.get('learning_rate')}")
    print(f"gamma:          {data.get('discount_factor')}")
    print(f"eps_decay:      {data.get('epsilon_decay')}")
    print(f"min_epsilon:    {data.get('min_epsilon')}")
    print(f"Q-table states: {len(q_table)}")
    print()

    if q_table:
        all_q = [v for row in q_table.values() for v in row]
        print(f"Q min:  {min(all_q):.4f}")
        print(f"Q max:  {max(all_q):.4f}")
        print(f"Q mean: {sum(all_q) / len(all_q):.4f}")
        print()
        print("Sample states (first 5):")
        for i, (state_key, values) in enumerate(q_table.items()):
            if i >= 5:
                break
            print(f"  {state_key} -> {[round(v, 3) for v in values]}")


if __name__ == "__main__":
    main()
