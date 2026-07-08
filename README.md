# Snake RL Agent

A reinforcement learning project for training an agent to play Snake. The environment is implemented in pure Python with no UI or graphics dependencies.

## Project structure

```
snake-rl-agent/
├── src/
│   └── snake_rl/
│       ├── __init__.py
│       └── envs/
│           ├── __init__.py
│           └── snake_env.py
├── scripts/
│   └── test_env.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Quick test

Run a short random-action episode to verify the environment:

```bash
python scripts/test_env.py
```

## Environment API

`SnakeEnv` follows a Gymnasium-like interface:

- `reset()` → `(observation, info)`
- `step(action)` → `(observation, reward, terminated, truncated, info)`

**Actions:** `0=up`, `1=right`, `2=down`, `3=left`

**Observation:** flat `int8` array of length `grid_size * grid_size` where each cell is:

| Value | Meaning     |
|-------|-------------|
| 0     | empty       |
| 1     | snake body  |
| 2     | snake head  |
| 3     | food        |

**Rewards:**

- `+10` for eating food
- `-10` for wall or self collision
- `-0.01` per step (encourages efficiency)

Use `render()` to print an ASCII grid for debugging.

## Example

```python
from snake_rl.envs import SnakeEnv

env = SnakeEnv(grid_size=10, seed=0)
obs, info = env.reset()

done = False
while not done:
    action = 1  # replace with your agent's policy
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
```
