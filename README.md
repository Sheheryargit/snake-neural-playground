# Snake Neural Playground

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)

Train, battle, and **watch AI agents play Snake** — with a pygame dashboard and a live browser visualization that syncs every game step to the neural network or Q-table lookup.

**Repository:** [github.com/Sheheryargit/snake-neural-playground](https://github.com/Sheheryargit/snake-neural-playground)

---

## Overview

Snake Neural Playground is a reinforcement learning lab built around a lightweight Snake environment. It ships with multiple agent types (from random baselines to deep Q-networks), headless training scripts, saved model checkpoints, and an interactive arena for comparing agents in real time.

The standout feature is the **live neural visualization**: open a browser panel that streams each decision from the dashboard — animating DQN forward passes or showing exactly which `.pkl` Q-table row and action the agent used.

---

## Features

- **Snake environment** — configurable board size, 11-dim (simple) or 24-dim (advanced) state vectors
- **7 playable agents** — Random, Greedy, Q-Learning, DQN, Hamiltonian, Hybrid Hamiltonian, Manual
- **Pygame dashboard** — speed control, pause, board resize, battle mode, presentation mode
- **Training / Autonomous toggle** — explore with ε-greedy or run a frozen greedy policy
- **Live browser viz** — SSE stream at `http://127.0.0.1:8765/` with step-synced animations
- **Q-table inspector** — model file, table row index, state key, and chosen action (1–3) per step
- **Saved models** — pre-trained Q-Learning and DQN checkpoints included

---

## Quick Start

### Prerequisites

- Python 3.9+ (3.10+ recommended)
- macOS / Linux / Windows

### Installation

```bash
git clone https://github.com/Sheheryargit/snake-neural-playground.git
cd snake-neural-playground

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

### Run the dashboard

```bash
python scripts/watch_agents.py
```

| Key | Action |
|-----|--------|
| `1`–`6` | Select agent (Random, Greedy, Q-Learn, DQN, Hamilton, Hybrid) |
| `H` | Play manually |
| `N` | Open live neural / Q-table visualization in browser |
| `M` | Toggle Training ↔ Autonomous (Q-Learn & DQN) |
| `B` | Battle mode |
| `SPACE` | Pause |
| `R` | Reset episode |
| `S` | Save current model |
| `TAB` | Simple ↔ Advanced UI |
| `+` / `-` | Adjust game speed |

### Open the live visualization

1. Press `3` (Q-Learning) or `4` (DQN)
2. Press `N` or click **Neural Net** in the control bar
3. Browser opens at `http://127.0.0.1:8765/`

- **DQN** — animated forward pass: inputs → hidden layers → Q-values
- **Q-Learning** — live `.pkl` lookup: model file, table row, state key, action

---

## Agents

| # | Agent | Type | Best score (12×12) | Notes |
|---|-------|------|-------------------|-------|
| 1 | Random | Baseline | ~15 | Uniform random turns |
| 2 | Greedy Food | Rule-based | ~68 | Chases food, one-step lookahead |
| 3 | Q-Learning | Tabular RL | ~45 | `q_learning_10m.pkl` — 10M training steps |
| 4 | DQN | Deep RL | ~44 | `dqn_11_state.pt` — 11→128→128→3 MLP |
| 5 | Hamiltonian | Planner | 141 | Full-board Hamiltonian cycle |
| 6 | Hybrid Hamiltonian | Planner | 141 | Cycle + safe food shortcuts |
| 7 | Manual | Human | — | Arrow keys / WASD (`H`) |

---

## State Representation

Agents use **relative** actions (`STRAIGHT`, `RIGHT`, `LEFT`) and an 11-feature **simple state** (used by Q-Learning and DQN):

| Index | Feature |
|-------|---------|
| 0–2 | Danger straight / right / left |
| 3–6 | Current direction (left, right, up, down) |
| 7–10 | Food direction (left, right, up, down) |

An **advanced 24-feature state** is also available (open space, reachability, trap detection) for future agents.

---

## Reward Structure

| Event | Reward |
|-------|--------|
| Eat food | `+10` |
| Move closer to food | `+0.05` |
| Move farther from food | `-0.05` |
| Per-step penalty | `-0.01` |
| Wall / self collision | `-10` |
| Fill entire board | `+100` |

---

## Training

### Q-Learning (headless)

```bash
python scripts/train_q_learning.py --steps 10000000
```

| Hyperparameter | Value |
|----------------|-------|
| Learning rate (α) | 0.1 |
| Discount (γ) | 0.9 |
| ε decay | 0.995 per episode |
| Min ε | 0.05 |

Output: `models/q_learning_10m.pkl`

### DQN (headless)

```bash
python scripts/train_dqn.py --steps 1000000
```

| Hyperparameter | Value |
|----------------|-------|
| Architecture | 11 → 128 → 128 → 3 (ReLU) |
| Optimizer | Adam, lr = 5×10⁻⁴ |
| Replay buffer | 100,000 |
| Batch size | 128 |
| γ | 0.95 |
| Target network update | every 5,000 gradient steps |

Output: `models/dqn_11_state.pt`

### Inspect a saved Q-table

```bash
python scripts/inspect_pkl.py models/q_learning_10m.pkl
```

---

## Project Structure

```
snake-neural-playground/
├── src/snake_rl/
│   ├── agents/          # All agent implementations
│   ├── core/            # Metrics, replay memory, pathfinding
│   └── envs/            # SnakeEnv
├── scripts/
│   ├── watch_agents.py  # Main pygame dashboard
│   ├── train_q_learning.py
│   ├── train_dqn.py
│   └── ui/              # Dashboard layout, neural bridge, payloads
├── web/
│   └── neural-network-live.html   # Live browser visualization
├── models/              # Saved checkpoints and training logs
├── requirements.txt
└── pyproject.toml
```

---

## Architecture

```
┌─────────────────────┐     SSE / HTTP      ┌──────────────────────────┐
│  watch_agents.py    │ ──────────────────► │  neural-network-live.html │
│  (pygame dashboard) │   localhost:8765    │  (browser visualization)  │
└─────────┬───────────┘                     └──────────────────────────┘
          │
          ▼
┌─────────────────────┐
│  SnakeEnv           │  11-dim state, relative actions
│  QLearningAgent     │  Q-table lookup → 3 actions
│  DQNAgent           │  MLP forward pass → Q-values
└─────────────────────┘
```

Each game step publishes a JSON payload (`step_id`, board, state vector, Q-values, activations, `.pkl` lookup metadata) over Server-Sent Events.

---

## Development

### Run environment smoke test

```bash
python scripts/test_env.py
```

### Key dependencies

- [NumPy](https://numpy.org/) — state vectors and Q-table math
- [Pygame](https://www.pygame.org/) — dashboard UI
- [PyTorch](https://pytorch.org/) — DQN training and inference

---

## Roadmap

- [ ] CI pipeline for training smoke tests
- [ ] GIF demo in README
- [ ] Docker / one-click setup
- [ ] Advanced-state DQN agent

---

## Contributing

Issues and pull requests are welcome. Please open an issue first for large changes.

---

## Author

**Sheheryar** — [github.com/Sheheryargit](https://github.com/Sheheryargit)

---

## Acknowledgments

Built as a hands-on RL project exploring tabular Q-learning, deep Q-networks, and real-time decision visualization for the classic Snake game.
