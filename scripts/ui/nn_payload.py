"""Build live JSON payloads for the neural-network HTML visualization."""

from typing import Any, Dict, List, Optional

import numpy as np

from snake_rl.agents.dqn_agent import DQNAgent
from snake_rl.agents.q_learning_agent import QLearningAgent
from snake_rl.core.metrics import AgentMetrics
from snake_rl.envs.snake_env import Action, Direction, SnakeEnv
from ui.agent_data import agent_key_from_pygame
from ui.model_info import DEFAULT_MODEL_FILES, format_training_steps

STATE_LABELS = [
    "danger_straight", "danger_right", "danger_left",
    "dir_left", "dir_right", "dir_up", "dir_down",
    "food_left", "food_right", "food_up", "food_down",
]

STATE_LABELS_DISPLAY = [
    "Danger straight", "Danger right", "Danger left",
    "Moving left", "Moving right", "Moving up", "Moving down",
    "Food left", "Food right", "Food up", "Food down",
]

ACTION_LABELS = ["STRAIGHT", "RIGHT", "LEFT"]
TOP_HIDDEN = 16


def _direction_name(direction: Direction) -> str:
    return Direction(direction).name


def _top_neurons(values: List[float], k: int = TOP_HIDDEN) -> List[Dict]:
    indexed = [(i, float(v)) for i, v in enumerate(values)]
    indexed.sort(key=lambda x: x[1], reverse=True)
    return [{"index": i, "value": v} for i, v in indexed[:k]]


def build_live_payload(
    *,
    step_id: int,
    tick: int,
    agent: Any,
    agent_key: int,
    learning_mode: str,
    env: SnakeEnv,
    episode: Dict,
    metrics: AgentMetrics,
    decision_state: np.ndarray,
    trained_this_tick: bool = False,
) -> Optional[Dict]:
    """Build one streaming snapshot for Q-Learning or DQN agents."""
    agent_num = agent_key_from_pygame(agent_key)
    if agent_num not in (3, 4):
        return None

    decision = episode.get("agent_decision") or {}
    q_values = decision.get("q_values")
    if q_values is None and hasattr(agent, "get_q_values"):
        q_values = agent.get_q_values(decision_state)

    best_action = decision.get("best_action")
    chosen_action = episode.get("last_action")
    if best_action is not None and hasattr(best_action, "name"):
        best_name = best_action.name
    else:
        best_name = ACTION_LABELS[int(np.argmax(q_values))] if q_values else "STRAIGHT"
    if chosen_action is not None and hasattr(chosen_action, "name"):
        chosen_name = chosen_action.name
        chosen_index = int(chosen_action)
    else:
        chosen_name = best_name
        chosen_index = int(np.argmax(q_values)) if q_values else 0

    state_vec = [float(x) for x in decision_state.tolist()]
    epsilon = float(getattr(agent, "epsilon", 0.0))

    payload: Dict[str, Any] = {
        "step_id": step_id,
        "tick": tick,
        "timestamp_ms": tick,
        "agent": {
            "type": "dqn" if isinstance(agent, DQNAgent) else "q_learning",
            "name": agent.name,
            "key": agent_num,
            "learning_mode": learning_mode,
            "training": learning_mode == "training",
        },
        "board": {
            "width": env.width,
            "height": env.height,
            "snake": [list(p) for p in env.snake],
            "food": list(env.food) if env.food else None,
            "direction": _direction_name(env.direction),
            "score": env.score,
            "steps": env.steps,
            "fill_pct": round(100 * len(env.snake) / max(env.width * env.height, 1), 1),
        },
        "state": {
            "vector": state_vec,
            "labels": STATE_LABELS,
            "labels_display": STATE_LABELS_DISPLAY,
        },
        "decision": {
            "q_values": [float(v) for v in (q_values or [0.0, 0.0, 0.0])],
            "q_labels": ACTION_LABELS,
            "best_action": best_name,
            "chosen_action": chosen_name,
            "chosen_action_index": chosen_index,
            "chosen_action_number": chosen_index + 1,
            "action_count": len(ACTION_LABELS),
            "explored": bool(decision.get("explored", False)),
            "reason": decision.get("reason", ""),
        },
        "step_result": {
            "reward": float(episode.get("last_reward", 0.0)),
            "total_reward": float(episode.get("total_reward", 0.0)),
            "done": bool(episode.get("done", False)),
            "reason": episode.get("reason", "running"),
        },
        "training": {
            "epsilon": epsilon,
            "epsilon_pct": int(round(epsilon * 100)),
            "trained_this_tick": trained_this_tick,
        },
        "session": {
            "episode": metrics.episode_number,
            "best_score": metrics.best_score(),
            "avg_score": round(metrics.average_score(), 2),
        },
        "move": {
            "episode_step": env.steps,
            "live_step_id": step_id,
            "episode_number": metrics.episode_number,
        },
        "timeline": _timeline_phase(episode, learning_mode, trained_this_tick),
    }

    model_info = getattr(agent, "model_info", {}) or {}
    payload["model"] = {
        "file": model_info.get("path") or DEFAULT_MODEL_FILES.get(agent_num, "—"),
        "full_path": model_info.get("full_path"),
        "training_steps": model_info.get("training_steps"),
        "training_steps_display": format_training_steps(model_info.get("training_steps")),
        "saved_at": model_info.get("saved_at"),
        "loaded": bool(model_info.get("loaded", False)),
    }

    if isinstance(agent, DQNAgent):
        acts = agent.forward_with_activations(decision_state)
        payload["training"].update({
            "training_step_count": agent.training_step_count,
            "gradient_step_count": agent.gradient_step_count,
            "last_loss": agent.last_loss,
            "replay_size": len(agent.replay_memory),
            "replay_capacity": agent.replay_memory.memory.maxlen,
            "warmup_complete": agent.training_step_count > agent.warmup_steps,
        })
        payload["network"] = {
            "architecture": [11, 128, 128, 3],
            "activations": {
                "input": acts["input"],
                "hidden1_top": _top_neurons(acts["hidden1"]),
                "hidden2_top": _top_neurons(acts["hidden2"]),
                "output": acts["output"],
            },
        }
    elif isinstance(agent, QLearningAgent):
        state_key = agent.state_key_for(decision_state)
        table_index = agent.state_table_index(state_key)
        in_pkl = agent.state_in_loaded_table(state_key)
        payload["q_table"] = {
            "size": agent.table_size(),
            "loaded_size": agent.loaded_table_size(),
        }
        payload["q_lookup"] = {
            "state_key": list(state_key),
            "state_key_display": _format_state_key(state_key),
            "in_pkl": in_pkl,
            "table_index": table_index,
            "table_index_display": (
                f"{table_index} / {agent.loaded_table_size()}"
                if table_index is not None
                else f"new (not in .pkl · {agent.loaded_table_size()} saved states)"
            ),
            "chosen_q": float((q_values or [0.0, 0.0, 0.0])[chosen_index]),
            "best_action_index": int(np.argmax(q_values)) if q_values else 0,
        }
        payload["training"].update({
            "learning_rate": agent.learning_rate,
            "gamma": agent.discount_factor,
        })

    return payload


def _format_state_key(state_key: tuple) -> str:
    return "(" + ", ".join(str(int(x)) for x in state_key) + ")"


def _timeline_phase(episode: Dict, learning_mode: str, trained_this_tick: bool) -> str:
    if episode.get("done"):
        return "done"
    if trained_this_tick and learning_mode == "training":
        return "learn"
    if episode.get("last_action") is not None:
        return "act"
    return "think"
