"""Model metadata and benchmark helpers for Q-Learning / DQN display."""

from typing import Any, Dict, Optional

from snake_rl.core.metrics import AgentMetrics
from snake_rl.envs.snake_env import SnakeEnv
from ui.agent_data import AgentProfile

RL_MODEL_AGENT_KEYS = {3, 4}

DEFAULT_MODEL_FILES = {
    3: "q_learning_10m.pkl",
    4: "dqn_11_state.pt",
}


def is_rl_model_agent(agent_num: int) -> bool:
    return agent_num in RL_MODEL_AGENT_KEYS


def theoretical_max_score(width: int, height: int) -> int:
    """Max food eaten on a full board (snake starts with 3 cells)."""
    return max(0, width * height - 3)


def format_training_steps(steps: Optional[int]) -> str:
    if not steps:
        return "—"
    if steps >= 1_000_000:
        whole = steps / 1_000_000
        return f"{whole:.0f}M" if whole == int(whole) else f"{whole:.1f}M"
    if steps >= 1_000:
        whole = steps / 1_000
        return f"{whole:.0f}k" if whole == int(whole) else f"{whole:.1f}k"
    return str(steps)


def get_agent_epsilon(agent: Any) -> float:
    return float(getattr(agent, "epsilon", 0.0))


def build_rl_mode_state(agent: Any, learning_mode: str) -> Dict[str, Any]:
    """Structured training/autonomous state for visual UI."""
    is_training = learning_mode == "training"
    epsilon = get_agent_epsilon(agent)
    epsilon_pct = int(round(epsilon * 100))

    if is_training:
        mode_label = "TRAINING"
        mode_color = "warning"
        subtitle = (
            f"ε = {epsilon:.3f}  ·  {epsilon_pct}% random moves"
            if epsilon > 0
            else "ε = 0.000  ·  greedy policy (min exploration)"
        )
        detail = "Explores randomly and updates the model each step. Press M to switch to Autonomous."
        learns = True
    else:
        mode_label = "AUTONOMOUS"
        mode_color = "success"
        subtitle = "ε = 0.000  ·  pure learned policy (no exploration)"
        detail = "Runs the best learned policy only — no random moves, no learning. Press M to resume Training."
        learns = False

    return {
        "is_training": is_training,
        "mode_label": mode_label,
        "mode_color": mode_color,
        "epsilon": epsilon,
        "epsilon_pct": epsilon_pct,
        "epsilon_display": f"{epsilon:.3f}",
        "subtitle": subtitle,
        "detail": detail,
        "learns": learns,
        "toggle_hint": "M",
    }


def build_model_results(
    agent: Any,
    profile: AgentProfile,
    metrics: AgentMetrics,
    env: SnakeEnv,
    learning_mode: str,
) -> Dict[str, str]:
    """Build display strings for the model results panel."""
    info = getattr(agent, "model_info", {}) or {}
    mode_state = build_rl_mode_state(agent, learning_mode)
    model_file = info.get("path") or DEFAULT_MODEL_FILES.get(profile.key, "—")
    training_steps = format_training_steps(info.get("training_steps"))
    board_label = f"{env.width}×{env.height}"
    max_on_board = theoretical_max_score(env.width, env.height)

    trained_best = profile.benchmark_best
    trained_avg = profile.benchmark_avg
    trained_best_s = f"{trained_best:.0f}" if trained_best is not None else "—"
    trained_avg_s = f"{trained_avg:.1f}" if trained_avg is not None else "—"

    session_best = metrics.best_score()
    session_avg = metrics.average_score()
    session_avg_s = f"{session_avg:.1f}" if metrics.episode_number else "—"

    saved_at = info.get("saved_at")
    loaded = info.get("loaded", False)

    return {
        "model_file": model_file,
        "training_steps": training_steps,
        "trained_best": trained_best_s,
        "trained_avg": trained_avg_s,
        "benchmark_note": profile.benchmark_note,
        "board_label": board_label,
        "max_on_board": str(max_on_board),
        "session_best": str(session_best),
        "session_avg": session_avg_s,
        "mode_label": mode_state["mode_label"],
        "mode_subtitle": mode_state["subtitle"],
        "mode_detail": mode_state["detail"],
        "epsilon_display": mode_state["epsilon_display"],
        "epsilon_pct": str(mode_state["epsilon_pct"]),
        "is_training": mode_state["is_training"],
        "saved_at": saved_at or "",
        "loaded": "yes" if loaded else "no",
        "agent_name": profile.short_name,
    }

