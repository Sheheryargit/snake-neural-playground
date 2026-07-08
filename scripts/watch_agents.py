#!/usr/bin/env python3
"""
Snake RL Arena — dashboard with demo/presentation features.

Controls:
  H         Play yourself       P         Presentation mode
  1-6       Select AI agent     B         Battle mode
  TAB       Simple / Advanced   F         Fullscreen
  Arrows    Steer (manual)      G         Toggle ghost path
  SPACE     Pause               O         Toggle decision arrows
  R         Reset               V         Replay best run
  M         Training toggle     E         Export PNG snapshot
  S         Save model          N         Open neural net live view
  +/-       Fine-tune speed     , . ; '   Board size
  [ / ]     Halve / double speed
"""


import argparse
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pygame

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from snake_rl.agents.dqn_agent import DQNAgent
from snake_rl.agents.greedy_agent import GreedyFoodAgent
from snake_rl.agents.hamiltonian_agent import HamiltonianCycleAgent
from snake_rl.agents.hybrid_hamiltonian_agent import HybridHamiltonianAgent
from snake_rl.agents.manual_agent import ManualPlayer
from snake_rl.agents.q_learning_agent import QLearningAgent
from snake_rl.agents.random_agent import RandomAgent
from snake_rl.core.metrics import AgentMetrics
from snake_rl.envs.snake_env import Action, Direction, SnakeEnv
from ui.agent_data import AGENT_PROFILES, MANUAL_AGENT_KEY, agent_key_from_pygame
from ui.board_config import DEFAULT_BOARD_SIZE, clamp_board_size, hamiltonian_compatible, normalize_for_hamiltonian
from ui.dashboard import DashboardUI
from ui.demo_features import DEMO_PRESETS, DemoSession
from ui.nn_bridge import NeuralNetworkBridge
from ui.nn_payload import build_live_payload
from ui.theme import draw_background

WINDOW_WIDTH = 1440
WINDOW_HEIGHT = 900
ENV_SEED = 42
FPS_DEFAULT = 8
FPS_MANUAL = 10
FPS_MIN = 1
FPS_MAX = 2000

Q_MODEL_PATH = "models/q_learning_10m.pkl"
DQN_MODEL_PATH = "models/dqn_11_state.pt"
Q_AGENT_KEY = pygame.K_3
DQN_AGENT_KEY = pygame.K_4
LEARNING_AGENT_KEYS = {Q_AGENT_KEY, DQN_AGENT_KEY}
LEARNING_MODE_TRAINING = "training"
LEARNING_MODE_AUTONOMOUS = "autonomous"
HAMILTONIAN_AGENT_KEYS = {pygame.K_5, pygame.K_6}

DIRECTION_KEYS = {
    pygame.K_UP: Direction.UP, pygame.K_DOWN: Direction.DOWN,
    pygame.K_LEFT: Direction.LEFT, pygame.K_RIGHT: Direction.RIGHT,
    pygame.K_w: Direction.UP, pygame.K_s: Direction.DOWN,
    pygame.K_a: Direction.LEFT, pygame.K_d: Direction.RIGHT,
}


def make_agents():
    q_agent = QLearningAgent(seed=1)
    if Path(Q_MODEL_PATH).exists():
        q_agent.load(str(Q_MODEL_PATH))
        print(f"Loaded Q model from {Q_MODEL_PATH}")
    q_agent.set_training_mode()

    dqn_agent = DQNAgent(seed=1)
    if Path(DQN_MODEL_PATH).exists():
        dqn_agent.load(str(DQN_MODEL_PATH))
        print(f"Loaded DQN model from {DQN_MODEL_PATH}")
    dqn_agent.set_training_mode()

    return {
        pygame.K_1: RandomAgent(seed=1),
        pygame.K_2: GreedyFoodAgent(),
        Q_AGENT_KEY: q_agent,
        DQN_AGENT_KEY: dqn_agent,
        pygame.K_5: HamiltonianCycleAgent(),
        pygame.K_6: HybridHamiltonianAgent(),
        MANUAL_AGENT_KEY: ManualPlayer(),
    }


def is_learning_agent(agent) -> bool:
    return isinstance(agent, (QLearningAgent, DQNAgent))


def is_manual_agent(agent) -> bool:
    return isinstance(agent, ManualPlayer)


def prepare_board_size(width: int, height: int, selected_key: int) -> Tuple[int, int]:
    w, h = clamp_board_size(width), clamp_board_size(height)
    if selected_key in HAMILTONIAN_AGENT_KEYS:
        w, h = normalize_for_hamiltonian(w, h)
    return w, h


def get_board_warning(selected_key: int, width: int, height: int) -> str:
    if selected_key in HAMILTONIAN_AGENT_KEYS and not hamiltonian_compatible(width, height):
        return "Hamilton/Hybrid need an even width or height"
    return ""


def direction_after_action(direction, action):
    if action == Action.STRAIGHT:
        return direction
    if action == Action.RIGHT:
        return Direction((int(direction) + 1) % 4)
    return Direction((int(direction) - 1) % 4)


def analyse_action(env, action):
    nd = direction_after_action(env.direction, action)
    np_ = env._point_in_direction(env.snake[0], nd)
    return {
        "action": action,
        "next_point": np_,
        "danger": env._is_danger(np_),
        "distance": 0 if env.food is None else abs(np_[0] - env.food[0]) + abs(np_[1] - env.food[1]),
    }


def get_action_analysis(env):
    return {a: analyse_action(env, a) for a in [Action.STRAIGHT, Action.RIGHT, Action.LEFT]}


def reset_episode(env, agent, metrics, learning_mode=LEARNING_MODE_TRAINING):
    if isinstance(agent, QLearningAgent) and learning_mode == LEARNING_MODE_TRAINING:
        agent.reset()
    elif not is_learning_agent(agent):
        agent.reset()
    metrics.start_episode()
    state = agent.reset_env(env) if hasattr(agent, "reset_env") else env.reset()
    return {
        "state": state, "done": False, "reason": "running", "last_action": Action.STRAIGHT,
        "last_reward": 0.0, "total_reward": 0.0, "death_restart_at": None,
        "agent_decision": getattr(agent, "last_decision", None) if is_learning_agent(agent) else None,
    }


def apply_learning_mode(agent, learning_mode: str) -> None:
    if not is_learning_agent(agent):
        return
    if learning_mode == LEARNING_MODE_AUTONOMOUS:
        agent.set_autonomous_mode()
    else:
        agent.set_training_mode()


def adjust_fps(fps: int, delta: int) -> int:
    return max(FPS_MIN, min(FPS_MAX, fps + delta))


def select_agent(key, agents, metrics_by_key, learning_mode, fps):
    agent = agents[key]
    metrics = metrics_by_key[key]
    if is_learning_agent(agent):
        apply_learning_mode(agent, learning_mode)
    new_fps = FPS_MANUAL if is_manual_agent(agent) else fps
    return key, agent, metrics, new_fps


def apply_board_resize(bw, bh, dw, dh, selected_key, agent, metrics, learning_mode):
    nw, nh = prepare_board_size(bw + dw, bh + dh, selected_key)
    if nw == bw and nh == bh:
        return bw, bh, None, None, get_board_warning(selected_key, nw, nh)
    env = SnakeEnv(width=nw, height=nh, seed=ENV_SEED)
    ep = reset_episode(env, agent, metrics, learning_mode)
    return nw, nh, env, ep, get_board_warning(selected_key, nw, nh)


def ensure_env_for_agent(bw, bh, env, selected_key, agent, metrics, learning_mode):
    nw, nh = prepare_board_size(bw, bh, selected_key)
    if nw != env.width or nh != env.height:
        env = SnakeEnv(width=nw, height=nh, seed=ENV_SEED)
    ep = reset_episode(env, agent, metrics, learning_mode)
    return nw, nh, env, ep, get_board_warning(selected_key, nw, nh)


def setup_battle(demo, board_width, board_height, agents, metrics_by_key, learning_mode, key_a=None, key_b=None):
    """Initialize or refresh battle with chosen fighters."""
    if key_a is None:
        key_a = demo.battle_key_a if demo.battle_key_a else pygame.K_2
    if key_b is None:
        key_b = demo.battle_key_b if demo.battle_key_b else pygame.K_1
    if key_a == key_b:
        key_b = pygame.K_1 if key_a != pygame.K_1 else pygame.K_2
    demo.battle_key_a = key_a
    demo.battle_key_b = key_b
    demo.battle_env = SnakeEnv(width=board_width, height=board_height, seed=ENV_SEED + 1)
    demo.battle_metrics = metrics_by_key[demo.battle_key_b]
    demo.battle_episode = reset_episode(demo.battle_env, agents[demo.battle_key_b], demo.battle_metrics, learning_mode)


def apply_battle_fighter(demo, key, side, env, board_width, board_height, agents, metrics_by_key, learning_mode):
    """Assign fighter A (left) or B (right) and reset that board."""
    if side == "a":
        if demo.battle_key_b and key == demo.battle_key_b:
            return None
        demo.battle_key_a = key
        return reset_episode(env, agents[key], metrics_by_key[key], learning_mode)
    if demo.battle_key_a and key == demo.battle_key_a:
        alt = pygame.K_1 if key != pygame.K_1 else pygame.K_2
        key = alt
    demo.battle_key_b = key
    if demo.battle_env is None or demo.battle_env.width != board_width or demo.battle_env.height != board_height:
        demo.battle_env = SnakeEnv(width=board_width, height=board_height, seed=ENV_SEED + 1)
    demo.battle_metrics = metrics_by_key[key]
    demo.battle_episode = reset_episode(demo.battle_env, agents[key], demo.battle_metrics, learning_mode)
    return None


def restart_battle_envs(demo, env, board_width, board_height, agents, metrics_by_key, learning_mode):
    """Reset both battle games after fighter change or R key."""
    ep_a = reset_episode(env, agents[demo.battle_key_a], metrics_by_key[demo.battle_key_a], learning_mode)
    if demo.battle_env is None or demo.battle_env.width != board_width or demo.battle_env.height != board_height:
        demo.battle_env = SnakeEnv(width=board_width, height=board_height, seed=ENV_SEED + 1)
    demo.battle_metrics = metrics_by_key[demo.battle_key_b]
    ep_b = reset_episode(demo.battle_env, agents[demo.battle_key_b], demo.battle_metrics, learning_mode)
    demo.battle_episode = ep_b
    return ep_a


def apply_preset(preset_name, agents, metrics_by_key, learning_mode, fps):
    preset = DEMO_PRESETS[preset_name]
    key = preset.agent_key
    agent = agents[key]
    metrics = metrics_by_key[key]
    if is_learning_agent(agent):
        learning_mode = preset.learning_mode
        apply_learning_mode(agent, learning_mode)
    fps = preset.fps
    bw, bh = prepare_board_size(preset.width, preset.height, key)
    env = SnakeEnv(width=bw, height=bh, seed=ENV_SEED)
    ep = reset_episode(env, agent, metrics, learning_mode)
    return key, agent, metrics, bw, bh, env, ep, fps, learning_mode


def step_agent(env, agent, episode, metrics, selected_key, learning_mode, demo, now, manual_mode, frame=0):
    old_state = episode["state"]
    episode["decision_state"] = old_state
    grad_before = getattr(agent, "gradient_step_count", 0) if isinstance(agent, DQNAgent) else 0
    action = agent.choose_action(old_state, env)
    result = env.step(action)

    trained_this_tick = False
    if selected_key in LEARNING_AGENT_KEYS and learning_mode == LEARNING_MODE_TRAINING:
        agent.learn(old_state, action, result.reward, result.state, result.done)
        if isinstance(agent, DQNAgent):
            trained_this_tick = agent.gradient_step_count > grad_before
        elif isinstance(agent, QLearningAgent):
            trained_this_tick = True

    metrics.record_step(action, result.reward)
    episode.update(state=result.state, last_action=action, last_reward=result.reward,
                   total_reward=episode["total_reward"] + result.reward, done=result.done, reason=result.info["reason"])
    if is_learning_agent(agent):
        episode["agent_decision"] = agent.last_decision

    if NeuralNetworkBridge.running() and selected_key in LEARNING_AGENT_KEYS:
        publish_neural_snapshot(
            demo, frame, agent, selected_key, learning_mode, env, episode, metrics,
            decision_state=old_state, trained_this_tick=trained_this_tick,
        )

    demo.replay.record(env)
    demo.effects.on_step(env, result.reward, result.done, result.info["reason"])

    if result.reward >= 10:
        demo.sound.play("eat")
        demo.set_narration_event("ate_food", now, 1200)
    if result.done:
        metrics.end_episode(result.info["score"], result.info["steps"], result.info["reason"])
        demo.leaderboard.record(selected_key, result.info["score"], env.width, env.height, result.info["reason"])
        if demo.replay.save_best_if_better(env):
            demo.sound.play("highscore")
        if result.info["reason"] == "collision":
            demo.sound.play("death")
            demo.set_narration_event("collision", now)
        elif result.info["reason"] == "board_full":
            demo.sound.play("win")
            demo.set_narration_event("board_full", now, 3000)
        elif result.info["reason"] == "timeout":
            demo.set_narration_event("timeout", now)
        episode["death_restart_at"] = None if manual_mode else now + 1500
    return episode


def parse_args():
    p = argparse.ArgumentParser(description="Snake RL Arena dashboard")
    p.add_argument("--width", type=int, default=DEFAULT_BOARD_SIZE)
    p.add_argument("--height", type=int, default=DEFAULT_BOARD_SIZE)
    return p.parse_args()


def publish_neural_snapshot(
    demo: DemoSession,
    frame: int,
    agent,
    selected_key: int,
    learning_mode: str,
    env: SnakeEnv,
    episode: dict,
    metrics: AgentMetrics,
    *,
    decision_state=None,
    trained_this_tick: bool = False,
) -> None:
    """Publish one live payload to the neural-network bridge."""
    if not NeuralNetworkBridge.running() or selected_key not in LEARNING_AGENT_KEYS:
        return
    state = decision_state
    if state is None:
        state = episode.get("decision_state")
    if state is None:
        state = episode.get("state")
    if state is None:
        return
    demo.nn_step += 1
    payload = build_live_payload(
        step_id=demo.nn_step,
        tick=frame,
        agent=agent,
        agent_key=selected_key,
        learning_mode=learning_mode,
        env=env,
        episode=episode,
        metrics=metrics,
        decision_state=state,
        trained_this_tick=trained_this_tick,
    )
    if payload:
        NeuralNetworkBridge.publish(payload)
        demo.nn_live = True


def open_neural_viz(
    demo: DemoSession,
    agents: dict,
    env: SnakeEnv,
    episode: dict,
    selected_key: int,
    selected_agent,
    selected_metrics: AgentMetrics,
    learning_mode: str,
    frame: int,
    now: int,
) -> None:
    """Start localhost bridge and open the live HTML visualization."""
    url = NeuralNetworkBridge.start()
    dqn = agents.get(DQN_AGENT_KEY)
    if isinstance(dqn, DQNAgent):
        NeuralNetworkBridge.set_weights_provider(lambda: dqn.export_weights_json())
    if not demo.nn_viz_opened:
        webbrowser.open(url)
        demo.nn_viz_opened = True
    publish_neural_snapshot(
        demo, frame, selected_agent, selected_key, learning_mode, env, episode, selected_metrics,
    )
    demo.nn_live = NeuralNetworkBridge.running()
    demo.set_narration_event("neural_viz", now)


def main():
    args = parse_args()
    pygame.init()
    demo = DemoSession()
    demo.sound.init()
    demo.sound.enabled = demo.sound_enabled

    board_width, board_height = prepare_board_size(args.width, args.height, pygame.K_1)
    env = SnakeEnv(width=board_width, height=board_height, seed=ENV_SEED)
    agents = make_agents()
    metrics_by_key = {key: AgentMetrics(max_history=200) for key in agents}

    selected_key = pygame.K_1
    selected_agent = agents[selected_key]
    selected_metrics = metrics_by_key[selected_key]

    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    dashboard = DashboardUI(WINDOW_WIDTH, WINDOW_HEIGHT)
    fullscreen = False

    running = True
    paused = False
    fps = FPS_DEFAULT
    learning_mode = LEARNING_MODE_TRAINING
    frame = 0
    board_warning = ""
    episode = reset_episode(env, selected_agent, selected_metrics, learning_mode)
    demo.effects.on_reset(env)
    replay_tick = 0

    while running:
        now = pygame.time.get_ticks()
        pulse = frame * 0.08
        manual_mode = is_manual_agent(selected_agent)
        demo.update_narration(selected_key, learning_mode, now)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE:
                if not fullscreen:
                    screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
                    dashboard.width, dashboard.height = screen.get_size()

            elif event.type == pygame.MOUSEMOTION:
                dashboard.handle_motion(event.pos)
                demo.mouse_pos = event.pos

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                click = dashboard.handle_click(event.pos, selected_key, battle_mode=demo.battle_mode)
                if not click:
                    continue
                action = click["action"]
                if action == "toggle_mode":
                    dashboard.ui_mode = "advanced" if dashboard.ui_mode == "simple" else "simple"
                elif action == "select_battle_a":
                    key = click["key"]
                    if key == demo.battle_key_b:
                        key = pygame.K_2 if demo.battle_key_b != pygame.K_2 else pygame.K_1
                    selected_key, selected_agent, selected_metrics, fps = select_agent(key, agents, metrics_by_key, learning_mode, fps)
                    demo.battle_key_a = selected_key
                    episode = apply_battle_fighter(demo, selected_key, "a", env, board_width, board_height, agents, metrics_by_key, learning_mode)
                    demo.effects.on_reset(env)
                elif action == "select_battle_b":
                    demo.battle_key_b = click["key"]
                    if demo.battle_key_b == demo.battle_key_a:
                        demo.battle_key_b = pygame.K_1 if demo.battle_key_a != pygame.K_1 else pygame.K_2
                    apply_battle_fighter(demo, demo.battle_key_b, "b", env, board_width, board_height, agents, metrics_by_key, learning_mode)
                elif action == "select_agent":
                    selected_key, selected_agent, selected_metrics, fps = select_agent(click["key"], agents, metrics_by_key, learning_mode, fps)
                    board_width, board_height, env, episode, board_warning = ensure_env_for_agent(board_width, board_height, env, selected_key, selected_agent, selected_metrics, learning_mode)
                    demo.effects.on_reset(env)
                    demo.replay.stop_replay()
                elif action == "select_tab":
                    dashboard.advanced_tab = click["tab"]
                elif action == "set_fps":
                    fps = click["fps"]
                elif action == "resize_board":
                    board_width, board_height, ne, nep, board_warning = apply_board_resize(board_width, board_height, click.get("dw", 0), click.get("dh", 0), selected_key, selected_agent, selected_metrics, learning_mode)
                    if ne:
                        env, episode = ne, nep
                        demo.effects.on_reset(env)
                elif action == "demo_preset":
                    selected_key, selected_agent, selected_metrics, board_width, board_height, env, episode, fps, learning_mode = apply_preset(click["preset"], agents, metrics_by_key, learning_mode, fps)
                    board_warning = get_board_warning(selected_key, board_width, board_height)
                    demo.effects.on_reset(env)
                elif action == "pause":
                    paused = not paused
                    if paused:
                        demo.set_narration_event("paused", now)
                elif action == "reset":
                    if demo.battle_mode:
                        episode = restart_battle_envs(demo, env, board_width, board_height, agents, metrics_by_key, learning_mode)
                    else:
                        episode = reset_episode(env, selected_agent, selected_metrics, learning_mode)
                    demo.effects.on_reset(env)
                    demo.replay.stop_replay()
                elif action == "replay":
                    if demo.replay.start_replay():
                        demo.set_narration_event("replay", now)
                        paused = False
                elif action == "demo":
                    demo.toggle_presentation()
                elif action == "battle":
                    demo.toggle_battle()
                    if demo.battle_mode:
                        demo.battle_key_a = selected_key
                        setup_battle(demo, board_width, board_height, agents, metrics_by_key, learning_mode, key_a=selected_key)
                        episode = reset_episode(env, agents[demo.battle_key_a], metrics_by_key[demo.battle_key_a], learning_mode)
                        demo.set_narration_event("battle", now)
                elif action == "export":
                    path = Path("exports") / f"snake_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    path.parent.mkdir(exist_ok=True)
                    pygame.image.save(screen, str(path))
                    demo.export_flash_until = now + 2000
                    print(f"Saved snapshot to {path}")
                elif action == "neural":
                    open_neural_viz(
                        demo, agents, env, episode, selected_key, selected_agent,
                        selected_metrics, learning_mode, frame, now,
                    )
                elif action == "toggle_fullscreen":
                    fullscreen = not fullscreen
                    screen = pygame.display.set_mode((0, 0) if fullscreen else (WINDOW_WIDTH, WINDOW_HEIGHT), pygame.FULLSCREEN if fullscreen else pygame.RESIZABLE)
                    dashboard.width, dashboard.height = screen.get_size()
                elif action == "toggle_sound":
                    demo.sound_enabled = not demo.sound_enabled
                    demo.sound.enabled = demo.sound_enabled
                elif action == "toggle_ghost":
                    demo.ghost_enabled = not demo.ghost_enabled
                elif action == "toggle_overlay":
                    demo.decision_overlay = not demo.decision_overlay

            elif event.type == pygame.KEYDOWN:
                if event.key in agents:
                    if demo.battle_mode:
                        side = "b" if pygame.key.get_mods() & pygame.KMOD_SHIFT else "a"
                        if side == "a":
                            selected_key, selected_agent, selected_metrics, fps = select_agent(event.key, agents, metrics_by_key, learning_mode, fps)
                            demo.battle_key_a = selected_key
                            episode = apply_battle_fighter(demo, selected_key, "a", env, board_width, board_height, agents, metrics_by_key, learning_mode)
                            demo.effects.on_reset(env)
                        else:
                            apply_battle_fighter(demo, event.key, "b", env, board_width, board_height, agents, metrics_by_key, learning_mode)
                    else:
                        selected_key, selected_agent, selected_metrics, fps = select_agent(event.key, agents, metrics_by_key, learning_mode, fps)
                        board_width, board_height, env, episode, board_warning = ensure_env_for_agent(board_width, board_height, env, selected_key, selected_agent, selected_metrics, learning_mode)
                        demo.effects.on_reset(env)
                elif event.key in DIRECTION_KEYS and manual_mode and not paused and not episode["done"] and not demo.replay.playing:
                    selected_agent.set_desired_direction(DIRECTION_KEYS[event.key], env.direction)
                elif event.key == pygame.K_TAB:
                    dashboard.ui_mode = "advanced" if dashboard.ui_mode == "simple" else "simple"
                elif event.key in (pygame.K_j, pygame.K_k) and dashboard.ui_mode == "advanced":
                    dashboard.advanced_tab = (dashboard.advanced_tab + (1 if event.key == pygame.K_k else -1)) % 5
                elif event.key == pygame.K_m and selected_key in LEARNING_AGENT_KEYS:
                    learning_mode = LEARNING_MODE_AUTONOMOUS if learning_mode == LEARNING_MODE_TRAINING else LEARNING_MODE_TRAINING
                    apply_learning_mode(selected_agent, learning_mode)
                    episode = reset_episode(env, selected_agent, selected_metrics, learning_mode)
                elif event.key == pygame.K_s and selected_key in (Q_AGENT_KEY, DQN_AGENT_KEY):
                    path = Q_MODEL_PATH if selected_key == Q_AGENT_KEY else DQN_MODEL_PATH
                    selected_agent.save(path)
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    if demo.battle_mode:
                        episode = restart_battle_envs(demo, env, board_width, board_height, agents, metrics_by_key, learning_mode)
                    else:
                        episode = reset_episode(env, selected_agent, selected_metrics, learning_mode)
                    demo.effects.on_reset(env)
                    demo.replay.stop_replay()
                elif event.key == pygame.K_p:
                    demo.toggle_presentation()
                elif event.key == pygame.K_b:
                    demo.toggle_battle()
                    if demo.battle_mode:
                        demo.battle_key_a = selected_key
                        setup_battle(demo, board_width, board_height, agents, metrics_by_key, learning_mode, key_a=selected_key)
                        episode = reset_episode(env, agents[demo.battle_key_a], metrics_by_key[demo.battle_key_a], learning_mode)
                elif event.key == pygame.K_f:
                    fullscreen = not fullscreen
                    screen = pygame.display.set_mode((0, 0) if fullscreen else (WINDOW_WIDTH, WINDOW_HEIGHT), pygame.FULLSCREEN if fullscreen else pygame.RESIZABLE)
                    dashboard.width, dashboard.height = screen.get_size()
                elif event.key == pygame.K_g:
                    demo.ghost_enabled = not demo.ghost_enabled
                elif event.key == pygame.K_o:
                    demo.decision_overlay = not demo.decision_overlay
                elif event.key == pygame.K_v:
                    demo.replay.start_replay()
                elif event.key == pygame.K_n:
                    open_neural_viz(
                        demo, agents, env, episode, selected_key, selected_agent,
                        selected_metrics, learning_mode, frame, now,
                    )
                elif event.key == pygame.K_e:
                    path = Path("exports") / f"snake_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    path.parent.mkdir(exist_ok=True)
                    pygame.image.save(screen, str(path))
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                    fps = adjust_fps(fps, 1)
                elif event.key == pygame.K_MINUS:
                    fps = adjust_fps(fps, -1)
                elif event.key == pygame.K_RIGHTBRACKET:
                    fps = adjust_fps(fps, max(1, fps // 2))
                elif event.key == pygame.K_LEFTBRACKET:
                    fps = adjust_fps(fps, fps)
                elif event.key == pygame.K_0:
                    fps = FPS_MAX
                elif event.key == pygame.K_COMMA:
                    board_width, board_height, ne, nep, board_warning = apply_board_resize(board_width, board_height, -1, 0, selected_key, selected_agent, selected_metrics, learning_mode)
                    if ne: env, episode = ne, nep
                elif event.key == pygame.K_PERIOD:
                    board_width, board_height, ne, nep, board_warning = apply_board_resize(board_width, board_height, 1, 0, selected_key, selected_agent, selected_metrics, learning_mode)
                    if ne: env, episode = ne, nep
                elif event.key == pygame.K_SEMICOLON:
                    board_width, board_height, ne, nep, board_warning = apply_board_resize(board_width, board_height, 0, -1, selected_key, selected_agent, selected_metrics, learning_mode)
                    if ne: env, episode = ne, nep
                elif event.key == pygame.K_QUOTE:
                    board_width, board_height, ne, nep, board_warning = apply_board_resize(board_width, board_height, 0, 1, selected_key, selected_agent, selected_metrics, learning_mode)
                    if ne: env, episode = ne, nep

        demo.effects.update()

        if demo.replay.playing:
            replay_tick += 1
            if replay_tick >= demo.replay.play_speed:
                replay_tick = 0
                demo.replay.advance()
        elif not paused and not demo.replay.playing:
            if episode["done"] and episode["death_restart_at"] is not None and now >= episode["death_restart_at"]:
                episode = reset_episode(env, selected_agent, selected_metrics, learning_mode)
                demo.effects.on_reset(env)

            if not episode["done"]:
                battle_key = demo.battle_key_a if demo.battle_mode else selected_key
                battle_agent = agents[battle_key]
                battle_metrics = metrics_by_key[battle_key]
                episode = step_agent(env, battle_agent, episode, battle_metrics, battle_key, learning_mode, demo, now, is_manual_agent(battle_agent), frame)

            if demo.battle_mode and demo.battle_env and not demo.replay.playing:
                bep = demo.battle_episode
                if bep["done"] and bep.get("death_restart_at") and now >= bep["death_restart_at"]:
                    demo.battle_episode = reset_episode(demo.battle_env, agents[demo.battle_key_b], demo.battle_metrics, learning_mode)
                elif not bep["done"]:
                    demo.battle_episode = step_agent(demo.battle_env, agents[demo.battle_key_b], bep, demo.battle_metrics, demo.battle_key_b, learning_mode, demo, now, False, frame)

        draw_background(screen)
        agent_num = agent_key_from_pygame(demo.battle_key_a if demo.battle_mode else selected_key) or 1
        agent_name = AGENT_PROFILES[agent_num].full_name
        action_analysis = get_action_analysis(env)

        dashboard.build_topbar_buttons(dashboard.ui_mode)
        dashboard.draw_topbar(screen, ui_mode=dashboard.ui_mode, fps=fps, paused=paused, agent_name=agent_name,
                              manual_mode=manual_mode, board_width=board_width, board_height=board_height, demo=demo,
                              agent=selected_agent, learning_mode=learning_mode, agent_num=agent_num)

        if dashboard.ui_mode == "simple":
            dashboard.draw_simple(screen, env=env, agent=selected_agent, metrics=selected_metrics, episode=episode, selected_key=selected_key,
                                  fps=fps, paused=paused, pulse=pulse, learning_mode=learning_mode,
                                  is_learning=is_learning_agent(selected_agent), board_warning=board_warning,
                                  demo=demo, action_analysis=action_analysis)
        else:
            metrics_by_num = {agent_key_from_pygame(k) or 0: m for k, m in metrics_by_key.items()}
            dashboard.draw_advanced(screen, env=env, agent=selected_agent, metrics=selected_metrics, episode=episode,
                                    selected_key=selected_key, fps=fps, paused=paused, pulse=pulse, learning_mode=learning_mode,
                                    all_metrics=metrics_by_num, state=episode["state"], action_analysis=action_analysis,
                                    board_warning=board_warning, demo=demo)

        if now < demo.export_flash_until:
            from ui.components import draw_text
            from ui.theme import THEME
            draw_text(screen, "Snapshot saved!", dashboard.width // 2, 80, 16, THEME.success, bold=True, center=True)

        pygame.display.flip()
        clock.tick(fps)
        frame += 1

    NeuralNetworkBridge.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
