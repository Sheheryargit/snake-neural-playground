#!/usr/bin/env python3
"""Capture dashboard and neural-viz screenshots for the README."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pygame

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "src"))

from snake_rl.core.metrics import AgentMetrics  # noqa: E402
from snake_rl.envs.snake_env import SnakeEnv  # noqa: E402
from ui.agent_data import AGENT_PROFILES, agent_key_from_pygame  # noqa: E402
from ui.dashboard import DashboardUI  # noqa: E402
from ui.demo_features import DemoSession  # noqa: E402
from ui.nn_bridge import NeuralNetworkBridge  # noqa: E402
from ui.theme import draw_background  # noqa: E402
from watch_agents import (  # noqa: E402
    DQN_AGENT_KEY,
    ENV_SEED,
    LEARNING_MODE_TRAINING,
    Q_AGENT_KEY,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
    get_action_analysis,
    is_learning_agent,
    make_agents,
    publish_neural_snapshot,
    reset_episode,
    step_agent,
)

OUT_DIR = ROOT / "docs" / "images"


def render_dashboard_frame(
    *,
    screen: pygame.Surface,
    dashboard: DashboardUI,
    env: SnakeEnv,
    agent,
    metrics: AgentMetrics,
    episode: dict,
    selected_key: int,
    demo: DemoSession,
    ui_mode: str = "simple",
    learning_mode: str = LEARNING_MODE_TRAINING,
    board_warning: str = "",
) -> None:
    dashboard.ui_mode = ui_mode
    draw_background(screen)
    agent_num = agent_key_from_pygame(selected_key) or 1
    agent_name = AGENT_PROFILES[agent_num].full_name
    action_analysis = get_action_analysis(env)
    dashboard.build_topbar_buttons(ui_mode)
    dashboard.draw_topbar(
        screen,
        ui_mode=ui_mode,
        fps=8,
        paused=False,
        agent_name=agent_name,
        manual_mode=False,
        board_width=env.width,
        board_height=env.height,
        demo=demo,
        agent=agent,
        learning_mode=learning_mode,
        agent_num=agent_num,
    )
    if ui_mode == "simple":
        dashboard.draw_simple(
            screen,
            env=env,
            agent=agent,
            metrics=metrics,
            episode=episode,
            selected_key=selected_key,
            fps=8,
            paused=False,
            pulse=0.0,
            learning_mode=learning_mode,
            is_learning=is_learning_agent(agent),
            board_warning=board_warning,
            demo=demo,
            action_analysis=action_analysis,
        )
    else:
        dashboard.draw_advanced(
            screen,
            env=env,
            agent=agent,
            metrics=metrics,
            episode=episode,
            selected_key=selected_key,
            fps=8,
            paused=False,
            pulse=0.0,
            learning_mode=learning_mode,
            all_metrics={agent_num: metrics},
            state=episode["state"],
            action_analysis=action_analysis,
            board_warning=board_warning,
            demo=demo,
        )


def simulate_game(agent_key: int, steps: int = 30):
    agents = make_agents()
    agent = agents[agent_key]
    metrics = AgentMetrics(max_history=200)
    env = SnakeEnv(width=12, height=12, seed=ENV_SEED)
    episode = reset_episode(env, agent, metrics, LEARNING_MODE_TRAINING)
    demo = DemoSession()
    demo.nn_live = True
    for i in range(steps):
        if episode["done"]:
            episode = reset_episode(env, agent, metrics, LEARNING_MODE_TRAINING)
        episode = step_agent(
            env, agent, episode, metrics, agent_key, LEARNING_MODE_TRAINING,
            demo, now=i * 100, manual_mode=False, frame=i + 1,
        )
    return env, agent, metrics, episode, demo


def capture_dashboard(agent_key: int, filename: str, *, steps: int = 35, ui_mode: str = "simple") -> Path:
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    dashboard = DashboardUI(WINDOW_WIDTH, WINDOW_HEIGHT)
    env, agent, metrics, episode, demo = simulate_game(agent_key, steps=steps)
    render_dashboard_frame(
        screen=screen,
        dashboard=dashboard,
        env=env,
        agent=agent,
        metrics=metrics,
        episode=episode,
        selected_key=agent_key,
        demo=demo,
        ui_mode=ui_mode,
    )
    out = OUT_DIR / filename
    pygame.image.save(screen, str(out))
    pygame.quit()
    print(f"Saved {out}")
    return out


def capture_neural_pages() -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit(
            "Playwright is required for neural viz screenshots.\n"
            "Install with: pip install playwright && playwright install chromium"
        ) from exc

    agents = make_agents()
    env = SnakeEnv(width=12, height=12, seed=ENV_SEED)
    metrics = AgentMetrics(max_history=200)
    demo = DemoSession()
    demo.nn_live = True

    captures = [
        (DQN_AGENT_KEY, agents[DQN_AGENT_KEY], "neural-dqn.png", "tabDqn"),
        (Q_AGENT_KEY, agents[Q_AGENT_KEY], "neural-qtable.png", "tabQ"),
    ]

    NeuralNetworkBridge.start()
    dqn = agents[DQN_AGENT_KEY]
    NeuralNetworkBridge.set_weights_provider(lambda: dqn.export_weights_json())

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 920})
        for agent_key, agent, filename, tab_id in captures:
            episode = reset_episode(env, agent, metrics, LEARNING_MODE_TRAINING)
            for i in range(28):
                if episode["done"]:
                    episode = reset_episode(env, agent, metrics, LEARNING_MODE_TRAINING)
                episode = step_agent(
                    env, agent, episode, metrics, agent_key, LEARNING_MODE_TRAINING,
                    demo, now=i * 100, manual_mode=False, frame=i + 1,
                )
            publish_neural_snapshot(
                demo, 99, agent, agent_key, LEARNING_MODE_TRAINING, env, episode, metrics,
            )
            page.goto("http://127.0.0.1:8765/", wait_until="load")
            page.wait_for_timeout(1500)
            if tab_id == "tabQ":
                page.click("#tabQ")
                page.wait_for_timeout(400)
            else:
                page.click("#tabDqn")
                page.wait_for_timeout(1200)
            out = OUT_DIR / filename
            page.screenshot(path=str(out), full_page=False)
            print(f"Saved {out}")
        browser.close()

    NeuralNetworkBridge.stop()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    capture_dashboard(DQN_AGENT_KEY, "dashboard-dqn.png", steps=40)
    capture_dashboard(Q_AGENT_KEY, "dashboard-qlearning.png", steps=32)
    capture_dashboard(DQN_AGENT_KEY, "dashboard-advanced.png", steps=36, ui_mode="advanced")
    capture_neural_pages()
    print("All README screenshots captured.")


if __name__ == "__main__":
    main()
