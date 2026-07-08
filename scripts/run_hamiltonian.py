#!/usr/bin/env python3
"""Run Hamiltonian Cycle agent headlessly for many environment steps."""

import argparse
import sys
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from snake_rl.agents.hamiltonian_agent import HamiltonianCycleAgent
from snake_rl.envs.snake_env import SnakeEnv


def run(
    total_steps: int,
    seed: int = 42,
    log_every: int = 100_000,
    log_path: str = "models/hamiltonian_1m_run.log",
    width: int = 12,
    height: int = 12,
) -> None:
    started_at = datetime.now()

    env = SnakeEnv(width=width, height=height, seed=seed)
    agent = HamiltonianCycleAgent()
    agent._ensure_cycle(width, height)

    lines = []
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    def emit(message: str) -> None:
        print(message, flush=True)
        lines.append(message)

    emit("=" * 72)
    emit("HAMILTONIAN CYCLE RUN")
    emit("=" * 72)
    emit(f"Started at:        {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    emit(f"Target steps:      {total_steps:,}")
    emit(f"Board size:        {env.width}x{env.height}")
    emit(f"Cycle length:      {len(agent.cycle)}")
    emit(f"Seed:              {seed}")
    emit("=" * 72)
    emit("")

    state = agent.reset_env(env)

    steps = 0
    episodes = 0
    best_score = 0
    best_score_episode = 0

    episode_reward = 0.0
    episode_steps = 0
    death_reasons: Counter = Counter()
    fallback_moves = 0

    recent_scores: deque = deque(maxlen=100)
    recent_rewards: deque = deque(maxlen=100)
    recent_lengths: deque = deque(maxlen=100)

    wall_clock_start = time.perf_counter()
    last_log_step = 0

    while steps < total_steps:
        action = agent.choose_action(state, env)
        if agent.debug_info.get("reason") != "following Hamiltonian cycle":
            fallback_moves += 1

        result = env.step(action)

        steps += 1
        episode_steps += 1
        episode_reward += result.reward
        state = result.state

        if result.done:
            episodes += 1
            score = result.info["score"]
            reason = result.info["reason"]
            death_reasons[reason] += 1

            recent_scores.append(score)
            recent_rewards.append(episode_reward)
            recent_lengths.append(episode_steps)

            if score > best_score:
                best_score = score
                best_score_episode = episodes

            agent.reset()
            state = agent.reset_env(env)
            episode_reward = 0.0
            episode_steps = 0

        if steps - last_log_step >= log_every or steps == total_steps:
            elapsed = time.perf_counter() - wall_clock_start
            steps_per_sec = steps / elapsed if elapsed > 0 else 0.0
            avg_score = sum(recent_scores) / len(recent_scores) if recent_scores else 0.0
            avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
            avg_len = sum(recent_lengths) / len(recent_lengths) if recent_lengths else 0.0

            emit(
                f"[{steps:>9,}/{total_steps:,} steps] "
                f"ep={episodes:>6,} | "
                f"best={best_score} (ep {best_score_episode}) | "
                f"avg100 score={avg_score:.2f} reward={avg_reward:.2f} len={avg_len:.1f} | "
                f"fallbacks={fallback_moves:,} | "
                f"{steps_per_sec:,.0f} steps/s"
            )
            last_log_step = steps

    elapsed = time.perf_counter() - wall_clock_start
    finished_at = datetime.now()
    all_scores = list(recent_scores)

    emit("")
    emit("=" * 72)
    emit("RUN COMPLETE")
    emit("=" * 72)
    emit(f"Finished at:       {finished_at.strftime('%Y-%m-%d %H:%M:%S')}")
    emit(f"Duration:          {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    emit(f"Total steps:       {steps:,}")
    emit(f"Total episodes:    {episodes:,}")
    emit(f"Avg steps/episode: {steps / episodes:.1f}" if episodes else "N/A")
    emit(f"Throughput:        {steps / elapsed:,.0f} steps/sec")
    emit("")
    emit("Scores:")
    emit(f"  best score:      {best_score} (episode {best_score_episode})")
    if all_scores:
        emit(f"  last-100 avg:    {sum(all_scores) / len(all_scores):.3f}")
    emit(f"  fallback moves:  {fallback_moves:,}")
    emit("")
    emit("Death reasons (all episodes):")
    for reason, count in death_reasons.most_common():
        pct = 100.0 * count / episodes if episodes else 0
        emit(f"  {reason:12s} {count:>8,}  ({pct:.1f}%)")
    emit("=" * 72)

    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    emit(f"Log saved to:      {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hamiltonian Cycle agent headlessly")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Environment steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100_000, help="Log interval in steps")
    parser.add_argument(
        "--log-path",
        type=str,
        default="models/hamiltonian_1m_run.log",
        help="Where to save the run log",
    )
    parser.add_argument("--width", type=int, default=12, help="Board width")
    parser.add_argument("--height", type=int, default=12, help="Board height")
    args = parser.parse_args()
    run(
        total_steps=args.steps,
        seed=args.seed,
        log_every=args.log_every,
        log_path=args.log_path,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
