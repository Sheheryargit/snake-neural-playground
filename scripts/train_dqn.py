#!/usr/bin/env python3
"""Train DQN agent for many environment steps (headless)."""

import argparse
import sys
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from snake_rl.agents.dqn_agent import DQNAgent
from snake_rl.envs.snake_env import SnakeEnv


def train(
    total_steps: int,
    seed: int = 42,
    log_every: int = 50_000,
    save_path: str = "models/dqn_11_state.pt",
    width: int = 12,
    height: int = 12,
) -> None:
    started_at = datetime.now()

    env = SnakeEnv(width=width, height=height, seed=seed)
    agent = DQNAgent(seed=seed)
    agent.set_training_mode()

    print("=" * 72)
    print("DQN TRAINING RUN")
    print("=" * 72)
    print(f"Started at:        {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target steps:      {total_steps:,}")
    print(f"Board size:        {env.width}x{env.height}")
    print()
    print("Hyperparameters:")
    print(f"  epsilon start:          {agent.epsilon_start}")
    print(f"  min_epsilon:            {agent.min_epsilon}")
    print(f"  epsilon_decay_steps:    {agent.epsilon_decay_steps:,}  (linear decay)")
    print(f"  warmup_steps:           {agent.warmup_steps:,}")
    print(f"  train_every:            {agent.train_every}")
    print(f"  gamma:                  {agent.gamma}")
    print(f"  learning rate:          {agent.learning_rate}")
    print(f"  batch size:             {agent.batch_size}")
    print(f"  memory size:            {agent.replay_memory.memory.maxlen:,}")
    print(f"  target update every:    {agent.target_update_every} gradient steps")
    print(f"  loss function:          SmoothL1Loss")
    print(f"  seed:                   {seed}")
    print("=" * 72)
    print()

    state = env.reset()
    agent.reset()

    steps = 0
    episodes = 0
    best_score = 0
    best_score_episode = 0

    episode_reward = 0.0
    episode_steps = 0
    death_reasons: Counter = Counter()

    recent_scores: deque = deque(maxlen=100)
    recent_rewards: deque = deque(maxlen=100)
    recent_lengths: deque = deque(maxlen=100)
    recent_losses: deque = deque(maxlen=100)

    wall_clock_start = time.perf_counter()
    last_log_step = 0

    while steps < total_steps:
        action = agent.choose_action(state, env)
        result = env.step(action)

        agent.learn(
            state=state,
            action=action,
            reward=result.reward,
            next_state=result.state,
            done=result.done,
        )

        if agent.last_loss is not None:
            recent_losses.append(agent.last_loss)

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
            state = env.reset()
            episode_reward = 0.0
            episode_steps = 0

        if steps - last_log_step >= log_every or steps == total_steps:
            elapsed = time.perf_counter() - wall_clock_start
            steps_per_sec = steps / elapsed if elapsed > 0 else 0.0
            avg_score = sum(recent_scores) / len(recent_scores) if recent_scores else 0.0
            avg_reward = sum(recent_rewards) / len(recent_rewards) if recent_rewards else 0.0
            avg_len = sum(recent_lengths) / len(recent_lengths) if recent_lengths else 0.0
            avg_loss = sum(recent_losses) / len(recent_losses) if recent_losses else None
            last_loss = agent.last_loss

            last_loss_text = f"{last_loss:.4f}" if last_loss is not None else "n/a"
            avg_loss_text = f"{avg_loss:.4f}" if avg_loss is not None else "n/a"

            print(
                f"[{steps:>9,}/{total_steps:,} steps] "
                f"ep={episodes:>6,} | "
                f"epsilon={agent.epsilon:.4f} | "
                f"train_steps={agent.training_step_count:>7,} | "
                f"grad_steps={agent.gradient_step_count:>6,} | "
                f"memory={len(agent.replay_memory):>6,} | "
                f"best={best_score} (ep {best_score_episode}) | "
                f"avg100 score={avg_score:.2f} reward={avg_reward:.2f} len={avg_len:.1f} | "
                f"last_loss={last_loss_text} avg_loss={avg_loss_text} | "
                f"{steps_per_sec:,.0f} steps/s"
            )
            last_log_step = steps

    elapsed = time.perf_counter() - wall_clock_start
    finished_at = datetime.now()

    all_scores = list(recent_scores)
    print()
    print("=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)
    print(f"Finished at:       {finished_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration:          {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Total steps:       {steps:,}")
    print(f"Total episodes:    {episodes:,}")
    print(f"Avg steps/episode: {steps / episodes:.1f}" if episodes else "N/A")
    print(f"Throughput:        {steps / elapsed:,.0f} steps/sec")
    print()
    print("Scores:")
    print(f"  best score:      {best_score} (episode {best_score_episode})")
    if all_scores:
        print(f"  last-100 avg:    {sum(all_scores) / len(all_scores):.3f}")
    print()
    print(f"Final epsilon:     {agent.epsilon:.6f}")
    print(f"Replay memory:     {len(agent.replay_memory):,}")
    print(f"Training steps:    {agent.training_step_count:,}")
    print(f"Gradient steps:    {agent.gradient_step_count:,}")
    print(f"Warmup steps:      {agent.warmup_steps:,}")
    print(f"Train every:       {agent.train_every}")
    print(f"Epsilon decay:     linear over {agent.epsilon_decay_steps:,} steps")
    if recent_losses:
        print(f"Last loss:         {agent.last_loss:.4f}")
        print(f"Avg loss (last 100 updates): {sum(recent_losses) / len(recent_losses):.4f}")
    print()
    print("Death reasons (all episodes):")
    for reason, count in death_reasons.most_common():
        pct = 100.0 * count / episodes if episodes else 0
        print(f"  {reason:12s} {count:>8,}  ({pct:.1f}%)")
    print("=" * 72)

    agent.save(save_path, training_steps=steps)
    print(f"Model save path:   {save_path}")
    print(f"  training_steps: {steps:,}")
    print(f"  saved_at:       {agent.model_info.get('saved_at')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN agent headlessly")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Environment steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50_000, help="Log interval in steps")
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/dqn_11_state.pt",
        help="Where to save the trained DQN model",
    )
    parser.add_argument("--width", type=int, default=12, help="Board width")
    parser.add_argument("--height", type=int, default=12, help="Board height")
    args = parser.parse_args()
    train(
        total_steps=args.steps,
        seed=args.seed,
        log_every=args.log_every,
        save_path=args.save_path,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
