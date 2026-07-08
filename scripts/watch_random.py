import random
import time

import pygame

from snake_rl.envs.snake_env import Action, SnakeEnv


CELL_SIZE = 40
SIDEBAR_WIDTH = 300
FPS = 6


def draw_text(screen, text, x, y, size=22):
    font = pygame.font.SysFont("Arial", size)
    surface = font.render(text, True, (240, 240, 240))
    screen.blit(surface, (x, y))


def draw_env(screen, env, last_action, last_reward, total_reward, done, reason):
    screen.fill((20, 20, 20))

    # Draw board cells
    for y in range(env.height):
        for x in range(env.width):
            rect = pygame.Rect(
                x * CELL_SIZE,
                y * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE,
            )

            pygame.draw.rect(screen, (35, 35, 35), rect)
            pygame.draw.rect(screen, (60, 60, 60), rect, 1)

    # Draw food
    if env.food is not None:
        food_x, food_y = env.food
        food_rect = pygame.Rect(
            food_x * CELL_SIZE,
            food_y * CELL_SIZE,
            CELL_SIZE,
            CELL_SIZE,
        )
        pygame.draw.rect(screen, (220, 70, 70), food_rect)

    # Draw snake body
    for body_x, body_y in env.snake[1:]:
        body_rect = pygame.Rect(
            body_x * CELL_SIZE,
            body_y * CELL_SIZE,
            CELL_SIZE,
            CELL_SIZE,
        )
        pygame.draw.rect(screen, (70, 170, 90), body_rect)

    # Draw snake head
    head_x, head_y = env.snake[0]
    head_rect = pygame.Rect(
        head_x * CELL_SIZE,
        head_y * CELL_SIZE,
        CELL_SIZE,
        CELL_SIZE,
    )
    pygame.draw.rect(screen, (90, 240, 120), head_rect)

    # Sidebar
    sidebar_x = env.width * CELL_SIZE + 20

    draw_text(screen, "Snake RL Visualizer", sidebar_x, 30, 26)
    draw_text(screen, "Agent: Random", sidebar_x, 75)
    draw_text(screen, f"Score: {env.score}", sidebar_x, 120)
    draw_text(screen, f"Steps: {env.steps}", sidebar_x, 160)
    draw_text(screen, f"Action: {last_action.name}", sidebar_x, 200)
    draw_text(screen, f"Reward: {last_reward:.2f}", sidebar_x, 240)
    draw_text(screen, f"Total reward: {total_reward:.2f}", sidebar_x, 280)

    if done:
        draw_text(screen, "GAME OVER", sidebar_x, 350, 28)
        draw_text(screen, f"Reason: {reason}", sidebar_x, 390)


def main():
    env = SnakeEnv(width=12, height=12, seed=42)
    env.reset()

    pygame.init()

    screen_width = env.width * CELL_SIZE + SIDEBAR_WIDTH
    screen_height = env.height * CELL_SIZE

    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Snake RL Environment")

    clock = pygame.time.Clock()

    running = True
    done = False
    reason = "running"
    last_action = Action.STRAIGHT
    last_reward = 0.0
    total_reward = 0.0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if not done:
            # This is currently dumb/random.
            # Later we replace this with agent.choose_action(state).
            last_action = random.choice([
                Action.STRAIGHT,
                Action.RIGHT,
                Action.LEFT,
            ])

            result = env.step(last_action)

            last_reward = result.reward
            total_reward += result.reward
            done = result.done
            reason = result.info["reason"]

        draw_env(
            screen=screen,
            env=env,
            last_action=last_action,
            last_reward=last_reward,
            total_reward=total_reward,
            done=done,
            reason=reason,
        )

        pygame.display.flip()
        clock.tick(FPS)

        # Restart after death so you can keep watching.
        if done:
            time.sleep(1.5)
            env.reset()
            done = False
            reason = "running"
            last_action = Action.STRAIGHT
            last_reward = 0.0
            total_reward = 0.0

    pygame.quit()


if __name__ == "__main__":
    main()
