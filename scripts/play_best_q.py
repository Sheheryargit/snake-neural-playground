import pygame

from snake_rl.agents.q_learning_agent import QLearningAgent
from snake_rl.envs.snake_env import Action, SnakeEnv


CELL_SIZE = 42
BOARD_WIDTH = 12
BOARD_HEIGHT = 12
SIDEBAR_WIDTH = 360
FPS = 10


def draw_text(screen, text, x, y, size=22, color=(240, 240, 240)):
    font = pygame.font.SysFont("Arial", size)
    surface = font.render(str(text), True, color)
    screen.blit(surface, (x, y))


def draw_board(screen, env, agent, last_action, last_reward, total_reward, q_values):
    screen.fill((15, 17, 22))

    board_x = 20
    board_y = 20

    # Board
    for y in range(env.height):
        for x in range(env.width):
            rect = pygame.Rect(
                board_x + x * CELL_SIZE,
                board_y + y * CELL_SIZE,
                CELL_SIZE,
                CELL_SIZE,
            )
            pygame.draw.rect(screen, (35, 40, 50), rect)
            pygame.draw.rect(screen, (58, 66, 82), rect, 1)

    # Food
    if env.food is not None:
        fx, fy = env.food
        rect = pygame.Rect(
            board_x + fx * CELL_SIZE + 6,
            board_y + fy * CELL_SIZE + 6,
            CELL_SIZE - 12,
            CELL_SIZE - 12,
        )
        pygame.draw.rect(screen, (235, 80, 85), rect, border_radius=10)
        draw_text(screen, "F", rect.x + 10, rect.y + 6, 18)

    # Snake body
    for sx, sy in env.snake[1:]:
        rect = pygame.Rect(
            board_x + sx * CELL_SIZE + 4,
            board_y + sy * CELL_SIZE + 4,
            CELL_SIZE - 8,
            CELL_SIZE - 8,
        )
        pygame.draw.rect(screen, (70, 175, 95), rect, border_radius=8)

    # Snake head
    hx, hy = env.snake[0]
    rect = pygame.Rect(
        board_x + hx * CELL_SIZE + 3,
        board_y + hy * CELL_SIZE + 3,
        CELL_SIZE - 6,
        CELL_SIZE - 6,
    )
    pygame.draw.rect(screen, (95, 235, 135), rect, border_radius=9)
    draw_text(screen, "H", rect.x + 12, rect.y + 7, 18, (10, 25, 15))

    # Sidebar
    sidebar_x = board_x + env.width * CELL_SIZE + 30

    draw_text(screen, "Best Q-Learning Run", sidebar_x, 30, 28, (110, 205, 255))
    draw_text(screen, "Mode: Evaluation", sidebar_x, 80)
    draw_text(screen, "Epsilon: 0.00", sidebar_x, 115, 22, (95, 235, 135))

    draw_text(screen, f"Score: {env.score}", sidebar_x, 170, 24)
    draw_text(screen, f"Steps: {env.steps}", sidebar_x, 205, 24)
    draw_text(screen, f"Last action: {last_action.name}", sidebar_x, 240, 24)
    draw_text(screen, f"Last reward: {last_reward:.2f}", sidebar_x, 275, 24)
    draw_text(screen, f"Total reward: {total_reward:.2f}", sidebar_x, 310, 24)

    draw_text(screen, "Q-values now:", sidebar_x, 380, 24, (110, 205, 255))

    actions = [Action.STRAIGHT, Action.RIGHT, Action.LEFT]
    for i, action in enumerate(actions):
        value = q_values[i]
        color = (240, 240, 240)

        if action == last_action:
            color = (95, 235, 135)

        draw_text(
            screen,
            f"{action.name}: {value:.3f}",
            sidebar_x,
            420 + i * 34,
            22,
            color,
        )

    draw_text(screen, "Controls:", sidebar_x, 570, 24, (110, 205, 255))
    draw_text(screen, "SPACE = pause", sidebar_x, 610, 20)
    draw_text(screen, "R = restart", sidebar_x, 640, 20)
    draw_text(screen, "ESC = quit", sidebar_x, 670, 20)


def main():
    pygame.init()

    env = SnakeEnv(width=BOARD_WIDTH, height=BOARD_HEIGHT, seed=42)

    agent = QLearningAgent(seed=42)
    agent.load("models/q_learning_best.pkl")

    # Best-Q / evaluation mode
    agent.epsilon = 0.0

    state = env.reset()
    done = False
    paused = False

    last_action = Action.STRAIGHT
    last_reward = 0.0
    total_reward = 0.0

    screen_width = BOARD_WIDTH * CELL_SIZE + SIDEBAR_WIDTH + 70
    screen_height = BOARD_HEIGHT * CELL_SIZE + 40

    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Best Q-Learning Snake")

    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                if event.key == pygame.K_SPACE:
                    paused = not paused

                if event.key == pygame.K_r:
                    state = env.reset()
                    done = False
                    total_reward = 0.0
                    last_action = Action.STRAIGHT
                    last_reward = 0.0

        if not paused and not done:
            last_action = agent.choose_action(state, env)
            result = env.step(last_action)

            state = result.state
            last_reward = result.reward
            total_reward += result.reward
            done = result.done

        q_values = agent.get_q_values(state)

        draw_board(
            screen=screen,
            env=env,
            agent=agent,
            last_action=last_action,
            last_reward=last_reward,
            total_reward=total_reward,
            q_values=q_values,
        )

        if done:
            draw_text(
                screen,
                "GAME OVER - Press R to restart",
                40,
                screen_height - 35,
                24,
                (255, 105, 105),
            )

        if paused:
            draw_text(
                screen,
                "PAUSED",
                210,
                screen_height - 35,
                24,
                (110, 205, 255),
            )

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
