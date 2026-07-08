from snake_rl.core.board_analysis import analyse_all_moves
from snake_rl.envs.snake_env import Direction


def main():
    width = 8
    height = 8

    snake = [
        (3, 3),  # head
        (2, 3),
        (1, 3),
    ]

    food = (6, 3)
    direction = Direction.RIGHT

    analysis = analyse_all_moves(
        snake=snake,
        food=food,
        current_direction=direction,
        width=width,
        height=height,
    )

    print("Board analysis test")
    print("=" * 50)

    for action, info in analysis.items():
        print(f"\nAction: {action.name}")
        print(f"  next_point:         {info['next_point']}")
        print(f"  dangerous:          {info['dangerous']}")
        print(f"  open_space:         {info['open_space']}")
        print(f"  open_space_bucket:  {info['open_space_bucket']}")
        print(f"  food_reachable:     {info['food_reachable']}")
        print(f"  tail_reachable:     {info['tail_reachable']}")

    print("\nTest completed.")


if __name__ == "__main__":
    main()
