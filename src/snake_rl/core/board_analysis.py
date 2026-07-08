from collections import deque
from typing import List, Optional, Set, Tuple

from snake_rl.envs.snake_env import Action, Direction


Point = Tuple[int, int]


def direction_after_action(current_direction: Direction, action: Action) -> Direction:
    """
    Convert a relative action into the next absolute direction.

    Example:
    If snake is facing RIGHT:
    - STRAIGHT means still RIGHT
    - RIGHT means turn DOWN
    - LEFT means turn UP
    """
    if action == Action.STRAIGHT:
        return current_direction

    if action == Action.RIGHT:
        return Direction((int(current_direction) + 1) % 4)

    if action == Action.LEFT:
        return Direction((int(current_direction) - 1) % 4)

    raise ValueError(f"Unknown action: {action}")


def point_in_direction(point: Point, direction: Direction) -> Point:
    """
    Return the next board position if we move one cell in a direction.
    """
    x, y = point

    if direction == Direction.RIGHT:
        return x + 1, y

    if direction == Direction.DOWN:
        return x, y + 1

    if direction == Direction.LEFT:
        return x - 1, y

    if direction == Direction.UP:
        return x, y - 1

    raise ValueError(f"Unknown direction: {direction}")


def next_point_after_action(
    head: Point,
    current_direction: Direction,
    action: Action,
) -> Point:
    """
    Given the snake head, current direction, and chosen action,
    return where the head would go next.
    """
    next_direction = direction_after_action(current_direction, action)
    return point_in_direction(head, next_direction)


def is_inside_board(point: Point, width: int, height: int) -> bool:
    """
    Check if a point is inside the board.
    """
    x, y = point
    return 0 <= x < width and 0 <= y < height


def is_dangerous_point(
    point: Point,
    snake: List[Point],
    width: int,
    height: int,
    ignore_tail: bool = False,
) -> bool:
    """
    Check whether a point is dangerous.

    Dangerous means:
    - outside the board
    - inside the snake body

    ignore_tail=True is useful because the tail usually moves away
    after the snake takes a step.
    """
    if not is_inside_board(point, width, height):
        return True

    body = set(snake)

    if ignore_tail and len(snake) > 0:
        body.discard(snake[-1])

    return point in body


def get_neighbors(point: Point) -> List[Point]:
    """
    Return the four neighboring cells around a point.
    """
    x, y = point

    return [
        (x + 1, y),
        (x - 1, y),
        (x, y + 1),
        (x, y - 1),
    ]


def count_open_space(
    start: Point,
    snake: List[Point],
    width: int,
    height: int,
    ignore_tail: bool = True,
    max_count: Optional[int] = None,
) -> int:
    """
    Count how many free cells are reachable from a starting point.

    This uses flood fill / BFS.

    Simple meaning:
    Starting from this cell, how much room can the snake move into?

    If the answer is small, the move may be a trap.
    """
    if is_dangerous_point(
        point=start,
        snake=snake,
        width=width,
        height=height,
        ignore_tail=ignore_tail,
    ):
        return 0

    blocked: Set[Point] = set(snake)

    if ignore_tail and len(snake) > 0:
        blocked.discard(snake[-1])

    visited: Set[Point] = set()
    queue = deque([start])

    while queue:
        current = queue.popleft()

        if current in visited:
            continue

        if not is_inside_board(current, width, height):
            continue

        if current in blocked:
            continue

        visited.add(current)

        if max_count is not None and len(visited) >= max_count:
            return len(visited)

        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                queue.append(neighbor)

    return len(visited)


def bucket_open_space(open_space: int, width: int, height: int) -> int:
    """
    Convert open-space count into a small category.

    This is important for tabular Q-learning.

    We do NOT want too many exact numbers like:
    17, 18, 19, 20, 21...

    Instead we use buckets:
    0 = low space
    1 = medium space
    2 = high space
    """
    board_area = width * height
    ratio = open_space / board_area

    if ratio < 0.15:
        return 0

    if ratio < 0.35:
        return 1

    return 2


def bucket_length(snake_length: int, width: int, height: int) -> int:
    """
    Bucket snake length relative to board size for tabular Q-learning.

    Returns:
        0 = short  (< 25% of board)
        1 = medium (25–50% of board)
        2 = long   (> 50% of board)
    """
    board_area = width * height
    ratio = snake_length / board_area

    if ratio < 0.25:
        return 0

    if ratio <= 0.50:
        return 1

    return 2


def is_danger_two_steps(
    head: Point,
    current_direction: Direction,
    action: Action,
    snake: List[Point],
    width: int,
    height: int,
) -> bool:
    """
    True if taking `action` leads into danger within the next two moves.

    Helps detect traps that are one step past the immediate cell.
    """
    first_dir = direction_after_action(current_direction, action)
    first_point = point_in_direction(head, first_dir)

    if is_dangerous_point(
        first_point, snake, width, height, ignore_tail=True
    ):
        return True

    simulated_snake = [first_point] + snake[:-1]

    for follow_action in (Action.STRAIGHT, Action.RIGHT, Action.LEFT):
        second_point = next_point_after_action(
            first_point, first_dir, follow_action
        )
        if is_dangerous_point(
            second_point, simulated_snake, width, height, ignore_tail=True
        ):
            return True

    return False


def can_reach_point(
    start: Point,
    target: Point,
    snake: List[Point],
    width: int,
    height: int,
    ignore_tail: bool = True,
) -> bool:
    """
    Check if target is reachable from start.

    This also uses BFS.

    Example:
    - Can the snake reach food?
    - Can the snake reach its tail?

    Reachability is important because a move may look safe now
    but cut off escape routes.
    """
    if start == target:
        return True

    if is_dangerous_point(
        point=start,
        snake=snake,
        width=width,
        height=height,
        ignore_tail=ignore_tail,
    ):
        return False

    blocked: Set[Point] = set(snake)

    if ignore_tail and len(snake) > 0:
        blocked.discard(snake[-1])

    visited: Set[Point] = set()
    queue = deque([start])

    while queue:
        current = queue.popleft()

        if current in visited:
            continue

        if not is_inside_board(current, width, height):
            continue

        if current in blocked:
            continue

        if current == target:
            return True

        visited.add(current)

        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                queue.append(neighbor)

    return False


def analyse_move(
    snake: List[Point],
    food: Optional[Point],
    current_direction: Direction,
    action: Action,
    width: int,
    height: int,
) -> dict:
    """
    Analyse one possible move.

    This answers:
    - where would the head go?
    - is it dangerous?
    - how much open space exists there?
    - can food be reached from there?
    - can tail be reached from there?
    """
    head = snake[0]
    next_point = next_point_after_action(
        head=head,
        current_direction=current_direction,
        action=action,
    )

    dangerous = is_dangerous_point(
        point=next_point,
        snake=snake,
        width=width,
        height=height,
        ignore_tail=True,
    )

    open_space = count_open_space(
        start=next_point,
        snake=snake,
        width=width,
        height=height,
        ignore_tail=True,
    )

    open_space_bucket = bucket_open_space(
        open_space=open_space,
        width=width,
        height=height,
    )

    if food is None:
        food_reachable = False
    else:
        food_reachable = can_reach_point(
            start=next_point,
            target=food,
            snake=snake,
            width=width,
            height=height,
            ignore_tail=True,
        )

    tail = snake[-1]

    tail_reachable = can_reach_point(
        start=next_point,
        target=tail,
        snake=snake,
        width=width,
        height=height,
        ignore_tail=True,
    )

    return {
        "action": action,
        "next_point": next_point,
        "dangerous": dangerous,
        "open_space": open_space,
        "open_space_bucket": open_space_bucket,
        "food_reachable": food_reachable,
        "tail_reachable": tail_reachable,
    }


def analyse_all_moves(
    snake: List[Point],
    food: Optional[Point],
    current_direction: Direction,
    width: int,
    height: int,
) -> dict:
    """
    Analyse STRAIGHT, RIGHT, and LEFT.

    This is useful for building advanced state features.
    """
    return {
        Action.STRAIGHT: analyse_move(
            snake=snake,
            food=food,
            current_direction=current_direction,
            action=Action.STRAIGHT,
            width=width,
            height=height,
        ),
        Action.RIGHT: analyse_move(
            snake=snake,
            food=food,
            current_direction=current_direction,
            action=Action.RIGHT,
            width=width,
            height=height,
        ),
        Action.LEFT: analyse_move(
            snake=snake,
            food=food,
            current_direction=current_direction,
            action=Action.LEFT,
            width=width,
            height=height,
        ),
    }
