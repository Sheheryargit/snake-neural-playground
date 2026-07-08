from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from snake_rl.envs.snake_env import Action, Direction


Point = Tuple[int, int]


def direction_after_action(direction: Direction, action: Action) -> Direction:
    if action == Action.STRAIGHT:
        return direction

    if action == Action.RIGHT:
        return Direction((int(direction) + 1) % 4)

    if action == Action.LEFT:
        return Direction((int(direction) - 1) % 4)

    raise ValueError(f"Unknown action: {action}")


def point_in_direction(point: Point, direction: Direction) -> Point:
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
    direction: Direction,
    action: Action,
) -> Point:
    next_direction = direction_after_action(direction, action)
    return point_in_direction(head, next_direction)


def action_to_reach_point(
    head: Point,
    direction: Direction,
    target_next_point: Point,
) -> Optional[Action]:
    """
    Convert a desired next cell into STRAIGHT / RIGHT / LEFT.
    """
    for action in [Action.STRAIGHT, Action.RIGHT, Action.LEFT]:
        possible_next = next_point_after_action(
            head=head,
            direction=direction,
            action=action,
        )

        if possible_next == target_next_point:
            return action

    return None


def is_inside_board(point: Point, width: int, height: int) -> bool:
    x, y = point
    return 0 <= x < width and 0 <= y < height


def get_neighbors(point: Point) -> List[Point]:
    x, y = point

    return [
        (x + 1, y),
        (x - 1, y),
        (x, y + 1),
        (x, y - 1),
    ]


def is_safe_point(
    point: Point,
    snake: List[Point],
    width: int,
    height: int,
    ignore_tail: bool = True,
) -> bool:
    """
    Safe means:
    - inside board
    - not inside snake body

    ignore_tail=True because the tail usually moves away next step.
    """
    if not is_inside_board(point, width, height):
        return False

    blocked = set(snake)

    if ignore_tail and len(snake) > 0:
        blocked.discard(snake[-1])

    return point not in blocked


def find_path(
    start: Point,
    target: Point,
    snake: List[Point],
    width: int,
    height: int,
    ignore_tail: bool = True,
) -> Optional[List[Point]]:
    """
    BFS shortest path.

    Returns:
        [start, ..., target]

    If no path:
        None
    """
    if not is_inside_board(start, width, height):
        return None

    if not is_inside_board(target, width, height):
        return None

    blocked: Set[Point] = set(snake)

    if ignore_tail and len(snake) > 0:
        blocked.discard(snake[-1])

    # Start is the current head, so allow it.
    blocked.discard(start)

    queue = deque([start])
    came_from: Dict[Point, Optional[Point]] = {start: None}

    while queue:
        current = queue.popleft()

        if current == target:
            break

        for neighbor in get_neighbors(current):
            if neighbor in came_from:
                continue

            if not is_inside_board(neighbor, width, height):
                continue

            if neighbor in blocked and neighbor != target:
                continue

            came_from[neighbor] = current
            queue.append(neighbor)

    if target not in came_from:
        return None

    path = []
    current: Optional[Point] = target

    while current is not None:
        path.append(current)
        current = came_from[current]

    path.reverse()
    return path


def is_boundary_point(point: Point, width: int, height: int) -> bool:
    x, y = point

    return (
        x == 0
        or x == width - 1
        or y == 0
        or y == height - 1
    )


def get_boundary_cycle(width: int, height: int) -> List[Point]:
    """
    Clockwise outer boundary cycle.

    Starts at top-left:
    top row left -> right
    right wall top -> bottom
    bottom row right -> left
    left wall bottom -> top
    """
    cycle: List[Point] = []

    # Top row
    for x in range(width):
        cycle.append((x, 0))

    # Right wall, excluding top corner
    for y in range(1, height):
        cycle.append((width - 1, y))

    # Bottom row, excluding right corner
    for x in range(width - 2, -1, -1):
        cycle.append((x, height - 1))

    # Left wall, excluding bottom and top corners
    for y in range(height - 2, 0, -1):
        cycle.append((0, y))

    return cycle


def get_next_boundary_point(
    point: Point,
    width: int,
    height: int,
) -> Point:
    """
    Get the next point in the clockwise boundary loop.
    """
    cycle = get_boundary_cycle(width, height)

    if point not in cycle:
        return point

    index = cycle.index(point)
    next_index = (index + 1) % len(cycle)

    return cycle[next_index]


def find_path_to_nearest_boundary(
    start: Point,
    snake: List[Point],
    width: int,
    height: int,
) -> Optional[List[Point]]:
    """
    Find shortest path to any safe boundary cell.
    """
    best_path: Optional[List[Point]] = None

    for boundary_point in get_boundary_cycle(width, height):
        path = find_path(
            start=start,
            target=boundary_point,
            snake=snake,
            width=width,
            height=height,
            ignore_tail=True,
        )

        if path is None:
            continue

        if best_path is None or len(path) < len(best_path):
            best_path = path

    return best_path
