# pygame_pathfinding_game.py
import pygame
import asyncio
import platform
import random
from enum import Enum
from typing import List, Tuple, Optional, Dict
from collections import deque
import heapq

# Screen and grid constants
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
TILE_SIZE = 40
ROWS = SCREEN_HEIGHT // TILE_SIZE
COLS = SCREEN_WIDTH // TILE_SIZE
FPS = 15
WALL_NUM = 35

PLAYER_COLOR = (0, 0, 255)  # Blue
TARGET_COLOR = (255, 0, 0)  # Red
WALL_COLOR = (50, 50, 50)   # Dark gray
GRID_COLOR = (200, 200, 200)  # Light gray
BACKGROUND_COLOR = (240, 240, 240)  # Very light gray
VICTORY_COLOR = (0, 255, 0)  # Green

# Define cell states
class CellState(Enum):
    EMPTY = 0
    WALL = 1
    PLAYER = 2
    TARGET = 3

# Initialize grid and walls
grid: List[List[CellState]] = [[CellState.EMPTY for _ in range(COLS)] for _ in range(ROWS)]
walls: List[Tuple[int,int]] = []

# Pygame globals
screen = None
player_image = None
target_image = None
wall_image = None
victory_image = None

# Orthogonal directions
DIRS = [(1,0),(-1,0),(0,1),(0,-1)]

# Path caches
path_caches = {
    'dfs': {'path': None, 'idx': 0, 'goal': None},
    'bfs': {'path': None, 'idx': 0, 'goal': None},
    'astar': {'path': None, 'idx': 0, 'goal': None},
    'bidi': {'path': None, 's_idx': 0, 'e_idx': -1, 'goal': None},
}

# --- helpers ---------------------------------------------------------------
def in_bounds(pos: Tuple[int,int]) -> bool:
    r,c = pos
    return 0 <= r < ROWS and 0 <= c < COLS

def neighbors(g: List[List[CellState]], pos: Tuple[int,int]):
    for dr, dc in DIRS:
        nr, nc = pos[0] + dr, pos[1] + dc
        if in_bounds((nr,nc)) and g[nr][nc] != CellState.WALL:
            yield (nr, nc)

def reconstruct_parent(parent: Dict[Tuple[int,int], Tuple[int,int]], start: Tuple[int,int], goal: Tuple[int,int]) -> List[Tuple[int,int]]:
    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = parent.get(cur)
        if cur is None:
            return []
    path.append(start)
    path.reverse()
    return path

# --- path finders ---------------------------------------------------------
def bfs_path(g: List[List[CellState]], start: Tuple[int,int], goal: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
    if start == goal:
        return [start]
    q = deque([start])
    visited = {start}
    parent: Dict[Tuple[int,int], Tuple[int,int]] = {}
    while q:
        u = q.popleft()
        for v in neighbors(g, u):
            if v in visited:
                continue
            visited.add(v)
            parent[v] = u
            if v == goal:
                return reconstruct_parent(parent, start, goal)
            q.append(v)
    return None

def dfs_path(g: List[List[CellState]], start: Tuple[int,int], goal: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
    if start == goal:
        return [start]
    visited = set()
    parent: Dict[Tuple[int,int], Tuple[int,int]] = {}
    found = False
    def dfs(u: Tuple[int,int]):
        nonlocal found
        if found:
            return
        if u == goal:
            found = True
            return
        visited.add(u)
        for v in neighbors(g, u):
            if v in visited:
                continue
            parent[v] = u
            dfs(v)
            if found:
                return
    dfs(start)
    if not found:
        return None
    return reconstruct_parent(parent, start, goal)

def manhattan(a: Tuple[int,int], b: Tuple[int,int]) -> int:
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar_path(g: List[List[CellState]], start: Tuple[int,int], goal: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
    if start == goal:
        return [start]
    open_heap = []  # (f, g, node)
    heapq.heappush(open_heap, (manhattan(start,goal), 0, start))
    g_score: Dict[Tuple[int,int], int] = {start: 0}
    parent: Dict[Tuple[int,int], Tuple[int,int]] = {}
    closed = set()
    while open_heap:
        f, gcur, u = heapq.heappop(open_heap)
        if u in closed:
            continue
        if u == goal:
            return reconstruct_parent(parent, start, goal)
        closed.add(u)
        for v in neighbors(g, u):
            tentative = g_score.get(u, 10**9) + 1
            if tentative < g_score.get(v, 10**9):
                g_score[v] = tentative
                parent[v] = u
                heapq.heappush(open_heap, (tentative + manhattan(v,goal), tentative, v))
    return None

def bidirectional_bfs_path(g: List[List[CellState]], start: Tuple[int,int], goal: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
    """
    Safe bidirectional BFS path finder.
    Returns a list of cells from start to goal if path exists, else None.
    """
    if start == goal:
        return [start]

    qf = deque([start])
    qb = deque([goal])
    pf: Dict[Tuple[int,int], Optional[Tuple[int,int]]] = {start: None}
    pb: Dict[Tuple[int,int], Optional[Tuple[int,int]]] = {goal: None}
    vf = {start}
    vb = {goal}
    meeting = None

    while qf and qb and meeting is None:
        # expand forward one layer
        for _ in range(len(qf)):
            u = qf.popleft()
            for v in neighbors(g, u):
                if v in vf:
                    continue
                pf[v] = u
                vf.add(v)
                qf.append(v)
                if v in vb:
                    meeting = v
                    break
            if meeting:
                break
        if meeting:
            break
        # expand backward one layer
        for _ in range(len(qb)):
            u = qb.popleft()
            for v in neighbors(g, u):
                if v in vb:
                    continue
                pb[v] = u
                vb.add(v)
                qb.append(v)
                if v in vf:
                    meeting = v
                    break
            if meeting:
                break

    if meeting is None:
        return None

    # Reconstruct forward part: start -> meeting
    forward = []
    cur = meeting
    while cur is not None:
        forward.append(cur)
        cur = pf.get(cur)
    forward.reverse()  # now start ... meeting

    # Reconstruct backward part: meeting -> goal (excluding meeting)
    backward = []
    cur = meeting
    while True:
        nxt = pb.get(cur)
        if nxt is None:
            break
        backward.append(nxt)
        cur = nxt

    full = forward + backward
    # sanity
    if not full or full[0] != start or full[-1] != goal:
        return None
    return full

# --- TODO functions (implemented) -----------------------------------------
def _ensure_path_and_sync(cache_key: str, compute_fn, player_pos: Tuple[int,int], target_pos: Tuple[int,int]):
    """
    Ensure path exists in cache and sync index with player_pos.
    Returns (path, idx) or (None, None) if no path.
    """
    cache = path_caches[cache_key]
    path = cache.get('path')
    # recompute if no path or goal changed
    if path is None or cache.get('goal') != target_pos:
        path = compute_fn(grid, player_pos, target_pos)
        cache['path'] = path
        cache['idx'] = 0
        cache['goal'] = target_pos
        if path is None:
            return None, None
    else:
        # sync index if needed
        idx = cache.get('idx', 0)
        if not path:
            return None, None
        if idx < 0 or idx >= len(path) or path[idx] != player_pos:
            # try find player's index
            try:
                idx = path.index(player_pos)
            except ValueError:
                # recompute from current player_pos
                path = compute_fn(grid, player_pos, target_pos)
                cache['path'] = path
                cache['idx'] = 0
                cache['goal'] = target_pos
                if path is None:
                    return None, None
                idx = 0
        cache['idx'] = idx
    return cache['path'], cache['idx']

def dfs_make_move(grid_param: List[List[CellState]], player_pos: Tuple[int,int], target_pos: Tuple[int,int]) -> Tuple[int,int]:
    path, idx = _ensure_path_and_sync('dfs', dfs_path, player_pos, target_pos)
    if path is None:
        print("DFS: no path")
        return player_pos
    if len(path) <= 1:
        return player_pos
    next_idx = min(idx + 1, len(path)-1)
    path_caches['dfs']['idx'] = next_idx
    return path[next_idx]

def bfs_make_move(grid_param: List[List[CellState]], player_pos: Tuple[int,int], target_pos: Tuple[int,int]) -> Tuple[int,int]:
    path, idx = _ensure_path_and_sync('bfs', bfs_path, player_pos, target_pos)
    if path is None:
        print("BFS: no path")
        return player_pos
    if len(path) <= 1:
        return player_pos
    next_idx = min(idx + 1, len(path)-1)
    path_caches['bfs']['idx'] = next_idx
    return path[next_idx]

def a_star_make_move(grid_param: List[List[CellState]], player_pos: Tuple[int,int], target_pos: Tuple[int,int]) -> Tuple[int,int]:
    path, idx = _ensure_path_and_sync('astar', astar_path, player_pos, target_pos)
    if path is None:
        print("A*: no path")
        return player_pos
    if len(path) <= 1:
        return player_pos
    next_idx = min(idx + 1, len(path)-1)
    path_caches['astar']['idx'] = next_idx
    return path[next_idx]

def bidi_a_star_make_move(grid_param: List[List[CellState]], player_pos: Tuple[int,int], target_pos: Tuple[int,int]) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    """
    Bidirectional step: moves player forward and target backward along full shortest path.
    When they meet or cross we return the meeting cell for both and clear the cache
    so the next frame a new path/search will be started.
    """
    cache = path_caches['bidi']
    path = cache.get('path')
    # If players already at same cell -> trigger reset (main loop shows victory and re-init)
    if player_pos == target_pos:
        cache['path'] = None
        return player_pos, target_pos

    # compute path if needed
    if path is None or cache.get('goal') != target_pos:
        path = bidirectional_bfs_path(grid_param, player_pos, target_pos)
        cache['path'] = path
        cache['s_idx'] = 0
        cache['e_idx'] = (len(path)-1) if path else -1
        cache['goal'] = target_pos
        if path is None:
            print("Bidirectional: no path")
            return player_pos, target_pos

    # ensure indices in-range and synced with actual positions
    s_idx = cache.get('s_idx', 0)
    e_idx = cache.get('e_idx', len(path)-1)
    # try to find player's index
    try:
        if path[s_idx] != player_pos:
            s_idx = path.index(player_pos)
    except (ValueError, IndexError):
        # recompute path from current positions
        path = bidirectional_bfs_path(grid_param, player_pos, target_pos)
        cache['path'] = path
        if path is None:
            cache['s_idx'] = 0
            cache['e_idx'] = -1
            return player_pos, target_pos
        s_idx = 0
        e_idx = len(path)-1

    try:
        if path[e_idx] != target_pos:
            e_idx = path.index(target_pos)
    except (ValueError, IndexError):
        path = bidirectional_bfs_path(grid_param, player_pos, target_pos)
        cache['path'] = path
        if path is None:
            cache['s_idx'] = 0
            cache['e_idx'] = -1
            return player_pos, target_pos
        s_idx = 0
        e_idx = len(path)-1

    # If already met or crossed -> produce meeting and reset
    if s_idx >= e_idx:
        meet = path[s_idx]
        cache['path'] = None
        cache['s_idx'] = 0
        cache['e_idx'] = -1
        return meet, meet

    # compute next indices and check if next move will meet/cross
    new_s_idx = s_idx + 1
    new_e_idx = e_idx - 1

    # if after moving they would be at the same cell or cross, return meeting and reset cache
    if new_s_idx >= new_e_idx:
        # choose meeting index: prefer new_s_idx if within bounds, else e_idx
        if 0 <= new_s_idx < len(path):
            meet_idx = new_s_idx
        else:
            meet_idx = e_idx
        meet = path[meet_idx]
        cache['path'] = None
        cache['s_idx'] = 0
        cache['e_idx'] = -1
        return meet, meet

    # otherwise advance
    cache['s_idx'] = new_s_idx
    cache['e_idx'] = new_e_idx
    return path[new_s_idx], path[new_e_idx]

# move functions array
move_funcs = [dfs_make_move, bfs_make_move, a_star_make_move]
#########
#########
#########

# Array of move functions
move_funcs = [dfs_make_move, bfs_make_move, a_star_make_move]

def get_random_int(min_val: int = 0, max_val: int = 200) -> int:
    return random.randint(min_val, max_val)

def init_grid(num_walls: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    global grid, walls, path_caches
    # Reset grid
    grid = [[CellState.EMPTY for _ in range(COLS)] for _ in range(ROWS)]
    walls = []

    # Place walls
    remaining = num_walls
    while remaining > 0:
        p = (get_random_int(0, ROWS-1), get_random_int(0, COLS-1))
        if grid[p[0]][p[1]] == CellState.EMPTY:
            walls.append(p)
            grid[p[0]][p[1]] = CellState.WALL
            remaining -= 1

    # Place player
    player_pos = None
    while not player_pos:
        p = (get_random_int(0, ROWS-1), get_random_int(0, COLS-1))
        if grid[p[0]][p[1]] == CellState.EMPTY:
            grid[p[0]][p[1]] = CellState.PLAYER
            player_pos = p

    # Place target
    target_pos = None
    while not target_pos:
        p = (get_random_int(0, ROWS-1), get_random_int(0, COLS-1))
        if grid[p[0]][p[1]] == CellState.EMPTY:
            grid[p[0]][p[1]] = CellState.TARGET
            target_pos = p

    # Clear path caches so algorithms will recompute on first move
    path_caches = {
        'dfs': {'path': None, 'idx': 0, 'goal': None},
        'bfs': {'path': None, 'idx': 0, 'goal': None},
        'astar': {'path': None, 'idx': 0, 'goal': None},
        'bidi': {'path': None, 's_idx': 0, 'e_idx': -1, 'goal': None},
    }

    return player_pos, target_pos

def valid_move(pos: Tuple[int, int], next_pos: Tuple[int, int]) -> bool:
    if (pos[0] >= ROWS or pos[0] < 0 or pos[1] >= COLS or pos[1] < 0 or
        next_pos[0] >= ROWS or next_pos[0] < 0 or next_pos[1] >= COLS or next_pos[1] < 0):
        return False

    if grid[pos[0]][pos[1]] == CellState.WALL or grid[next_pos[0]][next_pos[1]] == CellState.WALL:
        return False

    # ensure orthogonal single-step move
    if (pos[0] - next_pos[0])**2 + (pos[1] - next_pos[1])**2 > 1:
        return False
    return True

def load_image(path: str, size: Tuple[int, int], fallback_color: Tuple[int, int, int]) -> pygame.Surface:
    try:
        image = pygame.image.load(path)
        image = pygame.transform.scale(image, size)
        return image
    except Exception as e:
        # handle FileNotFoundError and pygame.error
        print(f"Unable to load image {path}! Error: {e}")
        surf = pygame.Surface(size)
        surf.fill(fallback_color)
        return surf

def init_pygame():
    global screen, player_image, target_image, wall_image, victory_image
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Discrete Mathematics Project")

    # Load and scale images (if not found, colored squares are used)
    player_image = load_image("Player.png", (TILE_SIZE, TILE_SIZE), PLAYER_COLOR)
    target_image = load_image("Target.png", (TILE_SIZE, TILE_SIZE), TARGET_COLOR)
    wall_image = load_image("Wall.png", (TILE_SIZE, TILE_SIZE), WALL_COLOR)
    victory_image = load_image("Victory.png", (490, SCREEN_HEIGHT), VICTORY_COLOR)

    return True

def close_pygame():
    global screen, player_image, target_image, wall_image, victory_image
    player_image = None
    target_image = None
    wall_image = None
    victory_image = None
    pygame.quit()

async def main():
    global grid, walls
    if not init_pygame():
        print("Pygame could not initialize!")
        return

    # Get algorithm choice
    print("Which algorithm do you want to run? press the corresponding number:")
    print("1. DFS\n2. BFS\n3. A*\n4. Bidirectional A*")
    try:
        alg_num = int(input().strip()) - 1
    except Exception:
        alg_num = 0
    if alg_num not in (0,1,2,3):
        alg_num = 0
    print("Thanks! Now let's see how your algorithm in action!")

    player_pos, target_pos = init_grid(WALL_NUM)
    clock = pygame.time.Clock()

    running = True
    while running:
        start_time = pygame.time.get_ticks()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Clear screen
        screen.fill(BACKGROUND_COLOR)

        # Draw grid lines
        for row in range(ROWS):
            for col in range(COLS):
                rect = pygame.Rect(col * TILE_SIZE, row * TILE_SIZE, TILE_SIZE, TILE_SIZE)
                pygame.draw.rect(screen, GRID_COLOR, rect, 1)

        # Draw walls
        for w in walls:
            wall_rect = pygame.Rect(w[1] * TILE_SIZE, w[0] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
            screen.blit(wall_image, wall_rect)

        # Draw player
        player_rect = pygame.Rect(player_pos[1] * TILE_SIZE, player_pos[0] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
        screen.blit(player_image, player_rect)

        # Draw target
        target_rect = pygame.Rect(target_pos[1] * TILE_SIZE, target_pos[0] * TILE_SIZE, TILE_SIZE, TILE_SIZE)
        screen.blit(target_image, target_rect)

        pygame.display.flip()

        if player_pos == target_pos:
            print("NICE JOB! Maybe you can pass this class after all.")
            victory_rect = pygame.Rect(70, 0, 490, SCREEN_HEIGHT)
            screen.blit(victory_image, victory_rect)
            pygame.display.flip()
            await asyncio.sleep(0.8)  # 800ms delay

            # Start anew
            player_pos, target_pos = init_grid(WALL_NUM)
            continue

        # Make move
        new_player_pos = player_pos
        new_target_pos = target_pos
        if alg_num < 3:
            new_player_pos = move_funcs[alg_num](grid, player_pos, target_pos)
        else:
            new_player_pos, new_target_pos = bidi_a_star_make_move(grid, player_pos, target_pos)

        # Validate moves
        if valid_move(player_pos, new_player_pos):
            grid[player_pos[0]][player_pos[1]] = CellState.EMPTY
            player_pos = new_player_pos
            grid[player_pos[0]][player_pos[1]] = CellState.PLAYER
        else:
            print(f"Player: Invalid move! Can't make this move {player_pos} -> {new_player_pos}! I'm going to stay here for this turn!")

        if valid_move(target_pos, new_target_pos):
            grid[target_pos[0]][target_pos[1]] = CellState.EMPTY
            target_pos = new_target_pos
            grid[target_pos[0]][target_pos[1]] = CellState.TARGET
        else:
            print(f"Target: Invalid move! Can't make this move {target_pos} -> {new_target_pos}! I'm going to stay here for this turn!")

        # Cap frame rate
        elapsed = pygame.time.get_ticks() - start_time
        delay = max(0, (1000 / FPS) - elapsed)
        await asyncio.sleep(delay / 1000)

    close_pygame()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
