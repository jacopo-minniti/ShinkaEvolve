import os
import json
import torch
import random
from collections import deque
import argparse

def generate_maze(height, width):
    """Generates a maze using recursive backtracking.
    0 = wall, 1 = empty space.
    """
    # Initialize with all walls
    maze = torch.zeros((height, width), dtype=torch.int8)
    
    # Starting cell (must be odd coordinates for the algorithm to work with the wall padding)
    start_x, start_y = 1, 1
    maze[start_y, start_x] = 1
    
    stack = [(start_x, start_y)]
    
    while stack:
        x, y = stack[-1]
        
        # Directions: Up, Down, Left, Right (step 2 to jump over walls)
        directions = [(0, -2), (0, 2), (-2, 0), (2, 0)]
        random.shuffle(directions)
        
        found_next = False
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            if 0 < nx < width - 1 and 0 < ny < height - 1 and maze[ny, nx] == 0:
                maze[ny, nx] = 1
                maze[y + dy // 2, x + dx // 2] = 1 # Remove wall between
                stack.append((nx, ny))
                found_next = True
                break
        
        if not found_next:
            stack.pop()
            
    return maze

def solve_maze_bfs(maze, start, goal):
    """Finds the shortest path using BFS.
    Returns path list of (row, col) or None.
    """
    rows, cols = maze.shape
    q = deque([start])
    visited = set([start])
    parent = {start: None}
    
    while q:
        r, c = q.popleft()
        
        if (r, c) == goal:
            path = []
            curr = goal
            while curr:
                path.append(curr)
                curr = parent[curr]
            return path[::-1]
            
        # Directions: Up, Down, Left, Right
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and maze[nr, nc] == 1 and (nr, nc) not in visited:
                visited.add((nr, nc))
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))
    return None

def get_action_from_delta(curr, next_pos):
    """Returns action index (0: U, 1: D, 2: L, 3: R)"""
    dr = next_pos[0] - curr[0]
    dc = next_pos[1] - curr[1]
    if dr == -1: return 0 # Up
    if dr == 1: return 1 # Down
    if dc == -1: return 2 # Left
    if dc == 1: return 3 # Right
    return -1

def get_local_crop(maze, pos, obs_size, goal_pos):
    """
    Extracts a local square crop centered at pos.
    Channels: 
    0: Walls (1 if wall, 0 if free)
    1: Goal (1 if goal is here, 0 otherwise)
    
    Returns tensor of shape (2, obs_size, obs_size)
    """
    rows, cols = maze.shape
    half = obs_size // 2
    r, c = pos
    
    crop = torch.zeros((2, obs_size, obs_size), dtype=torch.float32)
    
    # Pad maze with walls for out-of-bounds extraction
    pad_maze = torch.zeros((rows + 2 * half, cols + 2 * half), dtype=torch.int8) # 0 is wall here for ease? 
    # Actually, in maze: 0=wall, 1=free.
    # Let's make pad_maze consistent: 0=wall.
    
    # Place actual maze in center of padded
    pad_maze[half:rows+half, half:cols+half] = maze
    
    # Extract window
    # padded coords: r+half, c+half is the center
    # slice: (r+half)-half : (r+half)+half+1
    #      = r : r+obs_size
    window = pad_maze[r:r+obs_size, c:c+obs_size]
    
    # Channel 0: Walls. pad_maze has 0=wall, 1=free. We want 1=wall, 0=free.
    crop[0] = (window == 0).float()
    
    # Channel 1: Goal
    # Goal is at goal_pos in original coords.
    # In window coords (relative to r-half, c-half):
    # gr = goal_r - (r - half)
    # gc = goal_c - (c - half)
    gr = goal_pos[0] - (r - half)
    gc = goal_pos[1] - (c - half)
    
    if 0 <= gr < obs_size and 0 <= gc < obs_size:
        crop[1, gr, gc] = 1.0
        
    return crop

def generate_dataset(args):
    """Generates train and test datasets."""
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    splits = {'train': args.n_train, 'test': args.n_test}
    
    for split_name, n_samples in splits.items():
        data = []
        print(f"Generating {split_name} split ({n_samples} samples)...")
        
        count = 0
        while count < n_samples:
            # Random size (must be odd)
            size = random.randint(args.maze_size_min, args.maze_size_max)
            if size % 2 == 0: size += 1
            
            maze = generate_maze(size, size)
            
            # Find all free cells
            free_cells = [(r, c) for r in range(size) for c in range(size) if maze[r, c] == 1]
            if len(free_cells) < 2: continue
            
            start_pos = random.choice(free_cells)
            goal_pos = random.choice(free_cells)
            
            if start_pos == goal_pos: continue
            
            # Solve
            path = solve_maze_bfs(maze, start_pos, goal_pos)
            
            if not path or len(path) < args.min_path_len: continue
            
            # Create episode
            episode = []
            for i in range(len(path) - 1):
                curr = path[i]
                next_pos = path[i+1]
                
                obs = get_local_crop(maze, curr, args.obs_size, goal_pos)
                action = get_action_from_delta(curr, next_pos)
                done = (next_pos == goal_pos)
                
                episode.append({
                    'obs': obs, # Tensor
                    'pos': curr,
                    'action': action,
                    'distance': len(path) - 1 - i,
                    'done': False # We record the step TAKEN. At this step done is false.
                })
            
            # For the very last step, we are AT the goal?
            # The prompt says: "done: whether goal has been reached at this step"
            # Usually in RL: obs_t, action_t, reward_t, done_t, obs_{t+1}
            # Here "episode list of time steps".
            # Let's say we include the final state observation where done=True, if useful?
            # Prompt says "action taken by oracle". Oracle stops at goal.
            # So the last item in 'path' is goal.
            # The loop goes up to len(path)-1, so we cover the transition INTO goal.
            # The last transition: curr is path[-2], next is path[-1] (goal).
            # The step records obs at path[-2], action to get to goal, and done? 
            # Usually done is true AFTER the action.
            # Let's verify standard: obs, act -> new_state, reward, done.
            # If we store (obs, action, done), "done" usually means "did this action lead to termination?"
            # Yes, for the last step, done=True.
            
            episode[-1]['done'] = True
            
            data.append({
                'maze_id': count,
                'maze_grid': maze,
                'start_pos': start_pos,
                'goal_pos': goal_pos,
                'episode': episode
            })
            count += 1
            
        torch.save(data, os.path.join(args.out_dir, f"{split_name}.pt"))
        
    # Save meta
    meta = vars(args)
    with open(os.path.join(args.out_dir, "meta.json"), 'w') as f:
        json.dump(meta, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/maze_quick")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--n_test", type=int, default=100)
    parser.add_argument("--maze_size_min", type=int, default=9)
    parser.add_argument("--maze_size_max", type=int, default=15)
    parser.add_argument("--obs_size", type=int, default=7)
    parser.add_argument("--min_path_len", type=int, default=5)
    args = parser.parse_args()
    
    generate_dataset(args)
