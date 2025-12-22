
import torch
import argparse
import random
import os
import matplotlib
matplotlib.use('Agg') # Ensure headless plotting
import matplotlib.pyplot as plt

def visualize(data_path):
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return

    try:
        data = torch.load(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if not data:
        print("Dataset empty.")
        return

    # Pick random sample
    sample = random.choice(data)
    maze = sample['maze_grid'] # 0=wall, 1=free
    start = sample['start_pos']
    goal = sample['goal_pos']
    
    plt.figure(figsize=(6, 6))
    
    # Plot maze (0=wall=black, 1=free=white)
    plt.imshow(maze.numpy(), cmap='gray')
    
    # Plot Start (Green) and Goal (Red)
    # scatter takes (x, y) -> (col, row)
    plt.scatter(start[1], start[0], c='green', s=100, label='Start')
    plt.scatter(goal[1], goal[0], c='red', s=100, label='Goal')
    
    # Plot path
    path_rows = [step['pos'][0] for step in sample['episode']]
    path_cols = [step['pos'][1] for step in sample['episode']]
    path_rows.insert(0, start[0])
    path_cols.insert(0, start[1])
    path_rows.append(goal[0])
    path_cols.append(goal[1])

    plt.plot(path_cols, path_rows, c='blue', alpha=0.5, linewidth=2, label='Solution')
    
    plt.legend()
    plt.title(f"Maze ID: {sample['maze_id']}")
    plt.axis('off')
    
    out_file = os.path.join(os.path.dirname(data_path), f"viz_maze_{sample['maze_id']}.png")
    plt.savefig(out_file)
    print(f"Saved visualization to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="Path to train.pt")
    args = parser.parse_args()
    
    visualize(args.data_path)
