import os
import argparse
import numpy as np
import torch
import json
import builtins
import importlib.util
import sys
from typing import Dict, Any, List, Tuple, Union

from shinka.core.wrap_eval import save_json_results


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

def get_stats(model):
    return sum(p.numel() for p in model.parameters())

def train_model(model, train_data, args) -> Tuple[bool, Union[Dict, str]]:
    """
    Trains the model for a fixed number of steps.
    Returns (success, stats_dict) or (False, error_message).
    """
    if args.max_params and get_stats(model) > args.max_params:
         return False, f"Max params exceeded: {get_stats(model)} > {args.max_params}"

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    all_steps = []
    for ep in train_data:
        all_steps.extend(ep['episode'])
    
    random_indices = torch.randperm(len(all_steps))
    
    model.train()
    total_loss = 0
    num_batches = 0
    current_step = 0
    
    while current_step < args.train_steps:
        for i in range(0, len(all_steps), args.batch_size):
            if current_step >= args.train_steps: break
            
            indices = random_indices[i : i + args.batch_size]
            if len(indices) < args.batch_size: continue
            
            batch_steps = [all_steps[idx] for idx in indices]
            
            obs_batch = torch.stack([s['obs'] for s in batch_steps])
            action_batch = torch.tensor([s['action'] for s in batch_steps], dtype=torch.long)
            distance_batch = torch.tensor([s['distance'] for s in batch_steps], dtype=torch.float32)
            
            batch_dict = {
                'obs': obs_batch,
                'action': action_batch,
                'target': action_batch,
                'distance': distance_batch
            }
            
            optimizer.zero_grad()
            
            if hasattr(model, 'compute_loss'):
                outputs = model(obs_batch)
                loss = model.compute_loss(batch_dict, outputs)
            else:
                outputs = model(obs_batch)
                if isinstance(outputs, torch.Tensor) and outputs.numel() == 1:
                    loss = outputs
                elif isinstance(outputs, torch.Tensor):
                    loss = torch.nn.functional.cross_entropy(outputs, action_batch)
                else:
                    return False, "Model output format unclear and no compute_loss found"
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            current_step += 1

    return True, {"train_loss_final": total_loss / num_batches if num_batches else 0.0}

def evaluate_model(model, test_data, args) -> Tuple[bool, Union[Dict, str]]:
    """
    Evaluates the model.
    Returns (success, stats_dict) or (False, error_message).
    """
    model.eval()
    successes = 0
    total = len(test_data)
    total_steps_success = 0
    
    with torch.no_grad():
        for episode in test_data:
            maze = episode['maze_grid']
            curr_pos = episode['start_pos']
            goal_pos = episode['goal_pos']
            
            max_steps = args.maze_size_max * 4
            
            for step in range(max_steps):
                if curr_pos == goal_pos:
                    successes += 1
                    total_steps_success += step
                    break
                
                obs = get_local_crop(maze, curr_pos, args.obs_size, goal_pos).unsqueeze(0)
                
                outputs = model(obs)
                
                if isinstance(outputs, torch.Tensor):
                    action = torch.argmax(outputs, dim=1).item()
                elif hasattr(outputs, 'logits'):
                    action = torch.argmax(outputs.logits, dim=1).item()
                else:
                    action = 0 
                    
                # Execute action: 0:U, 1:D, 2:L, 3:R
                dr, dc = 0, 0
                if action == 0: dr, dc = -1, 0
                elif action == 1: dr, dc = 1, 0
                elif action == 2: dr, dc = 0, -1
                elif action == 3: dr, dc = 0, 1
                
                nr, nc = curr_pos[0] + dr, curr_pos[1] + dc
                
                if 0 <= nr < maze.shape[0] and 0 <= nc < maze.shape[1] and maze[nr, nc] == 1:
                    curr_pos = (nr, nc)
    
    success_rate = successes / total if total else 0
    avg_steps = total_steps_success / successes if successes else 0
    
    return True, {
        "test_success_rate": success_rate,
        "avg_steps_to_goal": avg_steps
    }

def load_module_from_path(path):
    # Get module name from file name
    module_name = os.path.basename(path).replace(".py", "")
    
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    return None

def main(args):
    results_dir = args.results_dir or "results"
    os.makedirs(results_dir, exist_ok=True)

    # Load data
    train_path = os.path.join(args.data_dir, "train.pt")
    test_path = os.path.join(args.data_dir, "test.pt")
    
    if not os.path.exists(train_path):
        save_json_results(
            results_dir=results_dir,
            metrics={"combined_score": 0.0, "public": {}, "private": {}},
            correct=False,
            error="Data not found",
        )
        return

    train_data = torch.load(train_path)
    test_data = torch.load(test_path)
    
    # Load EvolvedModel from program_path
    if not args.program_path:
        # Fallback to local EvolvedModel if it exists (e.g. for testing this script directly with a class defined here)
        # But currently no class is defined here.
        save_json_results(
            results_dir=results_dir,
            metrics={"combined_score": 0.0, "public": {}, "private": {}},
            correct=False,
            error="No program_path provided",
        )
        return

    try:
        module = load_module_from_path(args.program_path)
        if module is None or not hasattr(module, "EvolvedModel"):
             save_json_results(
                 results_dir=results_dir,
                 metrics={"combined_score": 0.0, "public": {}, "private": {}},
                 correct=False,
                 error=f"Failed to load EvolvedModel from {args.program_path}",
             )
             return
        EvolvedModel = module.EvolvedModel
    except Exception as e:
        save_json_results(
            results_dir=results_dir,
            metrics={"combined_score": 0.0, "public": {}, "private": {}},
            correct=False,
            error=f"Error loading module: {str(e)}",
        )
        return

    # Instantiate model
    try:
        model = EvolvedModel()
    except Exception as e:
        save_json_results(
            results_dir=results_dir,
            metrics={"combined_score": 0.0, "public": {}, "private": {}},
            correct=False,
            error=f"Error instantiating EvolvedModel: {str(e)}",
        )
        return
        
    # Check params
    try:
        param_count = get_stats(model)
    except Exception as e:
        save_json_results(
            results_dir=results_dir,
            metrics={"combined_score": 0.0, "public": {}, "private": {}},
            correct=False,
            error=f"Error counting params: {str(e)}",
        )
        return
    
    # Train
    success, train_result = train_model(model, train_data, args)
    if not success:
         # train_result is error string
         save_json_results(
             results_dir=results_dir,
             metrics={
                 "combined_score": 0.0,
                 "public": {"param_count": param_count},
                 "private": {},
             },
             correct=False,
             error=train_result,
         )
         return
    
    train_stats = train_result
         
    # Evaluate
    success, eval_result = evaluate_model(model, test_data, args)
    if not success:
         # eval_result is error string
         save_json_results(
             results_dir=results_dir,
             metrics={
                 "combined_score": 0.0,
                 "public": {**train_stats, "param_count": param_count},
                 "private": {},
             },
             correct=False,
             error=eval_result,
         )
         return

    eval_stats = eval_result
    
    fitness = eval_stats["test_success_rate"]

    metrics = {
        "combined_score": float(fitness),
        "public": {**train_stats, **eval_stats, "param_count": param_count},
        "private": {},
    }
    save_json_results(
        results_dir=results_dir,
        metrics=metrics,
        correct=True,
        error=None,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/maze_quick")
    parser.add_argument("--train_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_params", type=int, default=300_000)
    parser.add_argument("--obs_size", type=int, default=7)
    parser.add_argument("--maze_size_max", type=int, default=15)
    parser.add_argument("--program_path", type=str, required=False, help="Path to the python script containing EvolvedModel")
    parser.add_argument("--results_dir", type=str, required=False, help="Directory to save results")
    
    args = parser.parse_args()
    main(args)
