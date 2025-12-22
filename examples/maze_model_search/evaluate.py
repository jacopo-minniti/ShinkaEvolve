
import os
import argparse
import numpy as np
import torch
import json
from typing import Dict, Any, List, Tuple, Union

# Standard import - assuming running in an environment where imports work
from initial import EvolvedModel
from examples.maze_model_search.generate_data import get_local_crop

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
            
            batch_dict = {
                'obs': obs_batch,
                'action': action_batch,
                'target': action_batch
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

def main(args):
    # Load data
    train_path = os.path.join(args.data_dir, "train.pt")
    test_path = os.path.join(args.data_dir, "test.pt")
    
    if not os.path.exists(train_path):
        print(json.dumps({"fitness": 0.0, "error": "Data not found"}))
        return

    train_data = torch.load(train_path)
    test_data = torch.load(test_path)
    
    # Instantiate model
    model = EvolvedModel()
        
    # Check params
    param_count = get_stats(model)
    
    # Train
    success, train_result = train_model(model, train_data, args)
    if not success:
         # train_result is error string
         print(json.dumps({"fitness": 0.0, "error": train_result, "stats": {"param_count": param_count}}))
         return
    
    train_stats = train_result
         
    # Evaluate
    success, eval_result = evaluate_model(model, test_data, args)
    if not success:
         # eval_result is error string
         print(json.dumps({"fitness": 0.0, "error": eval_result, "stats": {**train_stats, "param_count": param_count}}))
         return

    eval_stats = eval_result
    
    fitness = eval_stats["test_success_rate"]
    
    results = {
        "fitness": fitness,
        "stats": {**train_stats, **eval_stats, "param_count": param_count}
    }
    print(json.dumps(results))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/maze_model_search")
    parser.add_argument("--train_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_params", type=int, default=1_000_000)
    parser.add_argument("--obs_size", type=int, default=7)
    parser.add_argument("--maze_size_max", type=int, default=15)
    
    args = parser.parse_args()
    main(args)
