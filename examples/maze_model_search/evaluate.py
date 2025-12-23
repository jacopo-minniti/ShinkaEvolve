import argparse
import importlib.util
import logging
import os
import sys
import traceback
from typing import Dict

import torch

from shinka.core.wrap_eval import save_json_results

logger = logging.getLogger(__name__)


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

def train_model(model, train_data, args, device) -> Dict:
    """
    Trains the model for a fixed number of steps.
    """
    if args.max_params and get_stats(model) > args.max_params:
        raise ValueError(
            f"Max params exceeded: {get_stats(model)} > {args.max_params}"
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    all_steps = []
    for ep in train_data:
        all_steps.extend(ep['episode'])
    
    random_indices = torch.randperm(len(all_steps))
    
    model.train()
    total_loss = 0
    num_batches = 0
    current_step = 0

    if not hasattr(model, "compute_loss"):
        raise ValueError("Model must implement compute_loss(batch, outputs)")

    logger.info(
        "Training start: total_steps=%d, batch_size=%d, device=%s",
        args.train_steps,
        args.batch_size,
        device,
    )
    logger.info("Training data: episodes=%d, steps=%d", len(train_data), len(all_steps))

    while current_step < args.train_steps:
        for i in range(0, len(all_steps), args.batch_size):
            if current_step >= args.train_steps: break
            
            indices = random_indices[i : i + args.batch_size]
            if len(indices) < args.batch_size: continue
            
            batch_steps = [all_steps[idx] for idx in indices]
            
            obs_batch = torch.stack([s['obs'] for s in batch_steps]).to(device)
            action_batch = torch.tensor(
                [s['action'] for s in batch_steps], dtype=torch.long, device=device
            )
            distance_batch = torch.tensor(
                [s['distance'] for s in batch_steps], dtype=torch.float32, device=device
            )
            
            batch_dict = {
                'obs': obs_batch,
                'action': action_batch,
                'target': action_batch,
                'distance': distance_batch
            }
            
            optimizer.zero_grad()

            outputs = model(obs_batch)
            loss = model.compute_loss(batch_dict, outputs)
            if not isinstance(loss, torch.Tensor):
                raise ValueError("compute_loss must return a torch.Tensor")
            if loss.numel() != 1:
                raise ValueError("compute_loss must return a scalar loss tensor")
            if not torch.isfinite(loss).all():
                raise ValueError(f"Loss is not finite: {loss.item()}")
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            current_step += 1

            if current_step % 100 == 0:
                avg_loss = total_loss / max(num_batches, 1)
                logger.info("Training progress: step=%d/%d avg_loss=%.6f",
                            current_step, args.train_steps, avg_loss)

    return {"train_loss_final": total_loss / num_batches if num_batches else 0.0}

def evaluate_model(model, test_data, args, device) -> Dict:
    """
    Evaluates the model.
    """
    model.eval()
    successes = 0
    total = len(test_data)
    total_steps_success = 0

    logger.info("Evaluation start: episodes=%d", total)

    with torch.no_grad():
        for ep_idx, episode in enumerate(test_data):
            maze = episode['maze_grid']
            curr_pos = episode['start_pos']
            goal_pos = episode['goal_pos']
            
            max_steps = args.maze_size_max * 4
            
            for step in range(max_steps):
                if curr_pos == goal_pos:
                    successes += 1
                    total_steps_success += step
                    break
                
                obs = (
                    get_local_crop(maze, curr_pos, args.obs_size, goal_pos)
                    .unsqueeze(0)
                    .to(device)
                )
                
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

            if (ep_idx + 1) % 50 == 0:
                logger.info("Evaluation progress: episode=%d/%d successes=%d",
                            ep_idx + 1, total, successes)

    success_rate = successes / total if total else 0
    avg_steps = total_steps_success / successes if successes else 0

    logger.info("Evaluation done: success_rate=%.6f avg_steps=%.2f",
                success_rate, avg_steps)

    return {
        "test_success_rate": success_rate,
        "avg_steps_to_goal": avg_steps
    }

def load_module_from_path(path):
    # Get module name from file name
    module_name = os.path.basename(path).replace(".py", "")
    
    spec = importlib.util.spec_from_file_location(module_name, path)
    if not spec or not spec.loader:
        raise ImportError(f"Could not load spec for module at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def main(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - maze_eval - %(levelname)s - %(message)s",
    )
    results_dir = args.results_dir or "results"
    os.makedirs(results_dir, exist_ok=True)
    try:
        train_path = os.path.join(args.data_dir, "train.pt")
        test_path = os.path.join(args.data_dir, "test.pt")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Data not found at {train_path}")

        logger.info("Loading data: train=%s test=%s", train_path, test_path)
        train_data = torch.load(train_path)
        test_data = torch.load(test_path)
        logger.info("Data loaded: train_episodes=%d test_episodes=%d",
                    len(train_data), len(test_data))

        if not args.program_path:
            raise ValueError("No program_path provided")

        logger.info("Loading program: %s", args.program_path)
        module = load_module_from_path(args.program_path)
        if not hasattr(module, "EvolvedModel"):
            raise AttributeError(
                f"Failed to load EvolvedModel from {args.program_path}"
            )
        logger.info("Instantiating model")
        model = module.EvolvedModel()
        param_count = get_stats(model)
        logger.info("Model params: %d", param_count)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Using device: %s", device)
        model = model.to(device)

        train_stats = train_model(model, train_data, args, device)
        eval_stats = evaluate_model(model, test_data, args, device)

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
        logger.info("Results saved: %s", results_dir)
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc)
        traceback.print_exc()
        save_json_results(
            results_dir=results_dir,
            metrics={"combined_score": 0.0, "public": {}, "private": {}},
            correct=False,
            error=str(exc),
        )
        raise SystemExit(1) from exc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/maze_quick")
    parser.add_argument("--train_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_params", type=int, default=100_000)
    parser.add_argument("--obs_size", type=int, default=7)
    parser.add_argument("--maze_size_max", type=int, default=15)
    parser.add_argument("--program_path", type=str, required=False, help="Path to the python script containing EvolvedModel")
    parser.add_argument("--results_dir", type=str, required=False, help="Directory to save results")
    
    args = parser.parse_args()
    main(args)
