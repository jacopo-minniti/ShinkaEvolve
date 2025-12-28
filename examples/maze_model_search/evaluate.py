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
    pad_maze = torch.zeros((rows + 2 * half, cols + 2 * half), dtype=torch.int8)
    
    # Place actual maze in center of padded
    pad_maze[half:rows+half, half:cols+half] = maze
    
    # Extract window
    window = pad_maze[r:r+obs_size, c:c+obs_size]
    
    # Channel 0: Walls. pad_maze has 0=wall, 1=free. We want 1=wall, 0=free.
    crop[0] = (window == 0).float()
    
    # Channel 1: Goal
    # Goal is at goal_pos in original coords.
    # In window coords (relative to r-half, c-half):
    gr = goal_pos[0] - (r - half)
    gc = goal_pos[1] - (c - half)
    
    if 0 <= gr < obs_size and 0 <= gc < obs_size:
        crop[1, gr, gc] = 1.0
        
    return crop

def get_stats(model):
    return sum(p.numel() for p in model.parameters())

def train_model(model, train_data, args, device) -> Dict:
    """
    Trains the model using sequence-based imitation learning.
    Each batch contains multiple episodes processed in parallel through time.
    """
    if args.max_params and get_stats(model) > args.max_params:
        raise ValueError(
            f"Max params exceeded: {get_stats(model)} > {args.max_params}"
        )

    if hasattr(model, "compute_optimizer"):
        optimizer = model.compute_optimizer()
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Extract episodes from training data
    episodes = [ep['episode'] for ep in train_data]
    
    if not hasattr(model, "compute_loss"):
        raise ValueError("Model must implement compute_loss(batch, outputs)")

    # Calculate number of epochs to achieve target number of training steps
    # If we have N episodes and batch_size B, we get N/B batches per epoch
    batches_per_epoch = max(1, len(episodes) // args.batch_size)
    num_epochs = max(1, args.train_steps // batches_per_epoch)
    
    logger.info(
        "Training start: episodes=%d, batch_size=%d, batches_per_epoch=%d, epochs=%d, train_steps=%d, device=%s",
        len(episodes), args.batch_size, batches_per_epoch, num_epochs, args.train_steps, device,
    )
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for epoch in range(num_epochs):
        # Shuffle episodes each epoch
        indices = torch.randperm(len(episodes))
        
        for i in range(0, len(episodes), args.batch_size):
            idx_batch = indices[i : i + args.batch_size]
            if len(idx_batch) < args.batch_size:
                continue  # Skip incomplete batches for consistent batch size
                
            batch_episodes = [episodes[idx] for idx in idx_batch]
            
            # Reset model state for this batch of episodes
            if hasattr(model, "reset_state"):
                model.reset_state()
                
            # Find max sequence length in this batch
            max_len = max(len(ep) for ep in batch_episodes)
            batch_size = len(batch_episodes)
            
            # Accumulate loss over the sequence
            sequence_loss = 0.0
            num_valid_steps = 0
            
            # Iterate through timesteps
            for t in range(max_len):
                # Collect data for timestep t from all episodes in batch
                current_steps = []
                mask = []  # 1.0 if valid timestep, 0.0 if padding
                
                for ep in batch_episodes:
                    if t < len(ep):
                        current_steps.append(ep[t])
                        mask.append(1.0)
                    else:
                        # Padding: reuse last step to maintain tensor shapes
                        current_steps.append(ep[-1])
                        mask.append(0.0)
                
                # Create batched tensors
                obs_batch = torch.stack([s['obs'] for s in current_steps]).to(device)
                action_batch = torch.tensor(
                    [s['action'] for s in current_steps], dtype=torch.long, device=device
                )
                distance_batch = torch.tensor(
                    [s['distance'] for s in current_steps], dtype=torch.float32, device=device
                )
                mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device)
                
                step_batch_dict = {
                    'obs': obs_batch,
                    'action': action_batch,
                    'target': action_batch,
                    'distance': distance_batch,
                    'mask': mask_tensor  # Provide mask in case model wants to use it
                }
                
                # Forward pass
                outputs = model(obs_batch)
                
                # Compute loss for this timestep
                loss = model.compute_loss(step_batch_dict, outputs)
                
                # Validate loss
                if not isinstance(loss, torch.Tensor):
                    raise ValueError("compute_loss must return a torch.Tensor")
                if not torch.isfinite(loss).all():
                    raise ValueError(f"Loss is not finite: {loss.item()}")
                
                # Weight loss by proportion of valid (non-padded) samples at this timestep
                mask_weight = mask_tensor.mean()  # Fraction of valid samples
                if mask_weight > 0:
                    sequence_loss += loss * mask_weight
                    num_valid_steps += 1
            
            # Average loss over valid timesteps in the sequence
            if num_valid_steps > 0:
                sequence_loss = sequence_loss / num_valid_steps
            else:
                # Edge case: no valid steps (shouldn't happen with proper data)
                continue
            
            # Backpropagation through the entire sequence
            optimizer.zero_grad()
            sequence_loss.backward()
            optimizer.step()
            
            total_loss += sequence_loss.item()
            num_batches += 1

            if num_batches % 50 == 0:
                avg_loss = total_loss / num_batches
                logger.info(
                    "Training progress: Epoch %d/%d | Batch %d | Avg Loss %.6f",
                    epoch + 1, num_epochs, num_batches, avg_loss
                )

    final_avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    logger.info("Training complete: total_batches=%d, final_avg_loss=%.6f", num_batches, final_avg_loss)
    
    return {"train_loss_final": final_avg_loss}

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
            # Reset state at start of episode
            if hasattr(model, "reset_state"):
                model.reset_state()
                
            maze = episode['maze_grid']
            curr_pos = episode['start_pos']
            goal_pos = episode['goal_pos']
            
            max_steps = args.maze_size_max ** 2
            
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
                    logits = outputs
                elif hasattr(outputs, "logits"):
                    logits = outputs.logits
                elif isinstance(outputs, (tuple, list)) and outputs:
                    logits = outputs[0]
                elif isinstance(outputs, dict) and "logits" in outputs:
                    logits = outputs["logits"]
                else:
                    logits = None

                if logits is None:
                    logger.warning(
                        "Model outputs do not expose logits; defaulting action to 0. "
                        "Expected Tensor, .logits, tuple/list with logits first, or dict['logits']."
                    )
                    action = 0
                else:
                    action = torch.argmax(logits, dim=1).item()
                    
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
    parser.add_argument("--train_steps", type=int, default=2000, help="Number of gradient update steps")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of episodes per batch")
    parser.add_argument("--max_params", type=int, default=100_000)
    parser.add_argument("--obs_size", type=int, default=7)
    parser.add_argument("--maze_size_max", type=int, default=15)
    parser.add_argument("--program_path", type=str, required=False, help="Path to the python script containing EvolvedModel")
    parser.add_argument("--results_dir", type=str, required=False, help="Directory to save results")
    
    args = parser.parse_args()
    main(args)
