from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig
import os
import logging

# Configure logging to DEBUG level
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

job_config = LocalJobConfig(eval_program_path="examples/maze_model_search/evaluate.py")
training_epochs = 5
max_params = 100_000

parent_config = dict(
    parent_selection_strategy="power_law",
    exploitation_alpha=1.0,
    exploitation_ratio=0.2,
)

db_config = DatabaseConfig(
    db_path="evolution_maze.sqlite",
    num_islands=2,
    archive_size=40,
    elite_selection_ratio=0.3,
    num_archive_inspirations=4,
    num_top_k_inspirations=2,
    migration_interval=10,
    migration_rate=0.1,
    island_elitism=True,
    **parent_config,
)

task_sys_msg = f"""You are an expert in deep learning and reinforcement learning architectures.
Your goal is to design a PyTorch model (EvolvedModel) that can learn to solve mazes under partial observability.

Input:
A batch of observations of shape (B, 2, obs_size, obs_size).
- Channel 0: Wall map (1=wall, 0=empty)
- Channel 1: Goal map (1=goal, 0=empty)
The agent is always at the center of the observation.

Output:
Logits for 4 actions: Up, Down, Left, Right.

Constraints:
- You must define `class EvolvedModel(nn.Module)`.
- It must implement `compute_loss(self, batch, outputs)`.
- `forward` output must expose action logits in one of these forms:
  - a single Tensor of shape (B, 4)
  - a tuple/list where logits are the first element
  - a dict with key "logits"
- `compute_loss` must return a scalar `torch.Tensor` just like standard PyTorch losses.
- Parameter count must stay under the limit ({max_params}).
- Training budget is fixed ({training_epochs} epochs). Code efficient, fast-converging architectures.
- The training loop trains on SEQUENCES (episodes).
- If your model is stateful (e.g. RNN, GRU, LSTM):
    - You MUST implement `reset_state(self)` to clear the internal hidden states.
    - `reset_state` will be called at the beginning of each episode/batch.
    - Ensure your forward pass handles batches appropriately (e.g. broadcasting or keeping state shape (1, B, H)).

Be creative with:
- Temporal Logic (RNNs, LSTMs, GRUs) to integrate information over time.
- Auxiliary losses to guide the learning (e.g. distance prediction).
- Attention mechanisms.

The evaluation script handles the training loop and data loading. You primarily control the architecture and loss function definition.
"""

evo_config = EvolutionConfig(
    task_sys_msg=task_sys_msg,
    num_generations=100,
    max_parallel_jobs=1,
    max_patch_resamples=3,
    max_repair_attempts=3,
    job_type="local",
    language="python",
    llm_models=["Qwen/Qwen3-30B-A3B-Thinking-2507"],
    llm_kwargs=dict(
        temperatures=[0.7],
        max_tokens=20000,
    ),
    results_dir="results/maze_qwen3-30B-bias",
    max_params=max_params,
    training_epochs=training_epochs,
    num_previous_gen_errors=3,
)

if __name__ == "__main__":
    evo_runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        verbose=True,
    )
    evo_runner.run()
