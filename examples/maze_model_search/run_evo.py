
from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig
import os

# Configure the job execution environment
# We point to our new evaluate.py
job_config = LocalJobConfig(eval_program_path="examples/maze_model_search/evaluate.py")

# Parent selection strategy (can be same as main.py default or simplified)
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

task_sys_msg = """You are an expert in deep learning and reinforcement learning architectures.
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
- It must implement `compute_loss(self, batch, outputs)` or return loss from forward.
- Parameter count must stay under the limit (e.g. 1M).
- Training budget is fixed. Code efficient, fast-converging architectures.
- Be creative with:
    - Experience replay buffers (if you implement them inside the model/loss loop)
    - Memory (RNNs, GRUs, LSTMs) to handle partial observability
    - Attention mechanisms (Transformers)
    - Auxiliary losses (e.g. predicting distance to goal)

The evaluation script handles the training loop and data loading. You primarily control the architecture and loss function definition.
"""

evo_config = EvolutionConfig(
    task_sys_msg=task_sys_msg,
    patch_types=["diff", "full", "cross"],
    patch_type_probs=[0.6, 0.3, 0.1],
    num_generations=50, # Example
    max_parallel_jobs=4,
    max_patch_resamples=3,
    max_patch_attempts=3,
    job_type="local",
    language="python",
    llm_models=["Qwen/Qwen3-30B-A3B-Thinking-2507"], # Or whatever is default/available
    llm_kwargs=dict(
        temperatures=[0.7],
        max_tokens=4000,
    ),
    meta_rec_interval=10,
    meta_llm_kwargs=dict(temperatures=[0.0], max_tokens=2000),
    embedding_model="gemini-embedding-001",
    code_embed_sim_threshold=0.995,
    init_program_path="examples/maze_model_search/initial.py",
    results_dir="results/maze_search",
)

def main():
    # Ensure data exists
    if not os.path.exists("data/maze_model_search/train.pt"):
        print("Warning: Data file data/maze_model_search/train.pt not found.")
        print("Please run: python examples/maze_model_search/generate_data.py")
        
    evo_runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        verbose=True,
    )
    evo_runner.run()

if __name__ == "__main__":
    main()
