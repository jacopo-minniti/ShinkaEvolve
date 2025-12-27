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
max_params = 100_000
train_steps = 2000  # Number of gradient update steps
batch_size = 16  # Number of episodes per training batch

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

**Task**: Design a PyTorch model (EvolvedModel) that learns to solve mazes under partial observability using **imitation learning on expert trajectories**.

**Training Paradigm**:
The model is trained on SEQUENCES of expert demonstrations:
- Each training batch contains {batch_size} episodes (expert trajectories) processed in parallel
- Episodes are processed step-by-step through time, allowing the model to maintain temporal state
- Your model's forward() is called sequentially: forward(t=0), forward(t=1), ..., forward(t=T)
- This sequential processing enables models with memory to learn long-term strategies

**Input**:
Observations of shape (B, 2, obs_size, obs_size) where obs_size=7:
- Channel 0: Wall map (1=wall, 0=empty)
- Channel 1: Goal map (1=goal visible, 0=not visible)
- The agent is always at the center (position 3,3 in the 7×7 grid)

**Output**:
Action logits for 4 actions: [Up, Down, Left, Right]
Must be exposed as one of:
  - A single Tensor of shape (B, 4)
  - A tuple/list where action logits are the first element
  - A dict with key "logits"

**Required Methods**:
1. `__init__(self)`: Initialize your architecture
2. `forward(self, x)`: Process one observation, return action logits
3. `compute_loss(self, batch, outputs)`: Return scalar loss tensor
   - batch dict contains: 'obs', 'action', 'target', 'distance', 'mask'
   - 'mask' indicates valid (non-padded) timesteps: 1.0 = valid, 0.0 = padding
4. **For stateful models only (if you use recurrent layers)**:
   - `reset_state(self)`: Reset hidden states to None at the start of new episodes
   - Store hidden state as instance variable (e.g., self.hidden_state)
   - In forward(), initialize hidden state if None, otherwise use stored state
   - The hidden state's batch dimension automatically handles parallel episodes

**Constraints**:
- Parameter count must be < {max_params}
- Training budget: {train_steps} gradient update steps
- The model must converge quickly with this limited budget

**Key Insight**:
Under partial observability, the agent cannot see the entire maze. Memory of past observations is crucial for navigation. Consider how your architecture can integrate information over time to build a mental representation of the environment.

The evaluation script handles all training and evaluation. You only control the architecture and loss function.
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
    train_steps=train_steps,
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
