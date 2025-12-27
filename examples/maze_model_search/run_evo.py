from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig
import os

# Training and evaluation configuration parameters
TRAIN_STEPS = 2000  # Number of gradient update steps
MAX_PARAMS = 100_000  # Maximum number of model parameters
OBS_SIZE = 7  # Size of the observation window
MAZE_SIZE_MAX = 15  # Maximum maze size (used to determine max episode steps)
BATCH_SIZE = 32

# Configure the job execution environment
# We point to our new evaluate.py and pass the configuration parameters
job_config = LocalJobConfig(
    eval_program_path="examples/maze_model_search/evaluate.py",
    extra_cmd_args={
        "train_steps": TRAIN_STEPS,
        "max_params": MAX_PARAMS,
        "obs_size": OBS_SIZE,
        "maze_size_max": MAZE_SIZE_MAX,
    }
)

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

task_sys_msg = f"""You are an expert in deep learning and reinforcement learning architectures.

**Task**: Design a PyTorch model (EvolvedModel) that learns to solve mazes under partial observability using **imitation learning on expert trajectories**.

**Training Paradigm**:
The model is trained on SEQUENCES of expert demonstrations:
- Each training batch contains {BATCH_SIZE} episodes (expert trajectories) processed in parallel
- Episodes are processed step-by-step through time, allowing the model to maintain temporal state
- Your model's forward() is called sequentially: forward(t=0), forward(t=1), ..., forward(t=T)
- This sequential processing enables models with memory to learn long-term strategies

**Input**:
A batch of observations of shape (B, 2, {OBS_SIZE}, {OBS_SIZE}).
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
- Parameter count must be < ({MAX_PARAMS:,} parameters.
- Training budget: {TRAIN_STEPS:,} gradient update steps
- The model must converge quickly with this limited budget

**Key Insight**:
Under partial observability, the agent cannot see the entire maze. Memory of past observations is crucial for navigation. Consider how your architecture can integrate information over time to build a mental representation of the environment.

The evaluation script handles all training and evaluation. You only control the architecture and loss function.
"""

evo_config = EvolutionConfig(
    task_sys_msg=task_sys_msg,
    patch_types=["diff", "full", "cross"],
    patch_type_probs=[0.6, 0.3, 0.1],
    num_generations=100,
    max_parallel_jobs=1,
    max_patch_resamples=3,
    max_patch_attempts=3,
    job_type="local",
    language="python",
    llm_models=["Qwen/Qwen3-30B-A3B-Thinking-2507"],
    llm_kwargs=dict(
        temperatures=[0.7],
        max_tokens=20000,
    ),
    meta_rec_interval=10,
    meta_llm_kwargs=dict(temperatures=[0.0], max_tokens=2000),
    embedding_model="gemini-embedding-001",
    code_embed_sim_threshold=0.995,
    init_program_path="examples/maze_model_search/initial.py",
    results_dir="results/maze_qwen3-30B",
)

if __name__ == "__main__":
    evo_runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        verbose=True,
    )
    evo_runner.run()
