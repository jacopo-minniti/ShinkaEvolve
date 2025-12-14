from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

# Configure the job execution environment
job_config = LocalJobConfig(eval_program_path="examples/circle_packing/evaluate.py")

# Configure the evolution database
db_config = DatabaseConfig(
    archive_size=20,
    num_archive_inspirations=4,
    num_islands=2,
    migration_interval=10,
)

# Configure the evolution parameters
evo_config = EvolutionConfig(
    num_generations=10,
    max_parallel_jobs=1,
    # USE YOUR LOCAL MODEL HERE
    # Format: local-<MODEL_ID>-<URL>
    llm_models=["local-Qwen/Qwen2.5-3B-Instruct-http://localhost:8000/v1"],
    # USE GEMINI EMBEDDINGS
    embedding_model="gemini-embedding-001",
    init_program_path="examples/circle_packing/initial.py",
    language="python",
    task_sys_msg="You are optimizing circle packing...",
)

# Run the evolution
runner = EvolutionRunner(
    evo_config=evo_config,
    job_config=job_config,
    db_config=db_config,
)

runner.run()
