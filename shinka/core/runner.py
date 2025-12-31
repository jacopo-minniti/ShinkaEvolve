import logging
import time
import uuid
import yaml
import shutil
import numpy as np
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from dataclasses import dataclass, field, asdict

from rich.logging import RichHandler
from rich.console import Console

from shinka.launch import JobScheduler, JobConfig, ProcessWithLogging
from shinka.database import ProgramDatabase, DatabaseConfig, Program
from shinka.logo import print_gradient_logo

from shinka.sibs.agents import (
    FirstOrderPlanner,
    SecondOrderInitializer,
    DesignMutator,
    ImplementationAgent,
    ReflectionWriter
)
from shinka.sibs.schema import SecondOrderGenome

FOLDER_PREFIX = "gen"

MINIMAL_TEMPLATE_CODE = """# REGION_ALPHA_START
import torch
import torch.nn as nn
import torch.nn.functional as F

class EvolvedModel(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Initialize components

    def forward(self, x):
        # TODO: Implement forward pass
        pass
# REGION_ALPHA_END

# REGION_PHI_START
    def compute_loss(self, batch, outputs):
        # TODO: Implement loss computation
        return torch.tensor(0.0, requires_grad=True)
# REGION_PHI_END

# REGION_METRICS_START
    def compute_metrics(self, batch, outputs):
        # TODO: Implement behavioral metrics (independent of fitness)
        # return {"metric_name": value}
        return {}
# REGION_METRICS_END

# REGION_OMEGA_START
    def compute_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
# REGION_OMEGA_END
"""

@dataclass
class EvolutionConfig:
    task_sys_msg: Optional[str] = None
    num_generations: int = 10
    max_parallel_jobs: int = 2
    max_patch_resamples: int = 3
    job_type: str = "local"
    language: str = "python"
    llm_models: List[str] = field(default_factory=lambda: ["gpt-4o"])
    llm_kwargs: dict = field(default_factory=lambda: {})
    results_dir: Optional[str] = None
    use_text_feedback: bool = False
    # SIBS specific
    mutation_weights: Dict[str, float] = field(default_factory=lambda: {"Alpha": 0.8, "Omega": 0.05, "Phi": 0.15})
    max_repair_attempts: int = 2
    # Constraints
    max_params: int = 100_000
    train_steps: int = 2000
    num_previous_gen_errors: int = 3

@dataclass
class RunningJob:
    """Represents a running job in the queue."""
    job_id: Union[str, Any]
    exec_fname: str
    results_dir: str
    start_time: float
    generation: int
    parent_id: Optional[str]
    archive_insp_ids: List[str]
    top_k_insp_ids: List[str]
    meta_patch_data: Optional[dict]
    retry_count: int = 0
    # For repair, we need to know the genome we were trying to implement
    genome_yaml: Optional[str] = None
    # Track the job directory for artifact saving
    job_dir: Optional[str] = None

# Set up logging
logger = logging.getLogger(__name__)

class EvolutionRunner:
    def __init__(
        self,
        evo_config: EvolutionConfig,
        job_config: JobConfig,
        db_config: DatabaseConfig,
        verbose: bool = True,
    ):
        self.evo_config = evo_config
        self.job_config = job_config
        self.db_config = db_config
        self.verbose = verbose

        print_gradient_logo((255, 0, 0), (255, 255, 255))
        if evo_config.results_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.results_dir = f"results_{timestamp}"
        else:
            self.results_dir = Path(evo_config.results_dir)

        if self.verbose:
            # Create log file path in results directory
            Path(self.results_dir).mkdir(parents=True, exist_ok=True)
            log_filename = f"{self.results_dir}/evolution_run.log"

            # Set up logging with both console and file handlers
            # Respect existing level if set (e.g. from run_evo.py)
            current_level = logging.getLogger().getEffectiveLevel()
            if current_level == logging.NOTSET:
               current_level = logging.INFO
            
            logging.basicConfig(
                level=current_level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
                handlers=[
                    RichHandler(
                        show_time=False, show_level=False, show_path=False
                    ),
                    logging.FileHandler(
                        log_filename, mode="a", encoding="utf-8"
                    ),
                ],
                force=True
            )

        # Inject constraints into JobConfig for evaluate.py
        if self.evo_config.max_params:
            self.job_config.extra_cmd_args["max_params"] = self.evo_config.max_params
        if self.evo_config.train_steps:
             self.job_config.extra_cmd_args["train_steps"] = self.evo_config.train_steps

        # Job counter per generation
        self.generation_job_counters = {} # gen -> int
        
        # Global failure history
        self.global_error_history = []

        # Initialize LLM Client
        from shinka.llm.llm import LLMClient
        self.llm = LLMClient(
            model_names=evo_config.llm_models,
            **evo_config.llm_kwargs,
            verbose=verbose,
        )

        # Initialize SIBS Agents
        self.fo_planner = FirstOrderPlanner(self.llm)
        self.so_initializer = SecondOrderInitializer(self.llm)
        self.design_mutator = DesignMutator(self.llm)
        self.implementation_agent = ImplementationAgent(self.llm)
        self.reflection_writer = ReflectionWriter(self.llm)

        # Initialize Database & Scheduler
        db_path = Path(f"{self.results_dir}/{db_config.db_path}")
        db_config.db_path = str(db_path)
        self.db = ProgramDatabase(config=db_config)
        self.scheduler = JobScheduler(
            job_type=evo_config.job_type,
            config=job_config,
            verbose=verbose,
        )

        self.console = Console()
        self.lang_ext = "py" # SIBS implementation assumes Python

        self.running_jobs: List[RunningJob] = []
        self.best_program_id: Optional[str] = None
        self.next_generation_to_submit = 0
        self.completed_generations = 0
        
        # Load evaluate.py content for context
        self.eval_script_content = ""
        if hasattr(job_config, "eval_program_path") and job_config.eval_program_path:
            try:
                self.eval_script_content = Path(job_config.eval_program_path).read_text(encoding="utf-8")
                logger.info("Loaded evaluate.py content for context.")
            except Exception as e:
                logger.warning(f"Failed to read evaluate.py: {e}")
        
        # Store plans
        self.island_plans = {}

    def run(self):
        """Main Evolutionary Loop (SIBS)"""
        max_jobs = self.evo_config.max_parallel_jobs
        target_gens = self.evo_config.num_generations
        
        logger.info(f"Starting SIBS Evolution: {target_gens} generations")

        # Phase 1: Initialize Islands (Plans)
        if not self.db.island_manager.get_island_plan(0): # Check if already planned
            self._initialize_islands()

        # Phase 2: Generation 0 (Initial Population from Plans)
        if self.completed_generations == 0 and target_gens > 0:
            self._spawn_initial_population()
            self.completed_generations = 1
            self.next_generation_to_submit = 1
        
        # Parallel Execution Loop
        while self.completed_generations < target_gens or len(self.running_jobs) > 0:
            # 1. Process Finished Jobs
            completed_jobs = self._check_completed_jobs()
            for job in completed_jobs:
                self._process_completed_job(job)
            
            # 2. Update Progress
            self._update_completed_generations()
            
            if self.completed_generations >= target_gens and not self.running_jobs:
                break
                
            # 3. Submit New Jobs
            if len(self.running_jobs) < max_jobs and self.next_generation_to_submit < target_gens:
                self._submit_new_job()
            
            time.sleep(2)

        self.db.print_summary()
        logger.info("SIBS Evolution Completed.")

    def _initialize_islands(self):
        """Generate First Order Bias Plans for each island."""
        logger.info("Phase 1: Generating Island Plans...")
        
        num_islands = getattr(self.db.config, "num_islands", 1)
        if num_islands < 1: num_islands = 1
        
        task_desc = self.evo_config.task_sys_msg or "Solve the task."
        # Append constraints to task desc
        task_desc += f"\nConstraints: Max Params={self.evo_config.max_params}, Train Steps={self.evo_config.train_steps}"
        
        dataset_type = "default"

        for island_idx in range(num_islands):
            logger.info(f"Planning Island {island_idx}...")
            fo_plan = self.fo_planner.plan(island_idx, task_desc, dataset_type)
            self.db.island_manager.set_island_plan(island_idx, fo_plan.model_dump_json())
            self.island_plans[island_idx] = fo_plan
            
            # Save plan to disk for inspection
            plan_dir = f"{self.results_dir}/island_{island_idx}"
            Path(plan_dir).mkdir(parents=True, exist_ok=True)
            with open(f"{plan_dir}/plan.json", "w") as f:
                f.write(fo_plan.model_dump_json(indent=2))

    def _spawn_initial_population(self):
        """SIBS Gen 0: Init -> Implement (for each island) multiple times if needed."""
        logger.info("Phase 2: Spawning Initial Population (Gen 0)...")
        
        current_gen = 0
        gen_dir = f"{self.results_dir}/{FOLDER_PREFIX}_{current_gen}"
        Path(gen_dir).mkdir(parents=True, exist_ok=True)
        
        num_islands = getattr(self.db.config, "num_islands", 1)
        if num_islands < 1: num_islands = 1
        
        task_desc = self.evo_config.task_sys_msg or "Solve the task."
        task_desc += f"\nConstraints: Max Params={self.evo_config.max_params}, Train Steps={self.evo_config.train_steps}"

        # In Gen 0, we might want multiple individuals per island if population size > num_islands
        # For now, let's assume 1 per island as per original code logic, or maybe more?
        # User said "islands are just grouped... used to start". 
        # Typically we want an initial population. Let's spawn 1 per island for now to match strict resource limits
        # unless configured otherwise.
        
        for island_idx in range(num_islands):
            # Load plan if not in memory
            if island_idx not in self.island_plans:
                p_yaml = self.db.island_manager.get_island_plan(island_idx)
                # Need to parse yaml back to obj... simpler to just assume it's there or user flow expects it.
                # If we restarted, we might need to reload. For now assume fresh run.
                pass

            # Create individual
            # 1. Init Second Order Genome from Plan
            # We need the plan object.
            # Hack: if we don't have the object, we re-parse or skip. 
            # Ideally agents return objects.
            fo_plan = self.island_plans.get(island_idx)
            if not fo_plan: 
                logger.warning(f"No plan for island {island_idx}, skipping.")
                continue

            so_genome = self.so_initializer.initialize(fo_plan, task_desc)
            so_genome.island_id = island_idx
            so_genome.generation = 0
            so_genome.genome_id = str(uuid.uuid4())
            
            # 2. Implement (Diff from Template)
            # Create a separate folder for this job
            if current_gen not in self.generation_job_counters:
                self.generation_job_counters[current_gen] = 0
            job_idx = self.generation_job_counters[current_gen]
            self.generation_job_counters[current_gen] += 1
            
            job_dir = f"{gen_dir}/job_{job_idx}"
            Path(job_dir).mkdir(parents=True, exist_ok=True)
            
            # Save genome
            with open(f"{job_dir}/genome.json", "w") as f:
                f.write(so_genome.model_dump_json(indent=2))
            
            code = self.implementation_agent.implement(
                genome=so_genome,
                parent_code=MINIMAL_TEMPLATE_CODE,
                component="All",
                artifact_dir=job_dir,
                eval_script_content=self.eval_script_content
            )
            
            # Save Main
            fname = f"{job_dir}/main.py"
            with open(fname, "w", encoding="utf-8") as f:
                f.write(code)
            
            logger.info(f"Submitting Gen 0 seed for Island {island_idx}...")
            # Results go to job_dir (evaluate.py writes there if we pass it as results_dir arg?
            # Wait, `runner` call to scheduler passes `results_dir`.
            # If scheduler runs evaluate.py, does it pass this dir?
            # LocalJobConfig usually passes --results_dir {results_dir}
            results, rtime = self.scheduler.run(fname, job_dir)
            
            self._save_result_to_db(
                results, rtime, code, so_genome, None, 0, island_idx=island_idx
            )

    def _submit_new_job(self):
        """SIBS: Sample -> Mutate/Crossover -> Implement -> Submit"""
        current_gen = self.next_generation_to_submit
        logger.info(f"Preparing Gen {current_gen}...")
        
        gen_dir = f"{self.results_dir}/{FOLDER_PREFIX}_{current_gen}"
        Path(gen_dir).mkdir(parents=True, exist_ok=True)
        
        for resample in range(self.evo_config.max_patch_resamples):
            try:
                # 1. Sample Parent
                parent_prog, archive_progs, top_k_progs = self.db.sample(
                    target_generation=current_gen,
                    resample_attempt=resample + 1
                )
                
                # Reconstruct Genome objects
                parent_genome = SecondOrderGenome.model_validate_json(parent_prog.genome)
                inspirations = [SecondOrderGenome.model_validate_json(p.genome) for p in archive_progs + top_k_progs if p.genome]
                
                # Context for mutation: The Island Plan (First Order)
                # We need to know the island of the parent.
                island_idx = parent_prog.island_idx if parent_prog.island_idx is not None else 0
                
                fo_plan_str = None
                if island_idx in self.island_plans:
                    fo_plan_str = self.island_plans[island_idx].model_dump_json(indent=2)
                else:
                    # Try to fetch from DB manager
                    fo_yaml = self.db.island_manager.get_island_plan(island_idx)
                    # We might need to convert yaml to json or just pass yaml. Agents usually handle both but prompt says "First Order Bias Plan". 
                    # Assuming yaml string is fine.
                    if fo_yaml:
                        fo_plan_str = fo_yaml

                
                # Setup Job Directory
                if current_gen not in self.generation_job_counters:
                    self.generation_job_counters[current_gen] = 0
                job_idx = self.generation_job_counters[current_gen]
                self.generation_job_counters[current_gen] += 1
                
                job_dir = f"{gen_dir}/job_{job_idx}"
                Path(job_dir).mkdir(parents=True, exist_ok=True)

                # 2. Mutate (Diff) or Crossover (Swap)
                target_comp = "All" # Default
                
                # Mutation Probs
                if np.random.random() < 0.8: # TODO: Make configurable
                    # Mutation
                    weights = self.evo_config.mutation_weights
                    comp = np.random.choice(list(weights.keys()), p=list(weights.values()))
                    logger.info(f"Mutating {comp}...")
                    
                    new_genome = self.design_mutator.mutate(
                        parent_genome, 
                        comp, 
                        archive_progs,
                        top_k_progs,
                        first_order_plan=fo_plan_str,
                        artifact_dir=job_dir
                    )
                    patch_type = f"mutation_{comp}"
                    target_comp = comp
                else:
                    # Crossover
                    if not inspirations:
                        continue 
                    partner = np.random.choice(inspirations)
                    comp = np.random.choice(["Alpha", "Omega", "Phi"])
                    logger.info(f"Crossover {comp}...")
                    
                    # Manual swap
                    new_genome = SecondOrderGenome.model_validate_json(parent_genome.model_dump_json())
                    partner_comp = getattr(partner.learner, comp)
                    setattr(new_genome.learner, comp, partner_comp)
                    patch_type = f"crossover_{comp}"
                    target_comp = comp # Technically we swapped this component, so we implement it (or All?)
                    # If we swap, the code for that component changes.
                    # But if we swap "Alpha", we need to rewrite Alpha region.
                
                new_genome.generation = current_gen
                new_genome.parent_id = parent_genome.genome_id
                
                # 3. Implement (Code Diff)
                parent_code = parent_prog.code
                

                
                # Save Genome
                with open(f"{job_dir}/genome.json", "w") as f:
                    f.write(new_genome.model_dump_json(indent=2))
                
                code = self.implementation_agent.implement(
                    genome=new_genome, 
                    parent_code=parent_code, 
                    component=target_comp,
                    artifact_dir=job_dir,
                    eval_script_content=self.eval_script_content,
                    historical_errors=self.global_error_history[-self.evo_config.num_previous_gen_errors:]
                )
                
                # 4. Submit
                exec_fname = f"{job_dir}/main.py"
                with open(exec_fname, "w", encoding="utf-8") as f:
                    f.write(code)
                
                job_id = self.scheduler.submit_async(exec_fname, job_dir)
                
                self.running_jobs.append(RunningJob(
                    job_id=job_id,
                    exec_fname=exec_fname,
                    results_dir=job_dir, # Use job_dir as results_dir
                    start_time=time.time(),
                    generation=current_gen,
                    parent_id=parent_prog.id,
                    archive_insp_ids=[p.id for p in archive_progs],
                    top_k_insp_ids=[p.id for p in top_k_progs],
                    meta_patch_data={"patch_type": patch_type},
                    retry_count=0,
                    genome_yaml=new_genome.model_dump_json(),
                    job_dir=job_dir
                ))
                
                self.next_generation_to_submit += 1
                return 

            except Exception as e:
                logger.warning(f"Resample failed: {e}")
                # traceback.print_exc()
                continue

    def _check_completed_jobs(self) -> List[RunningJob]:
        completed = []
        still_running = []
        for job in self.running_jobs:
            if not self.scheduler.check_job_status(job):
                completed.append(job)
            else:
                still_running.append(job)
        self.running_jobs = still_running
        return completed

    def _process_completed_job(self, job: RunningJob):
        """Handle job completion, including Reflection and Repair."""
        end_time = time.time()
        rtime = end_time - job.start_time
        # results_dir for get_job_results is job_dir
        results = self.scheduler.get_job_results(job.job_id, job.results_dir)
        
        try:
            code = Path(job.exec_fname).read_text(encoding="utf-8")
        except:
            code = ""
        
        correct = results.get("correct", {}).get("correct", False) if results else False
        stderr_log = results.get("stderr_log", "") if results else "No results found."

        if not correct:
             # Add to global history
             self.global_error_history.append(f"Job {job.job_id} Error:\n{stderr_log}")

        # REPAIR LOGIC
        if not correct and job.retry_count < self.evo_config.max_repair_attempts:
            logger.info(f"Job failed (Attempt {job.retry_count}). Attempting Repair...")
            try:
                genome = SecondOrderGenome.model_validate_json(job.genome_yaml)
                
                # Use passed artifact_dir?
                repaired_code = self.implementation_agent.implement(
                    genome=genome, 
                    parent_code=code, 
                    previous_errors=stderr_log,
                    component="All", # Repair usually needs global context or we assume component?
                    # If structure is broken, maybe All. If just invalid syntax in region, component?
                    # Safest is All for repair to fix imports etc.
                    artifact_dir=job.job_dir,
                    eval_script_content=self.eval_script_content,
                    historical_errors=self.global_error_history[-self.evo_config.num_previous_gen_errors:]
                )
                
                # Overwrite main.py? User said "replace also the retry_1 once you write retry_2".
                # Actually, if we overwrite `main.py`, we lose the history of failed attempts if we don't save them.
                # But user said "Remove the id... ensure to replace also the retry_1".
                # Implies keeping one main file.
                # Create NEW job directory for repair attempt
                # This ensures history is preserved: job_0 (fail) -> job_1 (repair 1) -> ...
                current_gen = job.generation
                if current_gen not in self.generation_job_counters:
                    self.generation_job_counters[current_gen] = 0
                job_idx = self.generation_job_counters[current_gen]
                self.generation_job_counters[current_gen] += 1
                
                # We need to know where the gen folder is. job.job_dir is a path.
                # Usually job.job_dir is ".../gen_X/job_Y". So parent is gen dir.
                gen_dir = Path(job.job_dir).parent
                new_job_dir = str(gen_dir / f"job_{job_idx}")
                Path(new_job_dir).mkdir(parents=True, exist_ok=True)

                new_fname = f"{new_job_dir}/main.py"
                with open(new_fname, "w", encoding="utf-8") as f:
                    f.write(repaired_code)
                    
                new_job_id = self.scheduler.submit_async(new_fname, new_job_dir)
                
                self.running_jobs.append(RunningJob(
                    job_id=new_job_id,
                    exec_fname=new_fname,
                    results_dir=new_job_dir,
                    start_time=time.time(),
                    generation=job.generation,
                    parent_id=job.parent_id,
                    archive_insp_ids=job.archive_insp_ids,
                    top_k_insp_ids=job.top_k_insp_ids,
                    meta_patch_data=job.meta_patch_data,
                    retry_count=job.retry_count + 1,
                    genome_yaml=job.genome_yaml,
                    job_dir=new_job_dir
                ))
                return 
            except Exception as e:
                logger.error(f"Repair failed: {e}")
                # Fall through to save as failed

        # Reflection Logic
        # We attempt reflection regardless of success, to capture why it failed or succeeded.
        genome = SecondOrderGenome.model_validate_json(job.genome_yaml)
        
        # Populate fitness in genome if available
        if results and results.get("metrics"):
            genome.fitness = results.get("metrics", {})
            
        # Attempt Reflection
        if job.parent_id:
            try:
                parent_prog = self.db.get(job.parent_id)
                if parent_prog and parent_prog.genome:
                    parent_genome = SecondOrderGenome.model_validate_json(parent_prog.genome)
                    parent_metrics = parent_prog.public_metrics or {}
                    
                    # If results check failed or metrics missing, use empty dicts
                    child_metrics = {}
                    if results and results.get("metrics"):
                        child_metrics = results.get("metrics", {}).get("public", {})
                    
                    # Add stderr to child metrics if failed, so reflection sees the error!
                    if not correct:
                         child_metrics["_error_log"] = stderr_log[:1000] # Truncate for prompt context
                    
                    genome = self.reflection_writer.reflect(
                        parent_genome=parent_genome,
                        parent_metrics=parent_metrics,
                        child_genome=genome,
                        child_metrics=child_metrics,
                        artifact_dir=job.job_dir
                    )
                    logger.info("Reflection completed successfully")
            except Exception as e:
                logger.error(f"Reflection failed: {e}. Continuing without reflection.")
        else:
            logger.info("Skipping reflection for initial genome (no parent)")
        
        self._save_result_to_db(results, rtime, code, genome, job.parent_id, job.generation, job_dir=job.job_dir)


    def _save_result_to_db(self, results, rtime, code, genome, parent_id, generation, island_idx=None, job_dir=None):
        metrics_val = results.get("metrics", {}) if results else {}
        correct_val = results.get("correct", {}).get("correct", False) if results else False
        
        program_id = str(uuid.uuid4())
        db_program = Program(
            id=program_id,
            code=code,
            language="python",
            parent_id=parent_id,
            generation=generation,
            archive_inspiration_ids=[],
            top_k_inspiration_ids=[],
            code_diff=None,
            embedding=[],
            correct=correct_val,
            combined_score=metrics_val.get("combined_score", 0.0),
            public_metrics=metrics_val.get("public", {}),
            private_metrics=metrics_val.get("private", {}),
            text_feedback=metrics_val.get("text_feedback", ""),
            metadata={
                "compute_time": rtime,
                "genome_id": genome.genome_id if genome else None
            },
            island_idx=island_idx,
            genome=genome.model_dump_json() if genome else None
        )
        self.db.add(db_program, verbose=True)
        self.db.save()
        self._update_best_solution(candidate_id=program_id, candidate_dir=job_dir)

    def _update_best_solution(self, candidate_id: Optional[str] = None, candidate_dir: Optional[str] = None):
         """Check if the candidate is the new global best and save artifacts if so."""
         best_prog = self.db.get_best_program()
         
         if not best_prog:
             return

         # If we have a new best program (or first one)
         if best_prog.id != self.best_program_id:
             self.best_program_id = best_prog.id
             logger.info(f"Global Best Program Updated: {best_prog.id} (Score: {best_prog.combined_score})")
             
             # If the new best is the candidate we just added, copy its files
             if candidate_id and candidate_dir and best_prog.id == candidate_id:
                 try:
                     best_dir = Path(self.results_dir) / "best"
                     # Clean previous best
                     if best_dir.exists():
                         shutil.rmtree(best_dir)
                     best_dir.mkdir(parents=True, exist_ok=True)
                     
                     # Copy content of job dir to best dir
                     shutil.copytree(candidate_dir, best_dir, dirs_exist_ok=True)
                     logger.info(f"Copied artifacts of best program to {best_dir}")
                     
                     # Add a metadata file about the best program
                     with open(best_dir / "best_program_info.json", "w") as f:
                         info = {
                             "id": best_prog.id,
                             "score": best_prog.combined_score,
                             "generation": best_prog.generation,
                             "metrics": best_prog.public_metrics
                         }
                         json.dump(info, f, indent=2)
                         
                 except Exception as e:
                     logger.error(f"Failed to copy best program artifacts: {e}")

    def _update_completed_generations(self):
        last_gen = self.db.last_iteration
        if last_gen == -1:
            self.completed_generations = 0
            return
        
        # Simple continuity check
        completed = 0
        for i in range(last_gen + 1):
            if self.db.get_programs_by_generation(i):
                completed = i + 1
            else:
                break
        self.completed_generations = completed


