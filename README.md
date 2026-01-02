<h1 align="center">
  <a href="shinka/favicon.png?raw=true"><img src="shinka/favicon.png?raw=true" width="180" /></a><br>
  <b><code>ShinkaBias</code>: Deep Learning Architecture Evolution (SIBS)</b><br>
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-%3E%3D3.10-blue" />
  <a href="https://github.com/SakanaAI/ShinkaEvolve/blob/master/LICENSE.md"><img src="https://img.shields.io/badge/license-Apache2.0-blue.svg" /></a>
  <a href="https://github.com/astral-sh/ruff"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" /></a>
  <a href="http://arxiv.org/abs/2509.19349"><img src="http://img.shields.io/badge/paper-arxiv.2509.19349-B31B1B.svg" /></a>
  <a href="https://colab.research.google.com/github/SakanaAI/ShinkaEvolve/blob/main/examples/shinka_tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" /></a>
</p>

`ShinkaEvolve` in this repository is a modified version of the original ShinkaEvolve framework, adapted specifically for **deep learning architecture evolution**. The core idea is to search over **inductive biases** rather than arbitrary code edits, and to keep architectural, objective, and optimizer decisions explicitly separated.

This fork implements a structured, agent-driven pipeline (see `shinka/sibs`) that:
- Plans **first-order bias requirements** from a task description.
- Translates them into a **SecondOrderGenome** with Alpha/Phi/Omega components.
- Mutates only one component at a time, using **inspirations from archive and top-k programs**.
- Applies edits only to the relevant code region (Alpha, Phi, or Omega).
- Evaluates and reflects on results to guide future mutations.

## What Is Different From Upstream ShinkaEvolve

- **Bias-first planning**: a FirstOrderBiasPlan is generated before any code edits.
- **Structured genomes**: biases are encoded in a SecondOrderGenome with explicit Alpha/Phi/Omega sections.
- **Region-restricted edits**: only the relevant code region is editable (`# REGION_ALPHA_START`, `# REGION_PHI_START`, `# REGION_OMEGA_START`).
- **Mutation context is explicit**: the mutator receives the parent genome plus archive and top-k inspirations, each with metadata and genomes.
- **LLM interaction logs**: each job directory contains `mutation_log.md` and `implementation_log.md` with full prompts and responses.

## SIBS Pipeline Overview

1. **FirstOrderPlanner**: produce `FirstOrderBiasPlan` (alpha/phi/omega requirements).
2. **SecondOrderInitializer**: convert that plan into a `BaseSecondOrderGenome`.
3. **DesignMutator**: mutate a single component using parent + inspirations.
4. **ImplementationAgent**: apply SEARCH/REPLACE patches to the relevant code region.
5. **Evaluation**: run `evaluate.py` to train and score the candidate.
6. **ReflectionWriter**: update genome reflections from metrics.

## Model Contract (Region-Based Editing)

The evolved program must define an `EvolvedModel` with explicit regions. The implementation agent edits only the target region.

```python
# REGION_ALPHA_START
import torch
import torch.nn as nn

class EvolvedModel(nn.Module):
    def __init__(self):
        super().__init__()
        # architecture here

    def forward(self, x):
        # forward pass here
        pass
# REGION_ALPHA_END

# REGION_PHI_START
    def compute_loss(self, batch, outputs):
        # loss here
        return torch.tensor(0.0, requires_grad=True)

    def compute_metrics(self, batch, outputs):
        # metrics only (no loss here)
        return {"accuracy": 0.0}
# REGION_PHI_END

# REGION_OMEGA_START
    def compute_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
# REGION_OMEGA_END
```

## Artifacts and Logs

Each job directory (for example, `results/.../gen_3/job_7/`) contains:
- `genome.json`: the generated or mutated SecondOrderGenome.
- `main.py`: the evolved code submitted for evaluation.
- `mutation_log.md`: system/user prompt + model response for the mutation step.
- `implementation_log.md`: system/user prompt + model response for implementation.

## Installation & Quick Start

```bash
# Clone the repository
git clone https://github.com/SakanaAI/ShinkaEvolve
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install
cd ShinkaEvolve
uv venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .

# Run the maze architecture evolution example
python examples/maze_model_search/run_evo.py
```

## Examples

| Example | Description | Entry Point |
|---------|-------------|-------------|
| 🧭 [Maze Model Search](examples/maze_model_search) | Evolve RL-friendly architectures for partial observability | `python examples/maze_model_search/run_evo.py` |
| ⭕ [Circle Packing](examples/circle_packing) | Optimization baseline from upstream ShinkaEvolve | `python examples/circle_packing/run_evo.py` |
| 🎯 [ALE-Bench](examples/ale_bench) | Code optimization tasks | `python examples/ale_bench/run_evo.py` |
| ✨ [Novelty Generator](examples/novelty_generator) | Creative outputs (ASCII art, etc.) | `shinka_launch variant=novelty_generator_example` |

## Configuration Notes

Key configuration surfaces for architecture evolution:
- `EvolutionConfig`: `task_sys_msg`, `num_generations`, `max_parallel_jobs`, `max_patch_resamples`, `max_repair_attempts`, `mutation_weights`, `max_params`, `training_epochs`, `llm_models`, `llm_kwargs`, `results_dir`.
- `DatabaseConfig`: `num_islands`, `archive_size`, `num_archive_inspirations`, `num_top_k_inspirations`, `parent_selection_strategy`, `exploitation_alpha`, `exploitation_ratio`.

Parent selection strategies (`DatabaseConfig.parent_selection_strategy`):
- `power_law`: rank-based sampling from archive or correct programs (default).
- `weighted`: combines performance and novelty via a weighted probability.
- `beam_search`: stick to a parent for multiple children, then switch to the best.
- `best_of_n`: always return the earliest correct seed (fallback to any correct).

## Documentation

- `docs/getting_started.md` for setup.
- `docs/configuration.md` for config options and launcher usage.
- `docs/support_local_llm.md` for local LLM setup.

## Citation

If you use `ShinkaEvolve` in your research, please cite the original paper:

```
@article{lange2025shinka,
  title={ShinkaEvolve: Towards Open-Ended And Sample-Efficient Program Evolution},
  author={Lange, Robert Tjarko and Imajuku, Yuki and Cetin, Edoardo},
  journal={arXiv preprint arXiv:2509.19349},
  year={2025}
}
```
