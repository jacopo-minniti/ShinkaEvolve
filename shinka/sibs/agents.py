import logging
import yaml
import re
from typing import List, Optional, Tuple, Any, Dict
from shinka.llm.llm import LLMClient
from shinka.sibs.schema import FirstOrderBiasPlan, SecondOrderGenome
from shinka.database import Program

logger = logging.getLogger(__name__)

FIRST_ORDER_PLANNER_SYS_PROMPT = """
You are the FirstOrderBiasPlanner.

Role
You operate at the level of *first-order inductive biases*. Your responsibility is to interpret a high-level task description and produce a structured FirstOrderBiasPlan that specifies what kinds of biases are desirable for solving the task, without committing to any concrete implementation details.

Conceptual scope
First-order biases describe *what the model should be biased toward*, not *how that bias is implemented*. They capture high-level properties of the task and data, such as:
- What information must be preserved or emphasized
- What invariances or equivariances are likely useful
- What forms of structure (temporal, spatial, hierarchical, relational, etc.) are inherent to the task
- What constraints arise from the data modality or observability assumptions

You must not propose architectures, hyperparameters, or training tricks. Those belong to second-order decisions.

Inputs
You will receive:
- A natural-language task description provided by the user
- Metadata about the task type and dataset (e.g. modality, supervision type, observability, sequence vs. static, etc.)

Bias dimensions
Your plan must reason explicitly about the following four components:

1. Alpha (Task-level inductive biases)
   These describe the abstract computational demands of the task.
   Examples: need for memory, long-range dependency handling, compositionality, causal reasoning, partial observability handling, etc.

2. Beta (Data-level inductive biases)
   These describe properties of the dataset and environment.
   Examples: noise structure, sparsity, class imbalance, symmetries, stationarity or non-stationarity, sample efficiency constraints.

3. Gamma (Objective-level inductive biases)
   These describe what the learning objective should implicitly encourage.
   Examples: smoothness vs. sharp decision boundaries, robustness to perturbations, exploration vs. exploitation, calibration, interpretability.

4. Delta (Constraint and evaluation biases)
   These describe constraints imposed by evaluation or deployment.
   Examples: latency limits, memory limits, generalization regime, out-of-distribution expectations, metric-driven behavior.

For each component, you must:
- Clearly state the bias objective
- Justify why it is relevant given the task and data
- Avoid any reference to specific layers, losses, optimizers, or code

Output format
Output YAML only.
The YAML must contain:
- A short task summary
- One section per bias component (Alpha, Beta, Gamma, Delta)
- Each section must include:
  - bias_name
  - bias_objective (what this bias is trying to achieve)
  - rationale (why this bias matters for the task)

Do not include second-order or implementation details.
Do not include free-form commentary outside the YAML.
"""


SECOND_ORDER_INITIALIZER_SYS_PROMPT = """
You are the SecondOrderInitializer.

Role
You translate a FirstOrderBiasPlan into an initial SecondOrderGenome. This is the step where abstract bias objectives are mapped into *concrete but still modular design choices*.

Conceptual scope
Second-order biases define *how* first-order biases are operationalized. They include:
- Architectural motifs
- Loss formulations and auxiliary objectives
- Information flow constraints
- Parameter sharing strategies
- Training-time mechanisms that realize the desired inductive bias

You must remain modular and explicit, but not over-engineered. The goal is to create a reasonable initial genome, not an optimized solution.

Inputs
You will receive:
- A FirstOrderBiasPlan expressed in YAML

Genome structure
The SecondOrderGenome must contain the following components:

1. Alpha (Architectural realization)
   How task-level biases are realized structurally.
   Examples: recurrence vs. feedforward, attention mechanisms, memory modules, locality constraints.

2. Omega (Objective and optimization realization)
   How objective-level biases are enforced.
   Examples: primary loss, auxiliary losses, regularization terms, uncertainty modeling.

3. Phi (Information and representation flow)
   How data-level and constraint-level biases are handled through representations.
   Examples: encoding strategies, bottlenecks, normalization choices, state aggregation.

For each component, you must:
- Explicitly map first-order bias objectives to concrete mechanisms
- Explain the intended effect of each design choice
- Keep choices simple and interpretable
- Avoid unnecessary hyperparameter tuning or exotic tricks

Output format
Output YAML only.
The YAML must:
- Preserve traceability to the original first-order biases
- Clearly separate Alpha, Omega, and Phi sections
- Include short rationales for each design choice

Do not modify or reinterpret the original bias objectives.
Do not include code.
"""


DESIGN_MUTATOR_SYS_PROMPT = """
You are the Design Mutator.

Role
You perform targeted mutations on an existing SecondOrderGenome. Your task is to explore the design space while preserving coherence with the original bias objectives.

Conceptual scope
A mutation is a *local, intentional change* to one component of the genome:
- Alpha: architectural changes
- Omega: objective or optimization changes
- Phi: representation or information-flow changes

Mutations should be:
- Minimal but meaningful
- Aligned with the stated bias objectives
- Easy to attribute during evaluation and reflection

Inputs
You will receive:
- A SecondOrderGenome in YAML
- An instruction specifying which component to mutate (Alpha, Omega, or Phi)
- Optionally, a motivation for the mutation (e.g. performance issue, instability, underfitting)

Rules
- Modify only the requested component
- Do not silently change other sections
- Preserve YAML validity
- Ensure the new design still implements the original first-order bias intent

Edit format
You must use SEARCH/REPLACE blocks exactly as specified.

Format:
<<<<<<< SEARCH
# Exact YAML content to replace
=======
# New YAML content
>>>>>>> REPLACE

Do not include explanations outside the YAML edits.
Do not output anything other than SEARCH/REPLACE blocks.
"""


IMPLEMENTATION_AGENT_SYS_PROMPT = """
You are the Implementation Agent.

Role
You translate a SecondOrderGenome into concrete PyTorch code changes. Your responsibility is to ensure the implementation faithfully reflects the genome specification.

Conceptual scope
You operate strictly at the code level:
- Modify model architecture
- Adjust forward passes
- Implement losses or auxiliary objectives
- Update data flow as required

You must not reinterpret the genome. If something is unclear, implement the most literal and minimal interpretation consistent with the specification.

Inputs
You will receive:
- Existing PyTorch code
- A SecondOrderGenome describing the desired design
- Optionally, error logs or failing behaviors from previous runs

Rules
- All edits must be done using SEARCH/REPLACE blocks
- The SEARCH block must match the original code exactly
- The REPLACE block must contain valid, runnable PyTorch code
- Preserve style and indentation consistency
- If previous errors are provided, prioritize fixing them

Edit format
Use the following structure exactly:

<DIFF>
<<<<<<< SEARCH
# Original code to find and replace (must match exactly including indentation)
=======
# New replacement code
>>>>>>> REPLACE
</DIFF>

Do not include commentary outside the DIFF blocks.
Do not output partial code fragments.
"""


REFLECTION_WRITER_SYS_PROMPT = """
You are the Reflection Writer.

Role
You analyze evaluation results and update the SecondOrderGenome with reflective annotations explaining how and why specific biases succeeded or failed.

Conceptual scope
Reflections operate at the bias level, not the code level. They should:
- Connect observed metrics to bias design choices
- Identify likely causal relationships
- Highlight trade-offs revealed by the evaluation

Inputs
You will receive:
- Evaluation metrics and qualitative observations
- The corresponding SecondOrderGenome

Reflection guidelines
For each affected bias:
- State what the bias was intended to achieve
- Describe how the observed results align or misalign with that intent
- Avoid overconfidence or absolute claims
- Distinguish between evidence and speculation

You may suggest future mutation directions, but only as reflections, not changes.

Output format
Output the updated SecondOrderGenome YAML only.
Add a dedicated reflection field to relevant components.
Do not remove existing genome content.
Do not include external commentary.
"""

def _clean_yaml_content(content: str) -> str:
    """
    Clean the content to ensure it is valid YAML.
    1. Remove <think> tags.
    2. Replace non-breaking spaces and other common invisible characters.
    3. Ensure ASCII compatible (optional, but safer for YAML scanners).
    """
    # Remove <think> blocks
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    
    # Replace non-breaking spaces
    content = content.replace('\u00A0', ' ')
    
    # Normalize fancy quotes (optional but good for copy-paste issues)
    content = content.replace('\u201c', '"').replace('\u201d', '"')
    content = content.replace('\u2018', "'").replace('\u2019', "'")

    return content.strip()

def _extract_yaml_heuristic(content: str, known_keys: List[str]) -> str:
    """
    Heuristically extract YAML content by finding the first occurrence of a known key.
    This handles cases where the LLM chats before outputting YAML without code blocks.
    """
    lines = content.split('\n')
    start_idx = -1
    
    # Try to find the start of the YAML
    for i, line in enumerate(lines):
        line = line.strip()
        # Check if line starts with a known key
        for key in known_keys:
            if line.startswith(f"{key}:") or line.startswith(f"- {key}:"):
                start_idx = i
                break
        if start_idx != -1:
            break
            
    if start_idx != -1:
        # Check for potential end (e.g. if chat resumes after YAML)
        # This is harder for YAML as it relies on indentation. 
        # For now, we assume the rest of the text is the YAML or at least
        # the YAML parser might be robust enough to ignore trailing chat if indent doesn't match?
        # Actually trailing chat often causes errors.
        # But cutting from start is better than nothing.
        return "\n".join(lines[start_idx:])
    
    return content



class FirstOrderPlanner:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def plan(self, island_id: int, task_description: str, dataset_type: str) -> FirstOrderBiasPlan:
        user_msg = f"Task Description: {task_description}\nDataset Type: {dataset_type}\nIsland ID: {island_id}\n\nGenerate a FirstOrderBiasPlan in YAML format."
        response = self.llm.query(msg=user_msg, system_msg=FIRST_ORDER_PLANNER_SYS_PROMPT)
        if response and response.content:
            logger.debug(f"FirstOrderPlanner Raw Response:\n{response.content}")
            
            # Clean content first
            content = _clean_yaml_content(response.content)
            logger.debug(f"FirstOrderPlanner Cleaned Content:\n{content}")
            
            from shinka.llm.llm import extract_between
            # Use extract_between with fallback to code blocks for YAML
            # We first try explicit code blocks, then generic
            # The prompt asks for YAML, so we look for ```yaml ... ``` first
            try:
                # First try strict YAML block
                data = extract_between(content, start="```yaml", end="```", return_dict=True, is_yaml=True)
                if data == "none" or data is None:
                     # Fallback to just ```...``` if no lang specified
                     data = extract_between(content, start="```", end="```", return_dict=True, is_yaml=True)
                
                if data != "none" and data is not None:
                    logger.debug("Successfully parsed FirstOrderBiasPlan from code block.")
                    return FirstOrderBiasPlan.from_dict(data) 
                
                # If strict extraction fails, try heuristic cleanup
                clean_content = content.replace("```yaml", "").replace("```", "").strip()
                # Specific keys for FirstOrderBiasPlan
                known_keys = ["first_order_version", "island_id", "task_summary", "alpha_requirements"]
                heuristic_content = _extract_yaml_heuristic(clean_content, known_keys)
                
                logger.debug(f"Attempting heuristic parse on:\n{heuristic_content}")
                return FirstOrderBiasPlan.from_yaml(heuristic_content)
            except Exception as e:
                logger.warning(f"Failed to parse FirstOrderBiasPlan: {e}.")
                logger.debug(f"Problematic Content (Cleaned):\n{content}")
                # Try raw/heuristic
                clean_content = content.replace("```yaml", "").replace("```", "").strip()
                known_keys = ["first_order_version", "island_id", "task_summary", "alpha_requirements"]
                heuristic_content = _extract_yaml_heuristic(clean_content, known_keys)
                return FirstOrderBiasPlan.from_yaml(heuristic_content)


        raise ValueError("Failed to generate FirstOrderBiasPlan")

class SecondOrderInitializer:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def initialize(self, first_order_plan: FirstOrderBiasPlan, task_description: str) -> SecondOrderGenome:
        user_msg = f"Task Description: {task_description}\nFirst Order Plan:\n{first_order_plan.to_yaml()}\n\nGenerate an initial SecondOrderGenome in YAML format."
        response = self.llm.query(msg=user_msg, system_msg=SECOND_ORDER_INITIALIZER_SYS_PROMPT)
        if response and response.content:
            logger.debug(f"SecondOrderInitializer Raw Response:\n{response.content}")
            content = _clean_yaml_content(response.content)
            logger.debug(f"SecondOrderInitializer Cleaned Content:\n{content}")
            
            from shinka.llm.llm import extract_between
            try:
                data = extract_between(content, start="```yaml", end="```", return_dict=True, is_yaml=True)
                if data == "none" or data is None:
                     data = extract_between(content, start="```", end="```", return_dict=True, is_yaml=True)
                
                if data != "none" and data is not None:
                    logger.debug("Successfully parsed SecondOrderGenome from code block.")
                    return SecondOrderGenome.from_dict(data)

                clean_content = content.replace("```yaml", "").replace("```", "").strip()
                known_keys = ["genome_version", "genome_id", "learner", "high_level_description"]
                heuristic_content = _extract_yaml_heuristic(clean_content, known_keys)
                logger.debug(f"Attempting heuristic parse on:\n{heuristic_content}")
                return SecondOrderGenome.from_yaml(heuristic_content)
            except Exception as e:
                logger.warning(f"Failed to parse SecondOrderGenome: {e}")
                logger.debug(f"Problematic Content (Cleaned):\n{content}")
                clean_content = content.replace("```yaml", "").replace("```", "").strip()
                known_keys = ["genome_version", "genome_id", "learner", "high_level_description"]
                heuristic_content = _extract_yaml_heuristic(clean_content, known_keys)
                return SecondOrderGenome.from_yaml(heuristic_content)
        raise ValueError("Failed to generate SecondOrderGenome")

class DesignMutator:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def mutate(self, parent_genome: SecondOrderGenome, component_to_mutate: str, 
               inspirations: List[SecondOrderGenome]) -> SecondOrderGenome:
        
        insp_str = "\n".join([f"Inspiration Genome:\n{g.to_yaml()}" for g in inspirations])
        
        user_msg = f"""
        Parent Genome:
        {parent_genome.to_yaml()}
        
        Component to Mutate: {component_to_mutate}
        
        Inspirations:
        {insp_str}
        
        Please provide a mutated version of the genome using SEARCH/REPLACE blocks.
        """
        response = self.llm.query(msg=user_msg, system_msg=DESIGN_MUTATOR_SYS_PROMPT)
        if response and response.content:
             # Apply the diff to the parent yaml string
             # Simplistic patch application for now
             patched_yaml = self._apply_diff(parent_genome.to_yaml(), response.content)
             return SecondOrderGenome.from_yaml(patched_yaml)
        raise ValueError("Failed to mutate genome")

    def _apply_diff(self, original_text: str, diff_text: str) -> str:
        # Simple regex based patch application
        # This mirrors shinka logic simplified
        pattern = re.compile(r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE", re.DOTALL)
        matches = pattern.findall(diff_text)
        
        patched_text = original_text
        for search_block, replace_block in matches:
            if search_block in patched_text:
                patched_text = patched_text.replace(search_block, replace_block, 1)
            else:
                logger.warning(f"Could not find search block: {search_block[:50]}...")
        return patched_text


class ImplementationAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def implement(self, genome: SecondOrderGenome, parent_code: str, previous_errors: Optional[str] = None) -> str:
        
        user_msg = f"""
        Target Genome Specification:
        {genome.to_yaml()}
        
        Current Code:
        ```python
        {parent_code}
        ```
        """
        if previous_errors:
            user_msg += f"\nPrevious Implementation Errors:\n{previous_errors}"
        else:
            user_msg += "\nModify the code to match the new genome."

        response = self.llm.query(msg=user_msg, system_msg=IMPLEMENTATION_AGENT_SYS_PROMPT)
        if response and response.content:
            # Apply diff
            return self._apply_diff(parent_code, response.content)
            
        raise ValueError("Failed to generate implementation code")
    
    def _apply_diff(self, original_text: str, diff_text: str) -> str:
        # Reusing similar logic or importing from shinka if strictly required
        # For now, implementing robust local patcher
        pattern = re.compile(r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE", re.DOTALL)
        matches = pattern.findall(diff_text)
        
        patched_text = original_text
        for search_block, replace_block in matches:
             # Normalize line endings just in case
             search_block = search_block.replace('\r\n', '\n')
             replace_block = replace_block.replace('\r\n', '\n')
             
             if search_block.strip() == "":
                 continue

             if search_block in patched_text:
                patched_text = patched_text.replace(search_block, replace_block, 1)
             else:
                # Try simple fuzzy match or logging
                logger.warning(f"ImplementationAgent: Could not find search block:\n{search_block}")
                
        return patched_text

class ReflectionWriter:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def reflect(self, genome: SecondOrderGenome, metrics: Dict[str, float]) -> SecondOrderGenome:
        user_msg = f"""
        Genome:
        {genome.to_yaml()}
        
        Evaluation Metrics:
        {metrics}
        
        Update the 'reflection' field for biases.
        """
        response = self.llm.query(msg=user_msg, system_msg=REFLECTION_WRITER_SYS_PROMPT)
        if response and response.content:
             logger.debug(f"ReflectionWriter Raw Response:\n{response.content}")
             content = _clean_yaml_content(response.content)
             logger.debug(f"ReflectionWriter Cleaned Content:\n{content}")

             from shinka.llm.llm import extract_between
             try:
                 data = extract_between(content, start="```yaml", end="```", return_dict=True, is_yaml=True)
                 if data == "none" or data is None:
                     data = extract_between(content, start="```", end="```", return_dict=True, is_yaml=True)
                 
                 if data != "none" and data is not None:
                     logger.debug("Successfully parsed reflected genome from code block.")
                     return SecondOrderGenome.from_dict(data)

                  clean_content = content.replace("```yaml", "").replace("```", "").strip()
                  known_keys = ["genome_version", "genome_id", "learner", "high_level_description"]
                  heuristic_content = _extract_yaml_heuristic(clean_content, known_keys)
                  logger.debug(f"Attempting heuristic parse on:\n{heuristic_content}")
                  return SecondOrderGenome.from_yaml(heuristic_content)
             except Exception as e:
                  logger.warning(f"Failed to parse reflected genome: {e}")
                  logger.debug(f"Problematic Content (Cleaned):\n{content}")
                  clean_content = content.replace("```yaml", "").replace("```", "").strip()
                  known_keys = ["genome_version", "genome_id", "learner", "high_level_description"]
                  heuristic_content = _extract_yaml_heuristic(clean_content, known_keys)
                  return SecondOrderGenome.from_yaml(heuristic_content)
        raise ValueError("Failed to reflect on genome")
