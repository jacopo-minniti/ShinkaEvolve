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
Output JSON only.
The JSON must contain:
- A short task summary
- One section per bias component (Alpha, Beta, Gamma, Delta)
- Each section must include:
  - bias_name
  - bias_objective (what this bias is trying to achieve)
  - rationale (why this bias matters for the task)

Do not include second-order or implementation details.
Do not include free-form commentary outside the JSON.
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

Target Component: {component}
You must ONLY modify the code relevant to this component.

Conceptual scope
- If Component is ALPHA (Architecture):
    - Modify `__init__` to define layers/modules.
    - Modify `forward` to change the main data flow through these layers.
    - Do NOT touch `compute_loss` or `compute_metrics`.
- If Component is OMEGA (Objective/Optimization):
    - Modify `compute_loss` to implement the loss function.
    - Modify `compute_metrics` to track relevant metrics.
    - Do NOT touch `__init__` or `forward` (unless adding simple auxiliary heads strictly needed for the loss).
- If Component is PHI (Information Flow):
    - Modify `forward` to change how data is processed/shaped (e.g. normalization, reshape).
    - Modify `__init__` only for normalization/embedding layers.

Inputs
You will receive:
- Existing PyTorch code
- A SecondOrderGenome describing the desired design
- Optionally, error logs or failing behaviors from previous runs

Rules
- All edits must be done using valid XML-style SEARCH/REPLACE blocks.
- The SEARCH block must match the original code exactly, including indentation and whitespace.
- The REPLACE block must contain valid, runnable PyTorch code.
- Preserve style and indentation consistency.
- If previous errors are provided, prioritize fixing them.
- STRICTLY adhere to the component boundaries defined above.

Edit format
Use the following structure exactly (XML tags):

<DIFF>
<<<<<<< SEARCH
# Original code to find (must match exactly)
=======
# New replacement code
>>>>>>> REPLACE
</DIFF>

Do not include commentary outside the DIFF blocks.
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
Output the updated SecondOrderGenome JSON only.
Add a dedicated reflection field to relevant components.
Do not remove existing genome content.
Do not include external commentary.
"""

def _clean_json_content(content: str) -> str:
    """
    Clean the content to ensure it is valid JSON.
    1. Remove <think> tags.
    2. Replace non-breaking spaces and other common invisible characters.
    3. Ensure ASCII compatible.
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
        user_msg = f"Task Description: {task_description}\nDataset Type: {dataset_type}\nIsland ID: {island_id}\n\nGenerate a FirstOrderBiasPlan in JSON format."
        
        response = self.llm.query(
            msg=user_msg, 
            system_msg=FIRST_ORDER_PLANNER_SYS_PROMPT,
            output_model=FirstOrderBiasPlan
        )
        
        if response and response.content:
            logger.debug(f"FirstOrderPlanner Raw Response:\n{response.content}")
            
            # Use Pydantic's validation directly on the content string (it should be valid JSON)
            content = _clean_json_content(response.content) # Reusing cleaner function name, acts as generic cleaner
            if "```json" in content:
                content = content.replace("```json", "").replace("```", "")
            elif "```" in content:
                content = content.replace("```", "")
            content = content.strip()
            
            try:
                return FirstOrderBiasPlan.model_validate_json(content)
            except Exception as e:
                logger.warning(f"Failed to parse FirstOrderBiasPlan JSON: {e}")
                
                # Attempt repair
                repaired_content = _repair_json(content)
                try:
                    logger.info("Attempting to repair JSON...")
                    return FirstOrderBiasPlan.model_validate_json(repaired_content)
                except Exception as inner_e:
                    logger.error(f"Repair failed: {inner_e}")
                    logger.debug(f"Problematic JSON Content:\n{content}")

        raise ValueError("Failed to generate FirstOrderBiasPlan")

class SecondOrderInitializer:
    def __init__(selfself, llm_client: LLMClient):
        self.llm = llm_client

    def initialize(self, first_order_plan: FirstOrderBiasPlan, task_description: str) -> SecondOrderGenome:
        user_msg = f"Task Description: {task_description}\nFirst Order Plan:\n{first_order_plan.model_dump_json(indent=2)}\n\nGenerate an initial SecondOrderGenome in JSON format."
        
        response = self.llm.query(
            msg=user_msg, 
            system_msg=SECOND_ORDER_INITIALIZER_SYS_PROMPT,
            output_model=SecondOrderGenome
        )
        
        if response and response.content:
            logger.debug(f"SecondOrderInitializer Raw Response:\n{response.content}")
            
            content = _clean_json_content(response.content)
            if "```json" in content:
                content = content.replace("```json", "").replace("```", "")
            elif "```" in content:
                content = content.replace("```", "")
            content = content.strip()
            
            try:
                return SecondOrderGenome.model_validate_json(content)
            except Exception as e:
                logger.warning(f"Failed to parse SecondOrderGenome JSON: {e}")
                
                # Attempt repair
                repaired_content = _repair_json(content)
                try:
                    logger.info(f"Attempting to repair JSON... (Appended {len(repaired_content) - len(content)} chars)")
                    return SecondOrderGenome.model_validate_json(repaired_content)
                except Exception as inner_e:
                    logger.error(f"Repair failed: {inner_e}")
                    logger.debug(f"Problematic JSON Content:\n{content}")
                     
        raise ValueError("Failed to generate SecondOrderGenome")

SECOND_ORDER_INITIALIZER_SYS_PROMPT = """
You are the SecondOrderInitializer.

Role
You translate a FirstOrderBiasPlan into an initial SecondOrderGenome. This is the step where abstract bias objectives are mapped into *concrete but still modular design choices*.

Conceptual scope
Second-order biases define *how* first-order biases are operationalized. They include:
- Architectural motifs
- Loss formulations and auxiliary objectives
- Optimizer choices and training dynamics

Genome structure
The SecondOrderGenome must contain the following components:

1. Alpha (Architectural realization)
   How task-level biases are realized structurally.
   Examples: recurrence vs. feedforward, attention mechanisms, memory modules, locality constraints.

2. Phi (Objective realization)
   How objective-level biases are enforced through the loss function.
   Examples: primary loss, auxiliary losses, regularization terms, uncertainty modeling.

3. Omega (Optimizer realization)
   How training dynamics and optimization are handled.
   Examples: optimizer choice (Adam/SGD), learning rate, schedulers (if applicable).

For each component, you must:
- Explicitly map first-order bias objectives to concrete mechanisms
- Explain the intended effect of each design choice
- Keep choices simple and interpretable
- Avoid unnecessary hyperparameter tuning or exotic tricks

Output format
Output JSON only.
The JSON must:
- Preserve traceability to the original first-order biases
- Clearly separate Alpha, Phi, and Omega sections
- Include short rationales for each design choice

Do not modify or reinterpret the original bias objectives.
Do not include code.
"""

class DesignMutator:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def mutate(self, parent_genome: SecondOrderGenome, component_to_mutate: str, 
               inspirations: List[SecondOrderGenome], first_order_plan: Optional[str] = None) -> SecondOrderGenome:
        
        # Use JSON for consistency with other agents, preserving logic
        insp_str = "\n".join([f"Inspiration Genome:\n{g.model_dump_json(indent=2)}" for g in inspirations])
        
        user_msg = f"""
        Parent Genome:
        {parent_genome.model_dump_json(indent=2)}
        
        Component to Mutate: {component_to_mutate}
        """
        
        if first_order_plan:
            user_msg += f"\n\nContext - First Order Bias Plan (Island):\n{first_order_plan}\n"
        
        user_msg += f"""
        Inspirations:
        {insp_str}
        
        Please provide a mutated version of the genome using SEARCH/REPLACE blocks.
        """
        response = self.llm.query(msg=user_msg, system_msg=DESIGN_MUTATOR_SYS_PROMPT)
        if response and response.content:
             # Apply the diff to the parent JSON string
             # The existing diff logic works on text, so it handles JSON strings fine
             patched_json = self._apply_diff(parent_genome.model_dump_json(indent=2), response.content)
             
             # Clean up potential artifacts if diff wasn't perfect, though JSON is fragile to diffs.
             # However, Search/Replace blocks are exact text matches, so if the LLM copies lines correctly, it works.
             return SecondOrderGenome.model_validate_json(patched_json)
        raise ValueError("Failed to mutate genome")

    def _apply_diff(self, original_text: str, diff_text: str) -> str:
        # Simple regex based patch application
        # This mirrors shinka logic simplified
        pattern = re.compile(
            r"(?:<DIFF>)?\s*<{7}\s*SEARCH\s*\n(.*?)\n\s*={7}\s*\n(.*?)\n\s*>{7}(?:\s*REPLACE)?\s*(?:</DIFF>)?",
            re.DOTALL,
        )
        matches = pattern.findall(diff_text)
        
        patched_text = original_text
        for search_block, replace_block in matches:
            if search_block in patched_text:
                patched_text = patched_text.replace(search_block, replace_block, 1)
            else:
                logger.warning(f"Could not find search block: {search_block[:50]}...")
        return patched_text


IMPLEMENTATION_AGENT_SYS_PROMPT = """
You are the Implementation Agent.

Role
You translate a SecondOrderGenome into concrete PyTorch code changes. Your responsibility is to ensure the implementation faithfully reflects the genome specification.

Target Component: {component}
You have been given a SPECIFIC REGION of the code to modify.
You must ONLY modify the code provided in the context.

Conceptual scope
- If Component is ALPHA (Architecture):
    - You are seeing the ALPHA region (Imports, Class definition, __init__, forward).
    - Modify structure, layers, and forward pass data flow.
- If Component is PHI (Objective):
    - You are seeing the PHI region (compute_loss, compute_metrics).
    - Modify loss logic and metrics tracking.
- If Component is OMEGA (Optimizer):
    - You are seeing the OMEGA region (compute_optimizer).
    - Modify optimizer configuration (e.g. Adam vs SGD, learning rates).
- If Component is All:
    - You are seeing the full file.
    - Perform all of the above (Alpha, Phi, Omega) based on the genome and update the templated logic.

Inputs
You will receive:
- A specific Code Region (subset of the full file)
- A SecondOrderGenome describing the desired design
- Optionally, error logs

Rules
- All edits must be done using valid XML-style SEARCH/REPLACE blocks.
- The SEARCH block must match the provided code region exactly.
- The REPLACE block must contain valid, runnable PyTorch code.
- **IMPORTANT**: Do NOT remove the region markers (# REGION_...) if they appear.
- Preserve style and indentation consistency.
- If previous errors are provided, prioritize fixing them.

Edit format
Use the following structure exactly (XML tags):

<DIFF>
<<<<<<< SEARCH
# Original code to find (must match exactly)
=======
# New replacement code
>>>>>>> REPLACE
</DIFF>
"""


class ImplementationAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def implement(
        self, 
        genome: SecondOrderGenome, 
        parent_code: str, 
        previous_errors: Optional[str] = None, 
        component: str = "All", 
        artifact_dir: Optional[str] = None,
        eval_script_content: Optional[str] = None
    ) -> str:
        
        # Determine strict region based on component
        region_tag_start = None
        region_tag_end = None
        
        if component == "Alpha":
            region_tag_start = "# REGION_ALPHA_START"
            region_tag_end = "# REGION_ALPHA_END"
        elif component == "Phi":
            region_tag_start = "# REGION_PHI_START"
            region_tag_end = "# REGION_PHI_END"
        elif component == "Omega":
            region_tag_start = "# REGION_OMEGA_START"
            region_tag_end = "# REGION_OMEGA_END"
            
        code_context = parent_code
        pre_context = ""
        post_context = ""
        
        # Try to extract region
        if region_tag_start and region_tag_end:
            pattern = re.compile(f"({re.escape(region_tag_start)}.*?{re.escape(region_tag_end)})", re.DOTALL)
            match = pattern.search(parent_code)
            if match:
                code_context = match.group(1)
                start_idx = match.start(1)
                end_idx = match.end(1)
                pre_context = parent_code[:start_idx]
                post_context = parent_code[end_idx:]
                logger.info(f"ImplementationAgent: Restricted editing to region {component} ({len(code_context)} chars)")
            else:
                logger.warning(f"ImplementationAgent: Region tags {region_tag_start}... not found. fallback to full code.")

        user_msg = f"""
        Target Genome Specification:
        {genome.model_dump_json(indent=2)}
        
        Code Region to Modify:
        ```python
        {code_context}
        ```
        """
        
        if eval_script_content:
            user_msg += f"\n\nReference Evaluation Script (evaluate.py):\n```python\n{eval_script_content}\n```\n"
            
        if previous_errors:
            user_msg += f"\nPrevious Implementation Errors:\n{previous_errors}"
        else:
            user_msg += "\nModify the code region to match the new genome."

        formatted_sys_msg = IMPLEMENTATION_AGENT_SYS_PROMPT.format(component=component)
        response = self.llm.query(msg=user_msg, system_msg=formatted_sys_msg)
        
        # Save interaction log
        if artifact_dir:
             try:
                 log_path = Path(artifact_dir) / "implementation_log.md"
                 with open(log_path, "w", encoding="utf-8") as f:
                     f.write(f"# Implementation Log\n\n## System Message\n{formatted_sys_msg}\n\n## User Message\n{user_msg}\n\n## Response\n{response.content if response else 'NO RESPONSE'}\n")
             except Exception as e:
                 logger.warning(f"Failed to save implementation log: {e}")

        if response and response.content:
            logger.debug(f"ImplementationAgent Raw Response ({component}):\n{response.content}")
            # Apply diff to the CONTEXT (partial code)
            patched_context = self._apply_diff(code_context, response.content)
            
            # Reassemble
            full_code = pre_context + patched_context + post_context
            return full_code
            
        raise ValueError("Failed to generate implementation code")
    
    def _apply_diff(self, original_text: str, diff_text: str) -> str:
        # Use robust regex handling for XML-like tags + git-style markers
        # Matches: <DIFF> ... <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE ... </DIFF>
        # or just the git-markers if the model forgets validity of tags
        pattern = re.compile(
            r"(?:<DIFF>)?\s*<{7}\s*SEARCH\s*\n(.*?)\n\s*={7}\s*\n(.*?)\n\s*>{7}(?:\s*REPLACE)?\s*(?:</DIFF>)?",
            re.DOTALL,
        )
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
                # Optional: fallback to stricter normalization comparison? 
                
        return patched_text

class ReflectionWriter:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def reflect(self, genome: SecondOrderGenome, metrics: Dict[str, float]) -> SecondOrderGenome:
        user_msg = f"""
        Genome:
        {genome.model_dump_json(indent=2)}
        
        Evaluation Metrics:
        {metrics}
        
        Update the 'reflection' field for biases.
        """
        
        response = self.llm.query(
            msg=user_msg, 
            system_msg=REFLECTION_WRITER_SYS_PROMPT,
            output_model=SecondOrderGenome
        )
        
        if response and response.content:
             logger.debug(f"ReflectionWriter Raw Response:\n{response.content}")
             content = _clean_json_content(response.content)
             if "```json" in content:
                 content = content.replace("```json", "").replace("```", "")
             elif "```" in content:
                 content = content.replace("```", "")
             content = content.strip()
             
             try:
                 return SecondOrderGenome.model_validate_json(content)
             except Exception as e:
                 logger.warning(f"Failed to parse reflected genome JSON: {e}")
                 
                 # Attempt repair
                 repaired_content = _repair_json(content)
                 try:
                     logger.info("Attempting to repair JSON...")
                     return SecondOrderGenome.model_validate_json(repaired_content)
                 except Exception as inner_e:
                     logger.error(f"Repair failed: {inner_e}")
                     logger.debug(f"Problematic JSON Content:\n{content}")
                        
        raise ValueError("Failed to reflect on genome")
