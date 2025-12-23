import logging
import yaml
import re
from typing import List, Optional, Tuple, Any, Dict
from shinka.llm.llm import LLMClient
from shinka.sibs.schema import FirstOrderBiasPlan, SecondOrderGenome
from shinka.database import Program

logger = logging.getLogger(__name__)

# Placeholder system prompts - User will populate these
FIRST_ORDER_PLANNER_SYS_PROMPT = """You are the FirstOrderBiasPlanner.
Your goal is to generate a FirstOrderBiasPlan for an evolutionary island based on the task description.
Output YAML only."""

SECOND_ORDER_INITIALIZER_SYS_PROMPT = """You are the SecondOrderInitializer.
Your goal is to translate a FirstOrderBiasPlan into an initial SecondOrderGenome.
Output YAML only."""

DESIGN_MUTATOR_SYS_PROMPT = """You are the Design Mutator.
Your goal is to mutate a specific component (Alpha, Omega, or Phi) of a SecondOrderGenome.
The genome is a YAML structure.
You must use SEARCH/REPLACE blocks to edit the YAML.

Format:
<<<<<<< SEARCH
# Exact content to replace
=======
# New content
>>>>>>> REPLACE

Focus changes on the requested component.
"""

IMPLEMENTATION_AGENT_SYS_PROMPT = """You are the Implementation Agent.
Your goal is to modify the existing PyTorch code to match the new SecondOrderGenome specification.
You must use SEARCH/REPLACE blocks to edit the code.

<DIFF>
<<<<<<< SEARCH
# Original code to find and replace (must match exactly including indentation)
=======
# New replacement code
>>>>>>> REPLACE
</DIFF>

If there are previous errors, use them to guide your fix.
"""

REFLECTION_WRITER_SYS_PROMPT = """You are the Reflection Writer.
Your goal is to analyze the evaluation results of a genome and add reflections to its biases.
Explain why a bias succeeded or failed based on the metrics.
Output the updated SecondOrderGenome YAML.
"""


class FirstOrderPlanner:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def plan(self, island_id: int, task_description: str, dataset_type: str) -> FirstOrderBiasPlan:
        user_msg = f"Task Description: {task_description}\nDataset Type: {dataset_type}\nIsland ID: {island_id}\n\nGenerate a FirstOrderBiasPlan in YAML format."
        response = self.llm.query(msg=user_msg, system_msg=FIRST_ORDER_PLANNER_SYS_PROMPT)
        if response and response.content:
             # Basic cleanup to extract YAML if wrapped in markdown
            content = response.content.replace("```yaml", "").replace("```", "").strip()
            return FirstOrderBiasPlan.from_yaml(content)
        raise ValueError("Failed to generate FirstOrderBiasPlan")

class SecondOrderInitializer:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def initialize(self, first_order_plan: FirstOrderBiasPlan, task_description: str) -> SecondOrderGenome:
        user_msg = f"Task Description: {task_description}\nFirst Order Plan:\n{first_order_plan.to_yaml()}\n\nGenerate an initial SecondOrderGenome in YAML format."
        response = self.llm.query(msg=user_msg, system_msg=SECOND_ORDER_INITIALIZER_SYS_PROMPT)
        if response and response.content:
            content = response.content.replace("```yaml", "").replace("```", "").strip()
            return SecondOrderGenome.from_yaml(content)
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
             content = response.content.replace("```yaml", "").replace("```", "").strip()
             return SecondOrderGenome.from_yaml(content)
        raise ValueError("Failed to reflect on genome")
