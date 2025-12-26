import logging
import yaml
import re
from typing import List, Optional, Tuple, Any, Dict
from shinka.llm.llm import LLMClient
from shinka.sibs.schema import FirstOrderBiasPlan, SecondOrderGenome, FirstOrderBiasSpec, BaseSecondOrderGenome
from shinka.database import Program
from shinka.sibs.prompts import (
    FIRST_ORDER_PLANNER_SYS_PROMPT,
    DESIGN_MUTATOR_SYS_PROMPT,
    IMPLEMENTATION_AGENT_SYS_PROMPT,
    REFLECTION_WRITER_SYS_PROMPT,
    SECOND_ORDER_INITIALIZER_SYS_PROMPT,
    IMPLEMENTATION_AGENT_SYS_PROMPT
)

logger = logging.getLogger(__name__)

class FirstOrderPlanner:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def plan(self, island_id: int, task_description: str, dataset_type: str) -> FirstOrderBiasPlan:
        user_msg = f"Task Description: {task_description}\nDataset Type: {dataset_type}\nIsland ID: {island_id}\n\nGenerate a FirstOrderBiasPlan in JSON format."
        
        response = self.llm.query(
            msg=user_msg, 
            system_msg=FIRST_ORDER_PLANNER_SYS_PROMPT,
            output_model=FirstOrderBiasSpec
        )
        
        if response and response.content:
            logger.debug(f"FirstOrderPlanner Raw Response:\n{response.content}")
            spec = FirstOrderBiasSpec.model_validate_json(response.content)
            # Enrich with system fields
            return FirstOrderBiasPlan(
                first_order_version=0.1,
                island_id=island_id,
                **spec.model_dump()
            )

        raise ValueError("Failed to generate FirstOrderBiasPlan")


class SecondOrderInitializer:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def initialize(self, first_order_plan: FirstOrderBiasPlan, task_description: str) -> SecondOrderGenome:
        user_msg = f"Task Description: {task_description}\nFirst Order Plan:\n{first_order_plan.model_dump_json(indent=2)}\n\nGenerate an initial SecondOrderGenome in JSON format."
        
        response = self.llm.query(
            msg=user_msg, 
            system_msg=SECOND_ORDER_INITIALIZER_SYS_PROMPT,
            output_model=BaseSecondOrderGenome
        )
        
        if response and response.content:
            logger.debug(f"SecondOrderGenome Raw Response:\n{response.content}")
            spec = BaseSecondOrderGenome.model_validate_json(response.content)
            # Enrich with system fields
            return SecondOrderGenome(
                genome_version=0.1,
                genome_id="genome_0",
                island_id=first_order_plan.island_id,
                generation=0,
                parent_id=None,
                fitness=None,
                **spec.model_dump()
            )

        raise ValueError("Failed to generate SecondOrderGenome")


class DesignMutator:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def mutate(self, parent_genome: SecondOrderGenome, component_to_mutate: str, 
               inspirations: List[SecondOrderGenome], first_order_plan: Optional[str] = None) -> SecondOrderGenome:
        
        # Use JSON for consistency with other agents, preserving logic
        # Use BaseSecondOrderGenome for inspirations to hide system fields
        insp_str = "\n".join([f"Inspiration Genome:\n{BaseSecondOrderGenome(**g.model_dump()).model_dump_json(indent=2)}" for g in inspirations])
        
        base_parent = BaseSecondOrderGenome(**parent_genome.model_dump())
        
        user_msg = f"""
        Parent Genome:
        {base_parent.model_dump_json(indent=2)}
        
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
             # Apply the diff to the partial JSON string
             # The existing diff logic works on text, so it handles JSON strings fine
             patched_json = self._apply_diff(base_parent.model_dump_json(indent=2), response.content)
             
             # Clean up potential artifacts if diff wasn't perfect.
             new_spec = BaseSecondOrderGenome.model_validate_json(patched_json)
             
             # Reconstruct full genome with parent's system fields
             # Note: Typically mutation might imply a new ID or generation, but that logic might be external.
             # We preserve parent's ID/Island etc. for now as per "enrich after/before based on hard data" logic.
             return SecondOrderGenome(
                 genome_version=parent_genome.genome_version,
                 genome_id=parent_genome.genome_id,
                 island_id=parent_genome.island_id,
                 parent_id=parent_genome.parent_id,
                 generation=parent_genome.generation,
                 fitness=parent_genome.fitness,
                 **new_spec.model_dump()
             )
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


class ImplementationAgent:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def implement(
        self, 
        genome: SecondOrderGenome, 
        parent_code: str, 
        component: str, 
        previous_errors: Optional[str] = None,
        historical_errors: Optional[List[str]] = None,
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
        {BaseSecondOrderGenome(**genome.model_dump()).model_dump_json(indent=2)}
        
        Code Region to Modify:
        ```python
        {code_context}
        ```
        """
        
        if eval_script_content:
            user_msg += f"\n\nReference Evaluation Script (evaluate.py):\n```python\n{eval_script_content}\n```\n"
            
        if previous_errors:
            user_msg += f"\nPrevious Implementation Errors:\n{previous_errors}"

        if historical_errors and len(historical_errors) > 0:
            user_msg += f"\n\nHistorical Errors (Context from previous generations):\n"
            for i, err in enumerate(historical_errors):
                user_msg += f"--- History {i+1} ---\n{err}\n"
        else:
            if not previous_errors:
                user_msg += "\nModify the code region to match the new genome."
        
        # Add instruction if any errors present
        if previous_errors or (historical_errors and len(historical_errors) > 0):
             user_msg += "\n\nCRITICAL: You MUST analyze the above errors. If they relate to your current task, ensure your implementation fixes or avoids them."

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
        base_genome = BaseSecondOrderGenome(**genome.model_dump())
        user_msg = f"""
        Genome:
        {base_genome.model_dump_json(indent=2)}
        
        Evaluation Metrics:
        {metrics}
        
        Update the 'reflection' field for biases.
        """
        
        response = self.llm.query(
            msg=user_msg, 
            system_msg=REFLECTION_WRITER_SYS_PROMPT,
            output_model=BaseSecondOrderGenome
        )
        
        if response and response.content:
             logger.debug(f"ReflectionWriter Raw Response:\n{response.content}")
             content = response.content
             if "```json" in content:
                 content = content.replace("```json", "").replace("```", "")
             elif "```" in content:
                 content = content.replace("```", "")
             content = content.strip()
             
             try:
                 new_spec = BaseSecondOrderGenome.model_validate_json(content)
                 # Merge back
                 return SecondOrderGenome(
                    genome_version=genome.genome_version,
                    genome_id=genome.genome_id,
                    island_id=genome.island_id,
                    parent_id=genome.parent_id,
                    generation=genome.generation,
                    fitness=genome.fitness,
                    **new_spec.model_dump()
                 )
             except Exception as e:
                 logger.warning(f"Failed to parse reflected genome JSON: {e}")
                 
                 # Attempt repair
                 repaired_content = _repair_json(content)
                 try:
                     logger.info("Attempting to repair JSON...")
                     new_spec = BaseSecondOrderGenome.model_validate_json(repaired_content)
              
                     # Merge back
                     return SecondOrderGenome(
                        genome_version=genome.genome_version,
                        genome_id=genome.genome_id,
                        island_id=genome.island_id,
                        parent_id=genome.parent_id,
                        generation=genome.generation,
                        fitness=genome.fitness,
                        **new_spec.model_dump()
                     )
                 except Exception as inner_e:
                     logger.error(f"Repair failed: {inner_e}")
                     logger.debug(f"Problematic JSON Content:\n{content}")
                        
        raise ValueError("Failed to reflect on genome")
