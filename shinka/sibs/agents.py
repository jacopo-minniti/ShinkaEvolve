import logging
import re
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any
from shinka.llm.llm import LLMClient
from shinka.sibs.schema import FirstOrderBiasPlan, SecondOrderGenome, FirstOrderBiasSpec, BaseSecondOrderGenome
from shinka.database import Program
from shinka.sibs.prompts import (
    FIRST_ORDER_PLANNER_SYS_PROMPT,
    DESIGN_MUTATOR_SYS_PROMPT,
    IMPLEMENTATION_AGENT_SYS_PROMPT,
    REFLECTION_WRITER_SYS_PROMPT,
    SECOND_ORDER_INITIALIZER_SYS_PROMPT,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Common Utilities
# ============================================================================

def _parse_structured_output(content: str, model_class):
    """Parse LLM output, stripping markdown code blocks if present."""
    content = content.strip()
    
    # Strip markdown code blocks
    if "```json" in content:
        content = content.replace("```json", "").replace("```", "")
    elif "```" in content:
        content = content.replace("```", "")
    
    content = content.strip()
    return model_class.model_validate_json(content)


def _apply_diff(original_text: str, diff_text: str) -> str:
    """
    Apply SEARCH/REPLACE diff blocks to original text.
    Pattern: <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE
    """
    pattern = re.compile(
        r"(?:<DIFF>)?\s*<{7}\s*SEARCH\s*\n(.*?)\n\s*={7}\s*\n(.*?)\n\s*>{7}(?:\s*REPLACE)?\s*(?:</DIFF>)?",
        re.DOTALL,
    )
    matches = pattern.findall(diff_text)
    
    patched_text = original_text
    for search_block, replace_block in matches:
        # Normalize line endings
        search_block = search_block.replace('\r\n', '\n')
        replace_block = replace_block.replace('\r\n', '\n')
        
        if search_block.strip() == "":
            continue

        if search_block in patched_text:
            patched_text = patched_text.replace(search_block, replace_block, 1)
        else:
            logger.warning(f"Could not find search block in diff application")
            
    return patched_text


# ============================================================================
# Helper Functions
# ============================================================================

def _assign_bias_ids(spec: BaseSecondOrderGenome) -> None:
    """Assign unique IDs to each bias in the genome."""
    components = [
        ("Alpha", "alpha_a"),
        ("Phi", "phi_p"),
        ("Omega", "omega_o"),
    ]
    for attr, prefix in components:
        comp = getattr(spec.learner, attr, None)
        if not comp or not comp.biases:
            continue
        for idx, bias in enumerate(comp.biases, start=1):
            bias.bias_id = f"{prefix}{idx}"


def _assign_first_order_ids(spec: FirstOrderBiasSpec) -> None:
    """Assign unique IDs to each first-order requirement."""
    components = [
        ("alpha_requirements", "alpha_r"),
        ("phi_requirements", "phi_r"),
        ("omega_requirements", "omega_r"),
    ]
    for attr, prefix in components:
        reqs = getattr(spec, attr, None)
        if not reqs:
            continue
        for idx, req in enumerate(reqs, start=1):
            req.id = f"{prefix}{idx}"


# ============================================================================
# Agent Classes
# ============================================================================

class FirstOrderPlanner:
    """Generates first-order bias plans for islands."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def plan(self, island_id: int, task_description: str, dataset_type: str) -> FirstOrderBiasPlan:
        """Generate a first-order bias plan for an island."""
        user_msg = (
            f"Task Description: {task_description}\n"
            f"Dataset Type: {dataset_type}\n"
            f"Island ID: {island_id}\n\n"
            f"Generate a FirstOrderBiasPlan in JSON format."
        )
        
        response = self.llm.query(
            msg=user_msg, 
            system_msg=FIRST_ORDER_PLANNER_SYS_PROMPT,
            output_model=FirstOrderBiasSpec
        )
        
        if not response or not response.content:
            raise ValueError("Failed to generate FirstOrderBiasPlan")
        
        try:
            spec = _parse_structured_output(response.content, FirstOrderBiasSpec)
        except Exception as e:
            logger.error(f"Failed to parse FirstOrderBiasPlan: {e}")
            raise
        
        _assign_first_order_ids(spec)
        return FirstOrderBiasPlan(
            first_order_version=0.1,
            island_id=island_id,
            **spec.model_dump()
        )


class SecondOrderInitializer:
    """Initializes second-order genomes from first-order plans."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def initialize(self, first_order_plan: FirstOrderBiasPlan, task_description: str) -> SecondOrderGenome:
        """Generate an initial second-order genome from a first-order plan."""
        user_msg = (
            f"Task Description: {task_description}\n"
            f"First Order Plan:\n{first_order_plan.model_dump_json(indent=2)}\n\n"
            f"Generate an initial SecondOrderGenome in JSON format."
        )
        
        response = self.llm.query(
            msg=user_msg, 
            system_msg=SECOND_ORDER_INITIALIZER_SYS_PROMPT,
            output_model=BaseSecondOrderGenome
        )
        
        if not response or not response.content:
            raise ValueError("Failed to generate SecondOrderGenome")
        
        logger.info(f"SecondOrderInitializer received response of length {len(response.content)}")
        
        try:
            spec = _parse_structured_output(response.content, BaseSecondOrderGenome)
        except Exception as e:
            logger.error(f"Failed to parse SecondOrderGenome: {e}")
            raise
        
        # Check for empty Omega and provide default if needed
        if not spec.learner.Omega.biases:
            logger.error("=" * 80)
            logger.error("OMEGA BIASES MISSING - LLM FAILED TO GENERATE OPTIMIZER BIASES")
            logger.error("=" * 80)
            logger.error(f"Alpha biases: {len(spec.learner.Alpha.biases)}")
            logger.error(f"Phi biases: {len(spec.learner.Phi.biases)}")
            logger.error(f"Omega biases: {len(spec.learner.Omega.biases)} (EMPTY!)")
            
            # Debug logging removed as per user request
            logger.error("OMEGA BIASES MISSING - LLM FAILED TO GENERATE OPTIMIZER BIASES")
            logger.info("Adding default Omega bias to allow system to continue")
            
            # Add a default Omega bias
            from shinka.sibs.schema import BiasEntry
            default_omega = BiasEntry(
                acts_on="Omega",
                intention="Default optimizer configuration for stable training",
                metric_to_investigate=None,
                content="Use Adam optimizer with standard learning rate for reliable convergence"
            )
            spec.learner.Omega.biases = [default_omega]
            spec.learner.Omega.summary = "Default optimizer configuration (LLM failed to generate)"
        
        _assign_bias_ids(spec)
        logger.info(f"SecondOrderGenome created with {len(spec.learner.Alpha.biases)} Alpha, {len(spec.learner.Phi.biases)} Phi, {len(spec.learner.Omega.biases)} Omega biases")
        
        return SecondOrderGenome(
            genome_version=0.1,
            genome_id="genome_0",
            island_id=first_order_plan.island_id,
            generation=0,
            parent_id=None,
            fitness=None,
            **spec.model_dump()
        )


class DesignMutator:
    """Mutates genome designs using diff-based edits."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def mutate(
        self,
        parent_genome: SecondOrderGenome,
        component_to_mutate: str,
        archive_inspirations: List[Program],
        top_k_inspirations: List[Program],
        first_order_plan: Optional[str] = None,
        artifact_dir: Optional[str] = None,
    ) -> SecondOrderGenome:
        """Mutate a specific component of the genome."""
        
        def _format_inspirations(label: str, inspirations: List[Program]) -> str:
            """Format inspiration programs for the prompt."""
            if not inspirations:
                return f"{label}: none"
            blocks = []
            for prog in inspirations:
                if not prog.genome:
                    continue
                try:
                    genome = BaseSecondOrderGenome.model_validate_json(prog.genome)
                    genome_json = genome.model_dump_json(indent=2)
                except Exception:
                    genome_json = prog.genome
                blocks.append(
                    "\n".join([
                        f"- id: {prog.id}",
                        f"  generation: {prog.generation}",
                        f"  island: {prog.island_idx}",
                        f"  combined_score: {prog.combined_score}",
                        f"  public_metrics: {prog.public_metrics}",
                        "  genome:",
                        genome_json,
                    ])
                )
            return f"{label}:\n" + "\n\n".join(blocks) if blocks else f"{label}: none"

        insp_str = "\n\n".join([
            _format_inspirations("Contextual Inspirations (Elites & Random from Archive)", archive_inspirations),
            _format_inspirations("Global Top-K Inspirations (Excluding Context)", top_k_inspirations),
        ])
        
        base_parent = BaseSecondOrderGenome(**parent_genome.model_dump())
        
        user_msg = f"""
        Parent Genome:
        {base_parent.model_dump_json(indent=2)}
        
        Parent Reflection (Previous Step Analysis):
        {parent_genome.reflection or "None"}
        
        Component to Mutate: {component_to_mutate}
        """
        
        if first_order_plan:
            user_msg += f"\n\nContext - First Order Bias Plan (Island):\n{first_order_plan}\n"
        
        user_msg += f"""
        Inspirations:
        {insp_str}
        
        Please provide a mutated version of the genome using SEARCH/REPLACE blocks.
        """
        
        response = None
        try:
            response = self.llm.query(msg=user_msg, system_msg=DESIGN_MUTATOR_SYS_PROMPT)
        finally:
            # Save mutation log if artifact_dir provided
            if artifact_dir:
                try:
                    Path(artifact_dir).mkdir(parents=True, exist_ok=True)
                    log_path = Path(artifact_dir) / "mutation_log.md"
                    with open(log_path, "w", encoding="utf-8") as f:
                        f.write(
                            "# Mutation Log\n\n"
                            f"## System Message\n{DESIGN_MUTATOR_SYS_PROMPT}\n\n"
                            f"## User Message\n{user_msg}\n\n"
                            f"## Response\n{response.content if response else 'NO RESPONSE'}\n"
                        )
                except Exception as e:
                    logger.warning(f"Failed to save mutation log: {e}")
        
        if not response or not response.content:
            raise ValueError("Failed to mutate genome")
        
        # Apply diff to parent genome JSON
        patched_json = _apply_diff(base_parent.model_dump_json(indent=2), response.content)
        
        try:
            new_spec = BaseSecondOrderGenome.model_validate_json(patched_json)
        except Exception as e:
            logger.error(f"Failed to parse mutated genome: {e}")
            raise
        
        _assign_bias_ids(new_spec)
        return SecondOrderGenome(
            genome_version=parent_genome.genome_version,
            genome_id=parent_genome.genome_id,
            island_id=parent_genome.island_id,
            parent_id=parent_genome.parent_id,
            generation=parent_genome.generation,
            fitness=parent_genome.fitness,
            **new_spec.model_dump()
        )


class ImplementationAgent:
    """Translates genome specifications into PyTorch code."""
    
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
        """Generate code implementation for a genome specification."""
        
        # Determine region tags based on component
        region_tag_start = None
        region_tag_end = None
        
        # Check if we need to investigate metrics (requires editing REGION_METRICS)
        # We need to scan the genome component to see if metric_to_investigate is set
        has_metric_investigation = False
        comp_obj = getattr(genome.learner, component, None) if component != "All" else None
        if comp_obj and comp_obj.biases:
             for bias in comp_obj.biases:
                 if bias.metric_to_investigate:
                     has_metric_investigation = True
                     break
        
        region_metrics_start = "# REGION_METRICS_START"
        region_metrics_end = "# REGION_METRICS_END"
        
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
        
        # Helper to extract a region
        def _get_region(code, start_tag, end_tag):
             pattern = re.compile(f"({re.escape(start_tag)}.*?{re.escape(end_tag)})", re.DOTALL)
             match = pattern.search(code)
             if match:
                 return match.group(1)
             return None

        # Logic:
        # If component is "All", we pass full code (code_context).
        # If component is specific, we look for its region.
        # IF component is specific AND has_metric_investigation is True, we pass BOTH regions.
        
        if component != "All":
            target_region = _get_region(parent_code, region_tag_start, region_tag_end)
            if target_region:
                if has_metric_investigation:
                    # Fetch metrics region too
                    metrics_region = _get_region(parent_code, region_metrics_start, region_metrics_end)
                    if metrics_region:
                         code_context = f"{target_region}\n\n...\n\n{metrics_region}"
                         logger.info(f"Editing regions {component} + Metrics")
                    else:
                         code_context = target_region
                         logger.warning("Metrics region not found despite investigation request")
                else:
                    code_context = target_region
                    logger.info(f"Editing region {component} ({len(code_context)} chars)")
            else:
                # If target region not found, fallback to full code? Or error?
                # Fallback to full code is safer but modifies "All" essentially.
                logger.warning(f"Region tags {region_tag_start}... not found, using full code")
                code_context = parent_code
        
        # NOTE: Apply diff will apply to the FULL parent_code if we use _apply_diff(parent_code, diff).
        # We generally construct the prompt with `code_context`.
        # BUT we must apply the diff to `parent_code`.
        # Previous logic was: split parent code into pre/post and apply to middle.
        # That logic FAILs if we show disjoint regions.
        # New strategy: Show the regions in the prompt.
        # Apply the diff to the FULL PARENT CODE.
        # This works because SEARCH blocks will find the unique context wherever it is.
        # So we don't need pre_context/post_context splitting if we just apply to full file.
        
        user_msg = f"""
        Target Genome Specification:
        {BaseSecondOrderGenome(**genome.model_dump()).model_dump_json(indent=2)}
        
        Code Regions to Modify (You may see multiple disjoint regions separated by '...'):
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

        if component == "Phi":
            user_msg += (
                "\n\nCRITICAL REMINDER for Phi component:\n"
                "- compute_metrics must NEVER return 'loss' (already tracked)\n"
                "- compute_metrics must NEVER return 'test_accuracy' or 'success_rate' (that IS the fitness)\n"
                "- ONLY return auxiliary metrics from metric_to_investigate, or return None/empty dict\n"
                "- If metric_to_investigate is set, you MUST modify compute_metrics in REGION_METRICS.\n"
            )
        
        if previous_errors or (historical_errors and len(historical_errors) > 0):
             user_msg += "\n\nCRITICAL: You MUST analyze the above errors and fix or avoid them."

        formatted_sys_msg = IMPLEMENTATION_AGENT_SYS_PROMPT.format(component=component)
        response = None
        try:
            response = self.llm.query(msg=user_msg, system_msg=formatted_sys_msg)
        finally:
            # Save implementation log if artifact_dir provided
            if artifact_dir:
                try:
                    Path(artifact_dir).mkdir(parents=True, exist_ok=True)
                    log_path = Path(artifact_dir) / "implementation_log.md"
                    with open(log_path, "w", encoding="utf-8") as f:
                        f.write(
                            "# Implementation Log\n\n"
                            f"## System Message\n{formatted_sys_msg}\n\n"
                            f"## User Message\n{user_msg}\n\n"
                            f"## Response\n{response.content if response else 'NO RESPONSE'}\n"
                        )
                except Exception as e:
                    logger.warning(f"Failed to save implementation log: {e}")

        if not response or not response.content:
            raise ValueError("Failed to generate implementation code")
        
        # Apply diff to FULL CODE
        # Using full parent_code because SEARCH blocks will locate the correct regions
        patched_code = _apply_diff(parent_code, response.content)
        
        return patched_code


class ReflectionWriter:
    """Generates reflections comparing parent and child genomes."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def reflect(
        self,
        parent_genome: SecondOrderGenome,
        parent_metrics: Dict[str, Any],
        child_genome: SecondOrderGenome,
        child_metrics: Dict[str, Any],
        artifact_dir: Optional[str] = None
    ) -> SecondOrderGenome:
        """
        Generate a plain text reflection comparing parent and child genomes.
        Returns the child genome with reflection stored in metadata.
        """
        parent_base = BaseSecondOrderGenome(**parent_genome.model_dump())
        child_base = BaseSecondOrderGenome(**child_genome.model_dump())
        
        user_msg = f"""
        Parent Genome:
        {parent_base.model_dump_json(indent=2)}
        
        Parent Fitness/Metrics:
        {parent_metrics}
        
        Child Genome (after mutation):
        {child_base.model_dump_json(indent=2)}
        
        Child Fitness/Metrics:
        {child_metrics}
        
        Write a reflection analyzing what changed and why.
        """
        
        response = None
        try:
             response = self.llm.query(
                msg=user_msg, 
                system_msg=REFLECTION_WRITER_SYS_PROMPT
            )
        finally:
             if artifact_dir:
                try:
                    Path(artifact_dir).mkdir(parents=True, exist_ok=True)
                    log_path = Path(artifact_dir) / "reflection_log.md"
                    with open(log_path, "w", encoding="utf-8") as f:
                        f.write(
                            "# Reflection Log\n\n"
                            f"## System Message\n{REFLECTION_WRITER_SYS_PROMPT}\n\n"
                            f"## User Message\n{user_msg}\n\n"
                            f"## Response\n{response.content if response else 'NO RESPONSE'}\n"
                        )
                except Exception as e:
                    logger.warning(f"Failed to save reflection log: {e}")
        
        if not response or not response.content:
            logger.warning("Reflection generation failed - no response from LLM")
            self._save_reflection_diagnostic(
                parent_genome, parent_metrics, child_genome, child_metrics,
                user_msg, None, "NO_RESPONSE"
            )
            return child_genome
        
        # Parse plain text reflection
        reflection_text = response.content.strip()
        
        # Strip markdown if present
        if "```" in reflection_text:
            reflection_text = reflection_text.replace("```", "").strip()
        
        # Check if reflection is meaningful (not just empty or very short)
        if len(reflection_text) < 20:
            logger.warning(f"Reflection is too short ({len(reflection_text)} chars): '{reflection_text}'")
            self._save_reflection_diagnostic(
                parent_genome, parent_metrics, child_genome, child_metrics,
                user_msg, response.content, "TOO_SHORT"
            )
        
        # Store reflection in child genome (top-level)
        child_genome.reflection = reflection_text
        # child_genome.metadata["reflection"] = reflection_text # Deprecated
        logger.info(f"Reflection generated successfully ({len(reflection_text)} chars)")
        
        return child_genome
    
    def _save_reflection_diagnostic(
        self,
        parent_genome: SecondOrderGenome,
        parent_metrics: Dict[str, Any],
        child_genome: SecondOrderGenome,
        child_metrics: Dict[str, Any],
        user_msg: str,
        llm_response: Optional[str],
        issue: str
    ):
        """Save diagnostic information when reflection fails or is problematic."""
        debug_dir = Path("debug_reflection_failures")
        debug_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_file = debug_dir / f"reflection_{issue}_{timestamp}.md"
        
        with open(debug_file, "w", encoding="utf-8") as f:
            f.write("# Reflection Diagnostic Report\n\n")
            f.write(f"**Timestamp**: {timestamp}\n")
            f.write(f"**Issue**: {issue}\n\n")
            
            f.write("---\n\n")
            f.write("## Parent Genome\n\n")
            f.write("```json\n")
            f.write(BaseSecondOrderGenome(**parent_genome.model_dump()).model_dump_json(indent=2))
            f.write("\n```\n\n")
            
            f.write("## Parent Metrics\n\n")
            f.write("```json\n")
            f.write(str(parent_metrics))
            f.write("\n```\n\n")
            
            f.write("---\n\n")
            f.write("## Child Genome\n\n")
            f.write("```json\n")
            f.write(BaseSecondOrderGenome(**child_genome.model_dump()).model_dump_json(indent=2))
            f.write("\n```\n\n")
            
            f.write("## Child Metrics\n\n")
            f.write("```json\n")
            f.write(str(child_metrics))
            f.write("\n```\n\n")
            
            f.write("---\n\n")
            f.write("## User Message (Prompt)\n\n")
            f.write("```\n")
            f.write(user_msg)
            f.write("\n```\n\n")
            
            f.write("---\n\n")
            f.write("## LLM Response\n\n")
            if llm_response:
                f.write(f"**Length**: {len(llm_response)} chars\n\n")
                f.write("```\n")
                f.write(llm_response)
                f.write("\n```\n")
            else:
                f.write("*No response received from LLM*\n")
        
        logger.error(f"Saved reflection diagnostic to: {debug_file}")

