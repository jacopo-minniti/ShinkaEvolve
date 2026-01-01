from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import json

class BiasRequirement(BaseModel):
    id: Optional[str] = None
    property: str
    why_task_requires_it: str

class FirstOrderBiasSpec(BaseModel):
    task_summary: str = Field(min_length=1)
    alpha_requirements: List[BiasRequirement] = Field(min_length=1)
    omega_requirements: List[BiasRequirement] = Field(min_length=1)
    phi_requirements: List[BiasRequirement] = Field(min_length=1)

class FirstOrderBiasPlan(FirstOrderBiasSpec):
    first_order_version: float = 0.1
    island_id: int = 0


class MetricToInvestigate(BaseModel):
    name: str = "metric"
    expectation: str = "increase" # "increase", "decrease"
    rationale: str = ""

class BiasEntry(BaseModel):
    bias_id: Optional[str] = None
    acts_on: str # "Alpha", "Omega", "Phi"
    intention: str
    metric_to_investigate: Optional[MetricToInvestigate] = None
    content: str = "" # The actual second-order bias text

class ComponentParams(BaseModel):
    summary: str = ""
    biases: List[BiasEntry] = Field(default_factory=list)

class LearnerSpec(BaseModel):
    Alpha: ComponentParams = Field(default_factory=lambda: ComponentParams(), alias="alpha")
    Omega: ComponentParams = Field(default_factory=lambda: ComponentParams(), alias="omega")
    Phi: ComponentParams = Field(default_factory=lambda: ComponentParams(), alias="phi")

    class Config:
        populate_by_name = True

class BaseSecondOrderGenome(BaseModel):
    high_level_description: str = ""
    learner: LearnerSpec = Field(default_factory=lambda: LearnerSpec())

class SecondOrderGenome(BaseSecondOrderGenome):
    genome_version: float = 0.1
    genome_id: str = "genome_0"
    island_id: int = 0
    parent_id: Optional[str] = None
    generation: int = 0
    generation: int = 0
    fitness: Optional[float] = None
    metrics: Optional[Dict[str, Any]] = None
    reflection: Optional[str] = None # Reflection on the genome performance
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Stores other metadata
