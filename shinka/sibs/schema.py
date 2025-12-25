from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import json

class BiasRequirement(BaseModel):
    id: str
    property: str
    why_task_requires_it: str

class FirstOrderBiasPlan(BaseModel):
    first_order_version: float = 0.1
    island_id: int = 0
    task_summary: str = ""
    alpha_requirements: List[BiasRequirement] = Field(default_factory=list)
    omega_requirements: List[BiasRequirement] = Field(default_factory=list)
    phi_requirements: List[BiasRequirement] = Field(default_factory=list)


class MetricToInvestigate(BaseModel):
    name: str = "metric"
    expectation: str = "increase" # "increase", "decrease"
    rationale: str = ""

class BiasEntry(BaseModel):
    bias_id: str
    acts_on: str # "Alpha", "Omega", "Phi"
    intention: str
    metric_to_investigate: MetricToInvestigate
    reflection: Optional[str] = None
    content: str = "" # The actual second-order bias text

class ComponentParams(BaseModel):
    summary: str = ""
    biases: List[BiasEntry] = Field(default_factory=list)

class LearnerSpec(BaseModel):
    Alpha: ComponentParams = Field(default_factory=lambda: ComponentParams())
    Omega: ComponentParams = Field(default_factory=lambda: ComponentParams())
    Phi: ComponentParams = Field(default_factory=lambda: ComponentParams())

class SecondOrderGenome(BaseModel):
    genome_version: float = 0.1
    genome_id: str = "genome_0"
    island_id: int = 0
    parent_id: Optional[str] = None
    generation: int = 0
    high_level_description: str = ""
    fitness: Optional[Dict[str, float]] = None
    learner: LearnerSpec = Field(default_factory=lambda: LearnerSpec())
