from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import yaml

@dataclass
class BiasRequirement:
    id: str
    property: str
    why_task_requires_it: str

@dataclass
class FirstOrderBiasPlan:
    first_order_version: float
    island_id: int
    task_summary: str
    alpha_requirements: List[BiasRequirement]
    omega_requirements: List[BiasRequirement]
    phi_requirements: List[BiasRequirement]

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "FirstOrderBiasPlan":
        data = yaml.safe_load(yaml_str)
        return cls(
            first_order_version=data.get("first_order_version", 0.1),
            island_id=data.get("island_id", 0),
            task_summary=data.get("task_summary", ""),
            alpha_requirements=[BiasRequirement(**item) for item in data.get("alpha_requirements", [])],
            omega_requirements=[BiasRequirement(**item) for item in data.get("omega_requirements", [])],
            phi_requirements=[BiasRequirement(**item) for item in data.get("phi_requirements", [])]
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FirstOrderBiasPlan":
        return cls(
            first_order_version=data.get("first_order_version", 0.1),
            island_id=data.get("island_id", 0),
            task_summary=data.get("task_summary", ""),
            alpha_requirements=[BiasRequirement(**item) for item in data.get("alpha_requirements", [])],
            omega_requirements=[BiasRequirement(**item) for item in data.get("omega_requirements", [])],
            phi_requirements=[BiasRequirement(**item) for item in data.get("phi_requirements", [])]
        )

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "FirstOrderBiasPlan":
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)
    
    def to_yaml(self) -> str:
        data = {
            "first_order_version": self.first_order_version,
            "island_id": self.island_id,
            "task_summary": self.task_summary,
            "alpha_requirements": [req.__dict__ for req in self.alpha_requirements],
            "omega_requirements": [req.__dict__ for req in self.omega_requirements],
            "phi_requirements": [req.__dict__ for req in self.phi_requirements],
        }
        return yaml.dump(data, sort_keys=False)


@dataclass
class MetricToInvestigate:
    name: str
    expectation: str # "increase", "decrease"
    rationale: str

@dataclass
class BiasEntry:
    bias_id: str
    acts_on: str # "Alpha", "Omega", "Phi"
    intention: str
    metric_to_investigate: MetricToInvestigate
    reflection: Optional[str] = None
    content: str = "" # The actual second-order bias text

@dataclass
class ComponentParams:
    summary: str
    biases: List[BiasEntry]

@dataclass
class LearnerSpec:
    Alpha: ComponentParams
    Omega: ComponentParams
    Phi: ComponentParams

@dataclass
class SecondOrderGenome:
    genome_version: float
    genome_id: str
    island_id: int
    parent_id: Optional[str]
    generation: int
    high_level_description: str
    fitness: Optional[Dict[str, float]]
    learner: LearnerSpec

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SecondOrderGenome":
        def parse_biases(biases_data):
            return [
                BiasEntry(
                    bias_id=b.get("bias_id"),
                    acts_on=b.get("acts_on"),
                    intention=b.get("intention"),
                    metric_to_investigate=MetricToInvestigate(**b.get("metric_to_investigate", {})),
                    reflection=b.get("reflection"),
                    content=b.get("content")
                ) for b in biases_data
            ]

        def parse_component(comp_data):
            return ComponentParams(
                summary=comp_data.get("summary", ""),
                biases=parse_biases(comp_data.get("biases", []))
            )

        learner_data = data.get("learner", {})
        learner = LearnerSpec(
            Alpha=parse_component(learner_data.get("Alpha", {})),
            Omega=parse_component(learner_data.get("Omega", {})),
            Phi=parse_component(learner_data.get("Phi", {}))
        )

        return cls(
            genome_version=data.get("genome_version", 0.1),
            genome_id=data.get("genome_id"),
            island_id=data.get("island_id"),
            parent_id=data.get("parent_id"),
            generation=data.get("generation"),
            high_level_description=data.get("high_level_description", ""),
            fitness=data.get("fitness"),
            learner=learner
        )

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "SecondOrderGenome":
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)

    def to_yaml(self) -> str:
        def dict_biases(biases):
             return [
                {
                    "bias_id": b.bias_id,
                    "acts_on": b.acts_on,
                    "intention": b.intention,
                    "metric_to_investigate": b.metric_to_investigate.__dict__,
                    "reflection": b.reflection,
                    "content": b.content
                } for b in biases
            ]
        
        data = {
            "genome_version": self.genome_version,
            "genome_id": self.genome_id,
            "island_id": self.island_id,
            "parent_id": self.parent_id,
            "generation": self.generation,
            "high_level_description": self.high_level_description,
            "fitness": self.fitness,
            "learner": {
                "Alpha": {
                    "summary": self.learner.Alpha.summary,
                    "biases": dict_biases(self.learner.Alpha.biases)
                },
                "Omega": {
                    "summary": self.learner.Omega.summary,
                    "biases": dict_biases(self.learner.Omega.biases)
                },
                "Phi": {
                    "summary": self.learner.Phi.summary,
                    "biases": dict_biases(self.learner.Phi.biases)
                }
            }
        }
        return yaml.dump(data, sort_keys=False)
