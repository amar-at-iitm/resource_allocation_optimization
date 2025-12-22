"""Data schemas and domain models for Multi-Period Resource Allocation and Scheduling."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from pydantic import BaseModel, Field, field_validator, model_validator


class Resource(BaseModel):
    """Resource entity (worker, machine, processing unit)."""
    id: str
    name: str
    skills: List[str] = Field(default_factory=list)
    capacity_per_period: float = Field(default=8.0, gt=0.0)
    regular_cost_per_period: float = Field(default=100.0, ge=0.0)
    overtime_cost_per_period: float = Field(default=150.0, ge=0.0)
    max_overtime_per_period: float = Field(default=2.0, ge=0.0)
    idle_cost_per_period: float = Field(default=25.0, ge=0.0)

    @field_validator("skills")
    @classmethod
    def validate_skills(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("Resource must possess at least one skill.")
        return [s.strip() for s in v]


class Task(BaseModel):
    """Task or activity entity requiring resources over time."""
    id: str
    name: str
    required_skill: str
    duration: int = Field(default=1, ge=1)
    resource_demand: float = Field(default=1.0, gt=0.0)
    priority: float = Field(default=1.0, gt=0.0)
    unmet_penalty: float = Field(default=1000.0, ge=0.0)
    predecessors: List[str] = Field(default_factory=list)
    release_date: int = Field(default=0, ge=0)
    due_date: Optional[int] = Field(default=None, ge=1)

    @field_validator("required_skill")
    @classmethod
    def validate_skill(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Task must specify a non-empty required_skill.")
        return s


class ProblemInstance(BaseModel):
    """Complete problem instance specification."""
    instance_id: str
    periods: int = Field(default=7, ge=1)
    resources: Dict[str, Resource]
    tasks: Dict[str, Task]
    scenario_name: str = "baseline"
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_instance(self) -> ProblemInstance:
        if not self.resources:
            raise ValueError("ProblemInstance must contain at least one resource.")
        if not self.tasks:
            raise ValueError("ProblemInstance must contain at least one task.")
        return self

    def get_skill_coverage(self) -> Dict[str, List[str]]:
        """Returns mapping from each skill to resource IDs possessing it."""
        coverage: Dict[str, List[str]] = {}
        for r_id, r in self.resources.items():
            for sk in r.skills:
                coverage.setdefault(sk, []).append(r_id)
        return coverage

    def save_json(self, file_path: Path | str) -> None:
        """Serializes problem instance to a JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load_json(cls, file_path: Path | str) -> ProblemInstance:
        """Loads a problem instance from a JSON file."""
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


class TaskAssignment(BaseModel):
    """Detailed assignment record for a task-resource-period triple."""
    resource_id: str
    task_id: str
    period: int
    assigned_work: float = Field(..., ge=0.0)


class AllocationSolution(BaseModel):
    """Output solution produced by an exact solver or metaheuristic."""
    instance_id: str
    status: str
    is_feasible: bool = True
    objective_value: Optional[float] = None
    solve_time_seconds: float = 0.0
    optimality_gap: Optional[float] = None
    assignments: List[TaskAssignment] = Field(default_factory=list)
    overtime: Dict[str, Dict[int, float]] = Field(default_factory=dict)
    idle: Dict[str, Dict[int, float]] = Field(default_factory=dict)
    unmet_demand: Dict[str, Dict[int, float]] = Field(default_factory=dict)
    task_start_times: Dict[str, int] = Field(default_factory=dict)
    task_completion_times: Dict[str, int] = Field(default_factory=dict)
    summary_kpis: Dict[str, Any] = Field(default_factory=dict)

    def save_json(self, file_path: Path | str) -> None:
        """Serializes solution to JSON."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load_json(cls, file_path: Path | str) -> AllocationSolution:
        """Loads solution from JSON."""
        path = Path(file_path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

