"""Synthetic problem instance generator with controlled seeds and DAG precedence."""

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from src.data.schemas import ProblemInstance, Resource, Task


DEFAULT_SKILLS = [
    "Engineering",
    "QualityAssurance",
    "Design",
    "DevOps",
    "Operations"
]


def generate_instance(
    n_resources: int = 10,
    n_tasks: int = 15,
    n_periods: int = 7,
    seed: int = 42,
    skill_pool: Optional[List[str]] = None,
    precedence_density: float = 0.2,
    instance_id: Optional[str] = None,
    scenario_name: str = "baseline"
) -> ProblemInstance:
    """Generates a reproducible problem instance with guaranteed feasibility properties.
    
    Args:
        n_resources: Number of available resources.
        n_tasks: Number of tasks to schedule.
        n_periods: Number of planning periods (time horizon).
        seed: Random seed for reproducibility.
        skill_pool: List of possible skill names.
        precedence_density: Probability of a precedence edge between task i and j (for i < j).
        instance_id: Custom identifier for the instance.
        scenario_name: Name of the scenario (e.g., 'baseline', 'high_demand').
    
    Returns:
        A validated ProblemInstance object.
    """
    rng = np.random.default_rng(seed)
    skills = skill_pool if skill_pool is not None else DEFAULT_SKILLS
    num_skills = len(skills)

    if instance_id is None:
        instance_id = f"inst_R{n_resources}_T{n_tasks}_P{n_periods}_s{seed}"

    # 1. Generate Resources ensuring all skills are covered
    resource_skills: Dict[int, List[str]] = {i: [] for i in range(n_resources)}
    for sk_idx, sk in enumerate(skills):
        resource_skills[sk_idx % n_resources].append(sk)

    resources: Dict[str, Resource] = {}
    for i in range(n_resources):
        r_id = f"R_{i+1:03d}"
        r_skills = list(resource_skills[i])
        # Add additional random skills with probability
        for sk in skills:
            if sk not in r_skills and rng.random() > 0.7:
                r_skills.append(sk)
        if not r_skills:
            r_skills = [str(rng.choice(skills))]

        base_capacity = float(rng.choice([8.0, 8.0, 7.5, 8.5]))
        reg_cost = float(rng.uniform(80.0, 140.0))
        ot_cost = round(reg_cost * 1.5, 2)
        idle_cost = round(reg_cost * 0.25, 2)
        max_ot = round(base_capacity * 0.25, 2)

        resources[r_id] = Resource(
            id=r_id,
            name=f"Resource_{i+1}",
            skills=r_skills,
            capacity_per_period=round(base_capacity, 2),
            regular_cost_per_period=round(reg_cost, 2),
            overtime_cost_per_period=ot_cost,
            max_overtime_per_period=max_ot,
            idle_cost_per_period=idle_cost
        )

    # 2. Generate Tasks with DAG Precedence (only i -> j where i < j)
    tasks: Dict[str, Task] = {}
    for j in range(n_tasks):
        t_id = f"T_{j+1:03d}"
        required_skill = str(rng.choice(skills))
        max_duration = max(1, min(4, n_periods // 2))
        duration = int(rng.integers(1, max_duration + 1))
        demand = round(float(rng.uniform(1.0, 4.0)), 2)
        priority = int(rng.integers(1, 6))
        unmet_penalty = round(priority * float(rng.uniform(300.0, 600.0)), 2)

        # Precedences: choose from previous tasks to strictly guarantee DAG (acyclicity)
        preds: List[str] = []
        if j > 0 and precedence_density > 0.0:
            # Candidate predecessors are tasks 0 .. j-1
            for prev_idx in range(j):
                if rng.random() < precedence_density:
                    preds.append(f"T_{prev_idx+1:03d}")
            # Cap predecessors to avoid overly constrained graphs
            if len(preds) > 3:
                preds = preds[-3:]

        release_date = int(rng.integers(0, max(1, n_periods // 4))) if n_periods > 1 else 0
        extra_due = int(rng.integers(1, max(2, n_periods))) if n_periods > 1 else 1
        due_date = int(min(max(1, n_periods), release_date + duration + extra_due))

        tasks[t_id] = Task(
            id=t_id,
            name=f"Task_{j+1}",
            required_skill=required_skill,
            duration=duration,
            resource_demand=demand,
            priority=float(priority),
            unmet_penalty=unmet_penalty,
            predecessors=preds,
            release_date=release_date,
            due_date=due_date
        )

    metadata = {
        "n_resources": n_resources,
        "n_tasks": n_tasks,
        "n_periods": n_periods,
        "seed": seed,
        "precedence_density": precedence_density
    }

    return ProblemInstance(
        instance_id=instance_id,
        periods=n_periods,
        resources=resources,
        tasks=tasks,
        scenario_name=scenario_name,
        metadata=metadata
    )


def generate_standard_suite(output_dir: Optional[Path | str] = None) -> Dict[str, ProblemInstance]:
    """Generates and saves the standard small, medium, and large benchmark suite."""
    out = Path(output_dir) if output_dir else Path(__file__).resolve().parent.parent.parent / "data" / "generated"
    out.mkdir(parents=True, exist_ok=True)

    suite = {
        "small": generate_instance(n_resources=10, n_tasks=15, n_periods=7, seed=42, instance_id="small_benchmark"),
        "medium": generate_instance(n_resources=50, n_tasks=40, n_periods=30, seed=42, instance_id="medium_benchmark"),
        "large": generate_instance(n_resources=200, n_tasks=100, n_periods=60, seed=42, instance_id="large_benchmark")
    }

    for name, inst in suite.items():
        inst.save_json(out / f"{name}_instance.json")

    return suite

