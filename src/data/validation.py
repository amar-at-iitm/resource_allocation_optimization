"""Data validation and consistency checks for ProblemInstance models."""

from collections import deque
from typing import Dict, List, Set, Tuple
from src.data.schemas import ProblemInstance, Task


class InstanceValidationError(Exception):
    """Raised when a ProblemInstance fails structural or mathematical validity checks."""
    pass


def is_dag(tasks: Dict[str, Task]) -> Tuple[bool, List[str]]:
    """Checks whether the task precedence relationships form a valid DAG.
    
    Returns:
        (is_valid_dag, topological_order_if_valid)
    """
    in_degree: Dict[str, int] = {t_id: 0 for t_id in tasks}
    adjacency: Dict[str, List[str]] = {t_id: [] for t_id in tasks}

    for t_id, task in tasks.items():
        for pred in task.predecessors:
            if pred in tasks:
                adjacency[pred].append(t_id)
                in_degree[t_id] += 1

    queue = deque([t_id for t_id, deg in in_degree.items() if deg == 0])
    topo_order: List[str] = []

    while queue:
        curr = queue.popleft()
        topo_order.append(curr)
        for neighbor in adjacency[curr]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(topo_order) == len(tasks):
        return True, topo_order
    return False, []


def validate_problem_instance(instance: ProblemInstance, raise_on_error: bool = True) -> List[str]:
    """Validates structural integrity, feasibility, and mathematical consistency.
    
    Args:
        instance: The ProblemInstance to validate.
        raise_on_error: If True, raises InstanceValidationError on any failure.
        
    Returns:
        List of validation error messages (empty if valid).
    """
    errors: List[str] = []

    # 1. Check Resources
    if not instance.resources:
        errors.append("Instance has no resources.")

    for r_id, res in instance.resources.items():
        if res.capacity_per_period <= 0:
            errors.append(f"Resource {r_id} has non-positive capacity: {res.capacity_per_period}")
        if res.regular_cost_per_period < 0:
            errors.append(f"Resource {r_id} has negative regular cost: {res.regular_cost_per_period}")
        if res.overtime_cost_per_period < res.regular_cost_per_period:
            errors.append(
                f"Resource {r_id} overtime cost ({res.overtime_cost_per_period}) is lower than regular cost ({res.regular_cost_per_period})"
            )
        if not res.skills:
            errors.append(f"Resource {r_id} has empty skills list.")

    # 2. Check Tasks
    if not instance.tasks:
        errors.append("Instance has no tasks.")

    skill_coverage = instance.get_skill_coverage()

    for t_id, task in instance.tasks.items():
        if task.duration < 1:
            errors.append(f"Task {t_id} duration must be >= 1, got {task.duration}")
        if task.resource_demand <= 0:
            errors.append(f"Task {t_id} demand must be > 0, got {task.resource_demand}")
        if task.required_skill not in skill_coverage or len(skill_coverage[task.required_skill]) == 0:
            errors.append(f"Task {t_id} requires skill '{task.required_skill}' which is not possessed by any resource.")

        for pred in task.predecessors:
            if pred not in instance.tasks:
                errors.append(f"Task {t_id} references non-existent predecessor '{pred}'")
            if pred == t_id:
                errors.append(f"Task {t_id} lists itself as predecessor.")

    # 3. Check Precedence DAG Acyclicity
    acyclic, topo = is_dag(instance.tasks)
    if not acyclic:
        errors.append("Precedence relations contain a circular dependency (not a DAG).")

    # 4. Check Planning Horizon
    if instance.periods < 1:
        errors.append(f"Planning horizon periods must be >= 1, got {instance.periods}")

    if errors and raise_on_error:
        raise InstanceValidationError(f"Instance validation failed with {len(errors)} error(s):\n" + "\n".join(f" - {e}" for e in errors))

    return errors

