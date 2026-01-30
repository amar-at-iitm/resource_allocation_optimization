"""Unit tests for operational, capacity, skill, and precedence constraints."""

import pytest
from src.data.generator import generate_instance
from src.models.deterministic_model import DeterministicAllocationModel
from src.models.multi_period_model import MultiPeriodSchedulingModel


def test_deterministic_capacity_and_skill_constraints():
    """Verifies capacity balances and skill constraints in single-period allocation."""
    inst = generate_instance(n_resources=5, n_tasks=8, n_periods=1, seed=10)
    model = DeterministicAllocationModel(inst)
    sol = model.solve(log_to_console=False)

    assert sol.is_feasible is True

    # 1. Check Skill Compatibility
    for assign in sol.assignments:
        res = inst.resources[assign.resource_id]
        task = inst.tasks[assign.task_id]
        assert task.required_skill in res.skills, (
            f"Resource {res.id} lacking skill {task.required_skill} assigned to task {task.id}"
        )

    # 2. Check Capacity Bounds
    workload_per_resource = {r_id: 0.0 for r_id in inst.resources}
    for assign in sol.assignments:
        workload_per_resource[assign.resource_id] += assign.assigned_work

    for r_id, total_work in workload_per_resource.items():
        res = inst.resources[r_id]
        ot = sol.overtime.get(r_id, {}).get(0, 0.0)
        assert ot <= res.max_overtime_per_period + 1e-4
        assert total_work <= res.capacity_per_period + ot + 1e-4


def test_multi_period_precedence_constraints():
    """Verifies that precedence relationships are strictly honored in multi-period scheduling."""
    inst = generate_instance(n_resources=4, n_tasks=6, n_periods=10, seed=42, precedence_density=0.4)
    model = MultiPeriodSchedulingModel(inst)
    sol = model.solve(log_to_console=False)

    assert sol.is_feasible is True

    # Check precedence among scheduled tasks
    for t_id, task in inst.tasks.items():
        if t_id in sol.task_start_times:
            task_start = sol.task_start_times[t_id]
            for pred_id in task.predecessors:
                if pred_id in inst.tasks:
                    assert pred_id in sol.task_completion_times, (
                        f"Predecessor {pred_id} must be completed before {t_id} can start."
                    )
                    pred_comp = sol.task_completion_times[pred_id]
                    assert task_start > pred_comp, (
                        f"Precedence violation: {pred_id} finished at {pred_comp}, but {t_id} started at {task_start}"
                    )

