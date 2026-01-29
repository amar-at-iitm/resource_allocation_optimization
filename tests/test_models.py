"""Integration tests for optimization models and results extraction."""

import pytest
from pathlib import Path

from src.data.generator import generate_instance
from src.data.loader import load_benchmark
from src.models.deterministic_model import DeterministicAllocationModel
from src.models.multi_period_model import MultiPeriodSchedulingModel
from src.optimization.results import SolutionExtractor


def test_deterministic_model_optimality():
    """Tests that the deterministic model solves and produces sensible metrics."""
    inst = generate_instance(n_resources=5, n_tasks=10, n_periods=1, seed=99)
    model = DeterministicAllocationModel(inst)
    sol = model.solve(log_to_console=False)

    assert sol.status == "OPTIMAL"
    assert sol.objective_value is not None
    assert sol.objective_value > 0.0
    assert sol.summary_kpis["demand_satisfaction_percent"] >= 0.0


def test_multi_period_model_solution_structure():
    """Tests multi-period scheduling model solution and results extractor."""
    inst = generate_instance(n_resources=4, n_tasks=6, n_periods=8, seed=77, precedence_density=0.2)
    model = MultiPeriodSchedulingModel(inst)
    sol = model.solve(log_to_console=False)

    assert sol.status == "OPTIMAL"
    assert sol.objective_value is not None

    # Test SolutionExtractor
    extractor = SolutionExtractor(sol, inst)
    assign_df = extractor.to_assignment_dataframe()
    assert isinstance(assign_df.shape[0], int)

    res_df = extractor.to_resource_kpi_dataframe()
    assert "utilization_percent" in res_df.columns
    assert len(res_df) == len(inst.resources) * inst.periods

    task_df = extractor.to_task_schedule_dataframe()
    assert "status" in task_df.columns
    assert len(task_df) == len(inst.tasks)


def test_solution_export(tmp_path):
    """Tests serializing and exporting solution artifacts."""
    inst = generate_instance(n_resources=3, n_tasks=5, n_periods=1, seed=42)
    model = DeterministicAllocationModel(inst)
    sol = model.solve(log_to_console=False)

    extractor = SolutionExtractor(sol, inst)
    files = extractor.export_results(output_dir=tmp_path, prefix="test")

    assert (tmp_path / "test_solution.json").exists()
    assert (tmp_path / "test_assignments.csv").exists()
    assert (tmp_path / "test_resource_kpis.csv").exists()
    assert (tmp_path / "test_task_schedule.csv").exists()

