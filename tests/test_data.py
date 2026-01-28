"""Unit tests for data schemas, synthetic generation, benchmark loader, and validation."""

import pytest
from pathlib import Path

from src.data.schemas import Resource, Task, ProblemInstance, AllocationSolution
from src.data.generator import generate_instance, generate_standard_suite
from src.data.loader import load_benchmark
from src.data.validation import validate_problem_instance, is_dag, InstanceValidationError


def test_resource_schema():
    """Tests Resource model creation and validations."""
    r = Resource(
        id="R_001",
        name="Worker 1",
        skills=["Python", "DevOps"],
        capacity_per_period=8.0,
        regular_cost_per_period=100.0,
        overtime_cost_per_period=150.0
    )
    assert r.id == "R_001"
    assert "Python" in r.skills
    assert r.max_overtime_per_period == 2.0

    # Empty skills should raise validation error
    with pytest.raises(ValueError):
        Resource(id="R_002", name="Bad Worker", skills=[])


def test_task_schema():
    """Tests Task model creation and validations."""
    t = Task(
        id="T_001",
        name="Task 1",
        required_skill="Python",
        duration=3,
        resource_demand=2.0
    )
    assert t.duration == 3
    assert t.priority == 1.0

    # Negative duration should fail
    with pytest.raises(ValueError):
        Task(id="T_002", name="Bad Task", required_skill="Python", duration=0)


def test_synthetic_generator():
    """Tests generation of reproducible instances."""
    inst1 = generate_instance(n_resources=6, n_tasks=12, n_periods=10, seed=123)
    inst2 = generate_instance(n_resources=6, n_tasks=12, n_periods=10, seed=123)

    assert len(inst1.resources) == 6
    assert len(inst1.tasks) == 12
    assert inst1.periods == 10
    # Strict determinism with same seed
    assert list(inst1.resources.keys()) == list(inst2.resources.keys())
    assert [t.required_skill for t in inst1.tasks.values()] == [t.required_skill for t in inst2.tasks.values()]

    # Validate the generated instance passes all integrity checks
    errors = validate_problem_instance(inst1)
    assert len(errors) == 0


def test_dag_validation():
    """Tests DAG cycle detection."""
    # Valid DAG
    tasks_valid = {
        "T1": Task(id="T1", name="1", required_skill="S", duration=1, resource_demand=1.0, predecessors=[]),
        "T2": Task(id="T2", name="2", required_skill="S", duration=1, resource_demand=1.0, predecessors=["T1"]),
        "T3": Task(id="T3", name="3", required_skill="S", duration=1, resource_demand=1.0, predecessors=["T2"]),
    }
    valid, order = is_dag(tasks_valid)
    assert valid is True
    assert order == ["T1", "T2", "T3"]

    # Cyclic graph: T1 -> T2 -> T3 -> T1
    tasks_cyclic = {
        "T1": Task(id="T1", name="1", required_skill="S", duration=1, resource_demand=1.0, predecessors=["T3"]),
        "T2": Task(id="T2", name="2", required_skill="S", duration=1, resource_demand=1.0, predecessors=["T1"]),
        "T3": Task(id="T3", name="3", required_skill="S", duration=1, resource_demand=1.0, predecessors=["T2"]),
    }
    valid_cyc, order_cyc = is_dag(tasks_cyclic)
    assert valid_cyc is False


def test_validation_catches_missing_skill():
    """Tests that a task requiring an unavailable skill triggers a validation error."""
    inst = generate_instance(n_resources=3, n_tasks=5, n_periods=7, seed=42)
    # Modify a task to require an unassigned skill
    inst.tasks["T_001"].required_skill = "NonExistentSkill"
    with pytest.raises(InstanceValidationError):
        validate_problem_instance(inst, raise_on_error=True)


def test_benchmark_loader():
    """Tests loading the sample PSPLIB/RCPSP benchmark file."""
    bench_file = Path("data/benchmarks/raw/sample_j30.sm")
    assert bench_file.exists(), "Sample benchmark file should exist"

    inst = load_benchmark(bench_file)
    assert len(inst.resources) >= 1
    assert len(inst.tasks) >= 1
    assert inst.periods >= 1

    # Ensure validation passes on parsed benchmark instance
    errors = validate_problem_instance(inst)
    assert len(errors) == 0


def test_standard_suite_generation(tmp_path):
    """Tests generating and saving the standard suite."""
    suite = generate_standard_suite(output_dir=tmp_path)
    assert "small" in suite
    assert "medium" in suite
    assert "large" in suite
    assert (tmp_path / "small_instance.json").exists()
    assert (tmp_path / "medium_instance.json").exists()
    assert (tmp_path / "large_instance.json").exists()

