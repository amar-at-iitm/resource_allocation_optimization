"""Unit tests for demand uncertainty, resource disruptions, stochastic and robust models."""

import pytest
from pathlib import Path

from src.data.generator import generate_instance
from src.scenarios.demand_uncertainty import DemandScenarioGenerator
from src.scenarios.resource_disruption import ResourceDisruptionGenerator
from src.models.stochastic_model import TwoStageStochasticAllocationModel
from src.models.robust_model import RobustAllocationModel


def test_demand_scenario_generator():
    """Tests deterministic scaling and stochastic Monte Carlo demand generation."""
    base = generate_instance(n_resources=3, n_tasks=5, n_periods=5, seed=42)
    gen = DemandScenarioGenerator(base, seed=42)

    high = gen.generate_scaled_scenario("high_demand", demand_multiplier=1.25)
    assert high.scenario_name == "high_demand"
    for t_id, task in base.tasks.items():
        assert high.tasks[t_id].resource_demand >= task.resource_demand

    stoch_scens = gen.generate_stochastic_scenarios(n_scenarios=4, cv=0.15)
    assert len(stoch_scens) == 4
    for sc in stoch_scens:
        assert sc.periods == base.periods
        assert len(sc.tasks) == len(base.tasks)


def test_resource_disruption_generator():
    """Tests resource downtime and combined disruption generation."""
    base = generate_instance(n_resources=5, n_tasks=5, n_periods=5, seed=12)
    r_gen = ResourceDisruptionGenerator(base, seed=12)

    disrupted = r_gen.generate_disruption_scenario("disrupted", disruption_probability=0.40)
    assert disrupted.scenario_name == "disrupted"
    assert "disrupted_resources" in disrupted.metadata
    assert len(disrupted.metadata["disrupted_resources"]) > 0

    combined = r_gen.generate_combined_scenario("combined", demand_multiplier=1.25, disruption_probability=0.30)
    assert combined.scenario_name == "combined"
    assert combined.metadata["demand_multiplier"] == 1.25


def test_stochastic_allocation_model():
    """Tests two-stage stochastic model formulation and expected cost calculation."""
    base = generate_instance(n_resources=3, n_tasks=4, n_periods=4, seed=42)
    gen = DemandScenarioGenerator(base)
    scens = [
        gen.generate_scaled_scenario("normal", 1.0),
        gen.generate_scaled_scenario("high", 1.25),
        gen.generate_scaled_scenario("low", 0.8)
    ]
    model = TwoStageStochasticAllocationModel(scens, probabilities=[0.5, 0.25, 0.25])
    sol = model.solve(log_to_console=False)

    assert sol.status == "OPTIMAL"
    assert sol.objective_value is not None
    assert "expected_total_cost" in sol.summary_kpis
    assert len(model.scenario_solutions) == 3


def test_robust_allocation_model():
    """Tests robust min-max regret / cost model."""
    base = generate_instance(n_resources=3, n_tasks=4, n_periods=4, seed=42)
    d_gen = DemandScenarioGenerator(base)
    r_gen = ResourceDisruptionGenerator(base)
    scens = [
        base,
        d_gen.generate_scaled_scenario("high_demand", 1.30),
        r_gen.generate_disruption_scenario("disrupted", 0.50)
    ]
    model = RobustAllocationModel(scens)
    sol = model.solve(log_to_console=False)

    assert sol.status == "OPTIMAL"
    assert sol.objective_value is not None
    assert model.worst_case_scenario is not None
    assert sol.objective_value >= max(model.scenario_costs.values()) - 1e-2


def test_scenario_export(tmp_path):
    """Tests CSV export of demand and disruption scenarios."""
    base = generate_instance(n_resources=3, n_tasks=4, n_periods=4, seed=42)
    d_gen = DemandScenarioGenerator(base)
    r_gen = ResourceDisruptionGenerator(base)

    d_scens = d_gen.generate_standard_scenarios()
    d_files = d_gen.export_scenarios_csv(d_scens, tmp_path / "demand")
    assert (tmp_path / "demand" / "baseline_demand.csv").exists()
    assert (tmp_path / "demand" / "high_demand_demand.csv").exists()

    r_scens = r_gen.generate_standard_disruptions()
    r_files = r_gen.export_disruptions_csv(r_scens, tmp_path / "disruptions")
    assert (tmp_path / "disruptions" / "resource_disruption_capacity.csv").exists()

