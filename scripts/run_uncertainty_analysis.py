"""Runner script for Comprehensive Uncertainty & Resilience Experiments."""

import argparse
import sys
from pathlib import Path
import time
from typing import Any, Dict, List
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.config_loader import load_experiment_config, load_model_config
from src.data.generator import generate_instance
from src.scenarios.demand_uncertainty import DemandScenarioGenerator
from src.scenarios.resource_disruption import ResourceDisruptionGenerator
from src.models.deterministic_model import DeterministicAllocationModel
from src.models.multi_period_model import MultiPeriodSchedulingModel
from src.models.stochastic_model import TwoStageStochasticAllocationModel
from src.models.robust_model import RobustAllocationModel


def run_uncertainty_matrix(
    n_resources: int = 15,
    n_tasks: int = 20,
    n_periods: int = 10,
    seed: int = 42,
    output_dir: Path | str = "results/scenario_results"
) -> pd.DataFrame:
    """Executes the full uncertainty matrix across Deterministic, Stochastic, and Robust models."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model_config = load_model_config()

    print(f"\n=======================================================")
    print(f"Running Uncertainty & Operational Resilience Analysis")
    print(f"Setup: {n_resources} Resources, {n_tasks} Tasks, {n_periods} Periods, Seed={seed}")
    print(f"=======================================================")

    # 1. Generate Baseline and Scenarios
    print("Generating scenario instances...")
    base_inst = generate_instance(n_resources=n_resources, n_tasks=n_tasks, n_periods=n_periods, seed=seed)
    d_gen = DemandScenarioGenerator(base_inst, seed=seed)
    r_gen = ResourceDisruptionGenerator(base_inst, seed=seed)

    scenarios: Dict[str, Any] = {
        "Scenario_A_Normal": d_gen.generate_scaled_scenario("normal", demand_multiplier=1.0),
        "Scenario_B_HighDemand": d_gen.generate_scaled_scenario("high_demand", demand_multiplier=1.25),
        "Scenario_C_Disruption": r_gen.generate_disruption_scenario("resource_disruption", disruption_probability=0.20),
        "Scenario_D_Combined": r_gen.generate_combined_scenario("combined", demand_multiplier=1.25, disruption_probability=0.20)
    }

    results: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------
    # Part 1: Scenario-Specific Multi-Period Deterministic Models
    # -----------------------------------------------------------------
    print("\n--- Part 1: Solving Multi-Period Deterministic Models per Scenario ---")
    for s_name, scen_inst in scenarios.items():
        print(f"Solving {s_name}...")
        start = time.time()
        model = MultiPeriodSchedulingModel(scen_inst, config=model_config)
        sol = model.solve(time_limit=30.0, log_to_console=False)
        elap = time.time() - start

        results.append({
            "scenario": s_name,
            "model_type": "Deterministic_PerfectInformation",
            "solver_status": sol.status,
            "total_cost": sol.objective_value,
            "solve_time_seconds": round(elap, 3),
            "demand_satisfaction_percent": sol.summary_kpis.get("demand_satisfaction_percent", 0.0),
            "total_overtime_hours": sol.summary_kpis.get("total_overtime_hours", 0.0),
            "total_idle_hours": sol.summary_kpis.get("total_idle_hours", 0.0),
            "unmet_tasks": sol.summary_kpis.get("total_unmet_tasks", 0)
        })
        print(f" -> Cost: {sol.objective_value} | Satisfaction: {sol.summary_kpis.get('demand_satisfaction_percent')}%")

    # -----------------------------------------------------------------
    # Part 2: Two-Stage Stochastic Optimization Model
    # -----------------------------------------------------------------
    print("\n--- Part 2: Solving Two-Stage Stochastic Optimization Model ---")
    stoch_list = list(scenarios.values())
    stoch_model = TwoStageStochasticAllocationModel(stoch_list, config=model_config)
    start_stoch = time.time()
    stoch_sol = stoch_model.solve(time_limit=60.0, log_to_console=False)
    elap_stoch = time.time() - start_stoch

    results.append({
        "scenario": "MultiScenario_Expected",
        "model_type": "TwoStageStochastic",
        "solver_status": stoch_sol.status,
        "total_cost": stoch_sol.objective_value,
        "solve_time_seconds": round(elap_stoch, 3),
        "demand_satisfaction_percent": stoch_sol.summary_kpis.get("demand_satisfaction_percent", 100.0),
        "total_overtime_hours": stoch_sol.summary_kpis.get("total_overtime_hours", 0.0),
        "total_idle_hours": stoch_sol.summary_kpis.get("total_idle_hours", 0.0),
        "unmet_tasks": 0
    })
    print(f" -> Stochastic Expected Cost: {stoch_sol.objective_value} in {elap_stoch:.2f}s")

    # -----------------------------------------------------------------
    # Part 3: Robust Optimization Model (Min-Max Regret)
    # -----------------------------------------------------------------
    print("\n--- Part 3: Solving Robust Min-Max Optimization Model ---")
    rob_model = RobustAllocationModel(stoch_list, config=model_config)
    start_rob = time.time()
    rob_sol = rob_model.solve(time_limit=60.0, log_to_console=False)
    elap_rob = time.time() - start_rob

    results.append({
        "scenario": f"WorstCase_{rob_model.worst_case_scenario}",
        "model_type": "RobustMinMax",
        "solver_status": rob_sol.status,
        "total_cost": rob_sol.objective_value,
        "solve_time_seconds": round(elap_rob, 3),
        "demand_satisfaction_percent": rob_sol.summary_kpis.get("demand_satisfaction_percent", 100.0),
        "total_overtime_hours": 0.0,
        "total_idle_hours": 0.0,
        "unmet_tasks": 0
    })
    print(f" -> Robust Worst-Case Cost (theta): {rob_sol.objective_value} in {elap_rob:.2f}s")

    df = pd.DataFrame.from_records(results)
    csv_file = out_path / "uncertainty_matrix_results.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nSaved uncertainty matrix results to: {csv_file}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Comprehensive Uncertainty and Scenario Analysis")
    parser.add_argument("--resources", type=int, default=15, help="Number of resources")
    parser.add_argument("--tasks", type=int, default=20, help="Number of tasks")
    parser.add_argument("--periods", type=int, default=10, help="Planning horizon periods")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="results/scenario_results", help="Output directory")
    args = parser.parse_args()

    run_uncertainty_matrix(
        n_resources=args.resources,
        n_tasks=args.tasks,
        n_periods=args.periods,
        seed=args.seed,
        output_dir=args.output_dir
    )

