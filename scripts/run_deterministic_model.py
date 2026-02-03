"""Runner script for Small- and Medium-Scale Deterministic Optimization Experiments."""

import argparse
import sys
from pathlib import Path
import time
from typing import Any, Dict, List
import pandas as pd

# Add project root to Python module search path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.config_loader import load_experiment_config, load_model_config, load_solver_config
from src.data.generator import generate_instance
from src.models.deterministic_model import DeterministicAllocationModel
from src.models.multi_period_model import MultiPeriodSchedulingModel
from src.optimization.results import SolutionExtractor


def run_experiments(
    scale: str = "small",
    timelimit: float = 60.0,
    num_replications: int = 3,
    output_dir: Path | str = "results/optimization_results"
) -> pd.DataFrame:
    """Runs deterministic optimization experiments across seeds and records computational performance."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    exp_config = load_experiment_config()
    model_config = load_model_config()

    scales_to_run = ["small", "medium"] if scale == "all" else [scale]
    all_results: List[Dict[str, Any]] = []

    for sc_name in scales_to_run:
        sc_spec = exp_config.scales.get(sc_name)
        if not sc_spec:
            raise ValueError(f"Scale '{sc_name}' not defined in experiment_config.yaml")

        print(f"\n=======================================================")
        print(f"Running Experiments for Scale: {sc_name.upper()}")
        print(f"Parameters: {sc_spec.n_resources} Resources, {sc_spec.n_tasks} Tasks, {sc_spec.n_periods} Periods")
        print(f"Time Limit: {timelimit}s | Replications: {num_replications}")
        print(f"=======================================================")

        for repl in range(num_replications):
            seed = exp_config.experiment.random_seed + repl
            print(f"\n[Scale: {sc_name} | Seed: {seed}] Generating instance...")

            # Multi-period instance
            inst = generate_instance(
                n_resources=sc_spec.n_resources,
                n_tasks=sc_spec.n_tasks,
                n_periods=sc_spec.n_periods,
                seed=seed,
                instance_id=f"{sc_name}_seed{seed}"
            )

            # 1. Multi-Period Scheduling Model
            print(f"Solving Multi-Period Scheduling Model...")
            mp_model = MultiPeriodSchedulingModel(inst, config=model_config)
            mp_sol = mp_model.solve(time_limit=timelimit, log_to_console=False)

            mp_record = {
                "instance_id": inst.instance_id,
                "scale": sc_name,
                "model_type": "MultiPeriodScheduling",
                "seed": seed,
                "num_resources": sc_spec.n_resources,
                "num_tasks": sc_spec.n_tasks,
                "num_periods": sc_spec.n_periods,
                "solver_status": mp_sol.status,
                "is_feasible": mp_sol.is_feasible,
                "objective_cost": round(mp_sol.objective_value, 2) if mp_sol.objective_value else None,
                "solve_time_seconds": mp_sol.solve_time_seconds,
                "num_variables": mp_model.solver.num_variables,
                "num_constraints": mp_model.solver.num_constraints,
                "assigned_tasks": mp_sol.summary_kpis.get("total_assigned_tasks", 0),
                "unmet_tasks": mp_sol.summary_kpis.get("total_unmet_tasks", 0),
                "overtime_hours": mp_sol.summary_kpis.get("total_overtime_hours", 0.0),
                "idle_hours": mp_sol.summary_kpis.get("total_idle_hours", 0.0),
                "utilization_percent": mp_sol.summary_kpis.get("capacity_utilization_percent", 0.0),
                "demand_satisfaction_percent": mp_sol.summary_kpis.get("demand_satisfaction_percent", 0.0),
                "makespan_periods": mp_sol.summary_kpis.get("makespan_periods", 0)
            }
            all_results.append(mp_record)
            print(f" -> Status: {mp_sol.status} | Cost: {mp_sol.objective_value} | Time: {mp_sol.solve_time_seconds:.3f}s")

            # 2. Single-Period Baseline (1 period slice)
            print(f"Solving Single-Period Static Baseline Model...")
            sp_inst = generate_instance(
                n_resources=sc_spec.n_resources,
                n_tasks=min(sc_spec.n_tasks, 25),
                n_periods=1,
                seed=seed,
                instance_id=f"{sc_name}_sp_seed{seed}"
            )
            sp_model = DeterministicAllocationModel(sp_inst, config=model_config)
            sp_sol = sp_model.solve(time_limit=timelimit, log_to_console=False)

            sp_record = {
                "instance_id": sp_inst.instance_id,
                "scale": sc_name,
                "model_type": "DeterministicSinglePeriod",
                "seed": seed,
                "num_resources": sc_spec.n_resources,
                "num_tasks": len(sp_inst.tasks),
                "num_periods": 1,
                "solver_status": sp_sol.status,
                "is_feasible": sp_sol.is_feasible,
                "objective_cost": round(sp_sol.objective_value, 2) if sp_sol.objective_value else None,
                "solve_time_seconds": sp_sol.solve_time_seconds,
                "num_variables": sp_model.solver.num_variables,
                "num_constraints": sp_model.solver.num_constraints,
                "assigned_tasks": sp_sol.summary_kpis.get("total_assigned_tasks", 0),
                "unmet_tasks": sp_sol.summary_kpis.get("total_unmet_tasks", 0),
                "overtime_hours": sp_sol.summary_kpis.get("total_overtime_hours", 0.0),
                "idle_hours": sp_sol.summary_kpis.get("total_idle_hours", 0.0),
                "utilization_percent": sp_sol.summary_kpis.get("capacity_utilization_percent", 0.0),
                "demand_satisfaction_percent": sp_sol.summary_kpis.get("demand_satisfaction_percent", 0.0),
                "makespan_periods": 1
            }
            all_results.append(sp_record)
            print(f" -> Status: {sp_sol.status} | Cost: {sp_sol.objective_value} | Time: {sp_sol.solve_time_seconds:.3f}s")

    df = pd.DataFrame.from_records(all_results)
    
    # Save individual scale or all results CSV
    if scale == "small":
        csv_file = out_path / "small_scale_results.csv"
    elif scale == "medium":
        csv_file = out_path / "medium_scale_results.csv"
    else:
        csv_file = out_path / "all_deterministic_results.csv"

    df.to_csv(csv_file, index=False)
    print(f"\nSaved experimental results to: {csv_file}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Deterministic Optimization Experiments")
    parser.add_argument("--scale", type=str, choices=["small", "medium", "all"], default="small",
                        help="Scale of instances to evaluate")
    parser.add_argument("--timelimit", type=float, default=60.0,
                        help="Time limit in seconds for exact solver")
    parser.add_argument("--replications", type=int, default=3,
                        help="Number of random seed replications")
    parser.add_argument("--output_dir", type=str, default="results/optimization_results",
                        help="Directory to save output CSV results")
    args = parser.parse_args()

    run_experiments(
        scale=args.scale,
        timelimit=args.timelimit,
        num_replications=args.replications,
        output_dir=args.output_dir
    )
